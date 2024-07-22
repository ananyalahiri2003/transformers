# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TF 2.0 SigLIP model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling

# Public API
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SiglipConfig"
_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"


LARGE_NEGATIVE = 1e-8    # CHECK


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


@dataclass
class TFSiglipOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`TFSiglipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`TFSiglipVisionModel`].
        text_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFSiglipTextModel`].
        vision_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFSiglipVisionModel`].
    """

    loss: Optional[tf.Tensor] = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TFSiglipVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: SiglipVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.config = config

        self.patch_embedding = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )

    def build(self, input_shape: tf.TensorShape = None):
        factor = self.config.initializer_factor

        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,),
            initializer=get_initializer(self.embed_dim ** -0.5 * factor),
            trainable=True,
            name="class_embedding",
        )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.config.initializer_range * factor),
                trainable=True,
                name="embeddings",
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "patch_embedding", None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, self.config.num_channels])

    def interpolate_pos_encoding(self, embeddings, height, width, position_embedding):
        """
        This method is an adapted method for SigLIP (due to SigLIP not having class embedding unlike other ViTs)
        that allows the model to interpolate the pre-trained position encodings such that it can be usable on
        higher resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = tf.expand_dims(position_embedding, 0)
        num_patches = tf.shape(embeddings)[1]
        num_positions = tf.shape(position_embeddings)[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = tf.shape(embeddings)[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = tf.reshape(position_embeddings,
                                     (1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim))
        patch_pos_embed = tf.transpose(patch_pos_embed, perm=[0, 3, 1, 2])
        patch_pos_embed = tf.image.resize(patch_pos_embed, size=(int(height), int(width)), method='bicubic')

        if int(height) != tf.shape(patch_pos_embed)[-2] or int(width) != tf.shape(patch_pos_embed)[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")

        patch_pos_embed = tf.transpose(patch_pos_embed, perm=[0, 2, 3, 1])
        patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, dim))

        return patch_pos_embed

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding=False) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""

        batch_size, num_channels, height, width = shape_list(pixel_values)
        # When running on CPU, `tf.nn.conv2d` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))

        if interpolate_pos_encoding:
            patch_embeds = patch_embeds + self.interpolate_pos_encoding(patch_embeds, height, width)
        else:
            patch_embeds = patch_embeds + self.position_embedding(self.position_ids)
        return patch_embeds


class TFSiglipTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: SiglipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size

        self.config = config

    def build(self, input_shape: tf.TensorShape = None):
        with tf.name_scope("token_embedding"):
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFSiglipAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SiglipConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.q_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        self.dropout = keras.layers.Dropout(rate=config.attention_dropout)

        self.out_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = shape_list(hidden_states)
        query_states = self.q_proj(inputs=hidden_states)
        key_states = self.k_proj(inputs=hidden_states)
        value_states = self.v_proj(inputs=hidden_states)

        query_states = self.transpose_for_scores(query_states, batch_size)
        key_states = self.transpose_for_scores(key_states, batch_size)
        value_states = self.transpose_for_scores(value_states, batch_size)

        k_v_seq_len = shape_list(key_states)[2]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if shape_list(attention_scores) != (batch_size, self.num_attention_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attention_scores.size()}"
            )

        if attention_mask is not None:
            if shape_list(attention_mask) != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, "
                    f"but is {shape_list(attention_mask)}"
                )
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=_attention_probs, training=training)
        attention_output = tf.matmul(attention_probs, value_states)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, q_len, self.embed_dim))

        attention_output = self.out_proj(attention_output, training=training)
        # In TFBert, attention weights are returned after dropout.
        # However, in CLIP, they are returned before dropout.
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])


# Copied from transformers.models.clip.modeling_clip.TFCLIPMLP with CLIP->Siglip
class TFSiglipMLP(keras.layers.Layer):
    def __init__(self, config: SiglipConfig, **kwargs):
        super().__init__(**kwargs)

        self.activation_fn = get_tf_activation(config.hidden_act)

        factor = config.initializer_factor
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        fc_std = (2 * config.hidden_size) ** -0.5 * factor

        self.fc1 = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        self.fc2 = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])


class TFSiglipEncoderLayer(keras.layers.Layer):
    def __init__(self, config: SiglipConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.self_attn = TFSiglipAttention(config, name="self_attn")
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFSiglipMLP(config, name="mlp")
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned
                tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(inputs=hidden_states)
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(inputs=hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attention_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])


class TFSiglipEncoder(keras.layers.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFSiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig, **kwargs):
        super().__init__(**kwargs)

        self.layers = [TFSiglipEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFSiglipTextTransformer(keras.layers.Layer):
    def __init__(self, config: SiglipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFSiglipTextEmbeddings(config, name="embeddings")
        self.encoder = TFSiglipEncoder(config, name="encoder")
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

        self.head = keras.layers.Dense(config.hidden_size, name="head")
        self.embed_dim = config.hidden_size

    def call(
            self,
            input_ids: TFModelInputType,
            attention_mask: tf.Tensor,
            position_ids: tf.Tensor,
            output_attentions: bool,
            output_hidden_states: bool,
            return_dict: bool,
            training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        input_shape = shape_list(input_ids)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_length = input_shape

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # check attention mask and invert
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = sequence_output[:, -1, :]
        pooled_output = self.head(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])


@keras_serializable
class TFCLIPTextMainLayer(keras.layers.Layer):
    config_class = SiglipTextConfig

    def __init__(self, config: SiglipTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFSiglipTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return text_model_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)


class TFCLIPVisionTransformer(keras.layers.Layer):
    def __init__(self, config: SiglipVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFSiglipVisionEmbeddings(config, name="embeddings")
        self.encoder = TFSiglipEncoder(config, name="encoder")
        self.post_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")
        self.embed_dim = config.hidden_size
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = TFSiglipMultiheadAttentionPoolingHead(config)

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        embedding_output = self.embeddings(pixel_values=pixel_values)
        embedding_output = self.pre_layernorm(inputs=embedding_output)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)

        pooled_output = self.head(sequence_output) if self.use_head else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )





