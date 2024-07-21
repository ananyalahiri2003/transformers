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

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"  # TODO


LARGE_NEGATIVE = 1e-8    # CHECK


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

    class TFCLIPTextEmbeddings(keras.layers.Layer):
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
