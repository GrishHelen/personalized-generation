# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
# 
# Copyright (c) 2023 AttendAndExcite
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling ConsiStory or otherwise documented as NVIDIA-proprietary
# are not a contribution and subject to the license under the LICENSE file located at the root directory.

import torch
from collections import defaultdict
import numpy as np
from typing import Union, List
from PIL import Image
from IPython.display import display

from utils.general_utils import attn_map_to_binary
import torch.nn.functional as F


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True,
                downscale_rate=None) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    if downscale_rate:
        pil_img = pil_img.resize((int(pil_img.size[0] // downscale_rate), int(pil_img.size[1] // downscale_rate)))

    if display_image:
        display(pil_img)
    return pil_img


class AttentionStore:
    def __init__(self, attention_store_kwargs):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.attn_res = attention_store_kwargs.get('attn_res', (32, 32))
        self.token_indices = attention_store_kwargs['token_indices']
        bsz = self.token_indices.size(1)
        self.original_attn_masks = attention_store_kwargs.get('original_attn_masks', None)
        self.extended_mapping = attention_store_kwargs.get('extended_mapping', torch.ones(bsz, bsz).bool())
        self.n_anchors = attention_store_kwargs.get('n_anchors', None)

        self.n_svd_k = attention_store_kwargs.get('n_svd_k', None)
        self.n_svd_v = attention_store_kwargs.get('n_svd_v', None)
        self.alpha = attention_store_kwargs.get('alpha', 0.)
        self.max_sigma = attention_store_kwargs.get('max_sigma', 1.)
        self.newton_p = attention_store_kwargs.get('newton_p', 1.)
        self.method = attention_store_kwargs.get('method', 'svd, nullify')

        self.curr_iter = 0
        self.ALL_RES = [32, 64]
        self.step_store = defaultdict(list)
        self.last_mask = {res: None for res in self.ALL_RES}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, attn_heads: int):
        if is_cross and attn.shape[1] == np.prod(self.attn_res):
            guidance_attention = attn[attn.size(0) // 2:]
            batched_guidance_attention = guidance_attention.reshape(
                [guidance_attention.shape[0] // attn_heads, attn_heads, *guidance_attention.shape[1:]])
            batched_guidance_attention = batched_guidance_attention.mean(dim=1)
            self.step_store[place_in_unet].append(batched_guidance_attention)

    def reset(self):
        self.step_store = defaultdict(list)
        self.last_mask = {res: None for res in self.ALL_RES}

        torch.cuda.empty_cache()

    def aggregate_last_steps_attention(self) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        attention_maps = torch.cat([torch.stack(x[-20:]) for x in self.step_store.values()]).mean(dim=0)
        bsz, wh, _ = attention_maps.shape

        # Create attention maps for each concept token, for each batch item
        agg_attn_maps = []
        for i in range(bsz):
            curr_prompt_indices = []

            for concept_token_indices in self.token_indices:
                if concept_token_indices[i] != -1:
                    curr_prompt_indices.append(attention_maps[i, :, concept_token_indices[i]].view(*self.attn_res))

            agg_attn_maps.append(torch.stack(curr_prompt_indices))

        # Upsample the attention maps to the target resolution
        # and create the attention masks, unifying masks across the different concepts
        for tgt_size in self.ALL_RES:
            tgt_agg_attn_maps = [F.interpolate(x.unsqueeze(1), size=tgt_size, mode='bilinear').squeeze(1) for x in
                                 agg_attn_maps]

            attn_masks = []
            for batch_item_map in tgt_agg_attn_maps:
                concept_attn_masks = []

                for concept_maps in batch_item_map:
                    concept_attn_masks.append(
                        torch.from_numpy(attn_map_to_binary(concept_maps, 1.)).to(attention_maps.device).bool().view(
                            -1))

                concept_attn_masks = torch.stack(concept_attn_masks, dim=0).max(dim=0).values
                attn_masks.append(concept_attn_masks)

            attn_masks = torch.stack(attn_masks)
            self.last_mask[tgt_size] = attn_masks.clone()

    def get_extended_attn_mask_instance(self, width, i, batch_size):
        n_patches = width ** 2

        output_attn_mask = torch.zeros((batch_size * n_patches,), dtype=torch.bool)
        for j in range(batch_size):
            if i == j:
                output_attn_mask[j * n_patches:(j + 1) * n_patches] = 1
            else:
                if self.extended_mapping[i, j]:
                    output_attn_mask[j * n_patches:(j + 1) * n_patches] = True

        return output_attn_mask
