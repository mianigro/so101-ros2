# Copyright 2026 Dmitri Manajev
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

"""In-context demonstration conditioning for pi0.5 on SO-101 (``pi05_icl``).

Implements ``ICL_IMPLEMENTATION.md`` (demo tokens in the prefix): frozen
``lerobot/pi05_base`` + DemoEncoder + LoRA adapters on the VLM attention.
Integration is subclassing + runtime registration only; the only repo edits
outside this package are ``"pi05_icl"`` in ``policy_server/inference_engine.py``
``SUPPORTED_POLICIES`` and pixi tasks/deps.

Import :mod:`so101_icl.registration` (or this package) before resolving the
``"pi05_icl"`` policy type through the lerobot registry.
"""

__version__ = "0.1.0"

BASE_CHECKPOINT = "lerobot/pi05_base"

# Camera renames applied at collation time (dataset key -> pi05_base key).
# SO-101 setups (rosbag_to_lerobot --setup / self-improve exports).
PI05_BASE_IMAGE_KEY_MAP = {
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.overhead_1": "observation.images.base_0_rgb",
    "observation.images.overhead_2": "observation.images.right_wrist_0_rgb",
}
# DROID (lerobot/droid_1.0.1): the second external camera occupies the
# right-wrist slot, the same pattern the SO-101 data uses for overhead_2.
DROID_IMAGE_KEY_MAP = {
    "observation.images.exterior_1_left": "observation.images.base_0_rgb",
    "observation.images.wrist_left": "observation.images.left_wrist_0_rgb",
    "observation.images.exterior_2_left": "observation.images.right_wrist_0_rgb",
}
