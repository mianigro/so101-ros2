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

"""Processor for ``pi05_icl``.

``lerobot.policies.factory._make_processors_from_policy_config`` resolves
``make_<policy_type>_pre_post_processors`` in a ``processor_<policy_type>``
module by convention — this module must exist or processor creation raises
(ICL §1, registry row). The ICL policy consumes exactly the pi05 inputs
(demo keyframes are not observation keys), so we delegate to pi05's builder.
"""

from __future__ import annotations

from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from .configuration_pi05_icl import ICLConfig


def make_pi05_icl_pre_post_processors(
    config: ICLConfig,
    dataset_stats: dict | None = None,
):
    """Delegate to pi05's processor; demo tokens never pass through it."""
    return make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)
