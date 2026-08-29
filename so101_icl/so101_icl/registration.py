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

"""Runtime registration of ``"pi05_icl"`` into the lerobot policy registry.

Importing this module registers ``ICLConfig`` (via the decorator in
``configuration_pi05_icl``) and makes the ``modeling_pi05_icl`` /
``processor_pi05_icl`` modules resolvable by the convention-based factory
(``lerobot.policies.factory._get_policy_cls_from_policy_name`` rewrites
``configuration_`` to ``modeling_`` on the config class's own module). No
lerobot files are touched.

Usage (serve_icl.py, tests, train scripts)::

    import so101_icl.registration  # noqa: F401
"""

from . import configuration_pi05_icl, modeling_pi05_icl, processor_pi05_icl  # noqa: F401


def register() -> None:
    """Idempotent no-op kept for explicitness at call sites."""
    _ = (configuration_pi05_icl, modeling_pi05_icl, processor_pi05_icl)
