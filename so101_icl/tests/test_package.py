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

"""Lightweight package check: the processor convention resolution."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPackage(unittest.TestCase):
    def test_processor_factory_resolution(self):
        """make_pre_post_processors must find processor_pi05_icl by convention."""
        import so101_icl.registration  # noqa: F401
        from lerobot.policies.factory import make_pre_post_processors
        from so101_icl.configuration_pi05_icl import icl_config_from_base

        cfg = icl_config_from_base(device="cpu")
        pre, post = make_pre_post_processors(cfg, pretrained_path=None)
        self.assertIsNotNone(pre)
        self.assertIsNotNone(post)


if __name__ == "__main__":
    unittest.main()
