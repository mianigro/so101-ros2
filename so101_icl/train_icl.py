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

"""Train pi05_icl (run inside the pixi ``lerobot`` env).

Single GPU::

    python so101_icl/train_icl.py --config so101_icl/configs/icl_smoke_local_v1.yaml

2x4080 DDP::

    accelerate launch --num_processes 2 --multi_gpu so101_icl/train_icl.py --config <stage.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from so101_icl.train_loop import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
