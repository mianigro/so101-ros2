"""Repository-owned RL model families for Isaac Lab training.

Each subpackage follows the same pattern so Isaac Lab users can pick a model
and develop new training scripts against one contract:

* ``models.py`` -- actor-critic classes implementing the RSL-RL model
  interface (``get_latent``/``forward``, observation groups, optional
  ``as_jit``/``as_onnx`` export wrappers).
* ``ppo_cfg.py`` -- RSL-RL training configurations (``@configclass``
  dataclasses whose ``class_name`` dotted paths select the classes above).

Tasks compose these configurations into runner presets and register them as
gym task IDs; the ``train``/``play``/``export`` entry points stay identical
for every family.
"""
