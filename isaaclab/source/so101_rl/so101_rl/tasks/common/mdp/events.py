"""Visual-domain randomization shared by SO-101 manipulation scenarios."""

from __future__ import annotations

import math
import random

import torch
import warp as wp

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


def _env_ids(env, env_ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def _rotation_jitter(count: int, degrees: float, device: str) -> torch.Tensor:
    limit = math.radians(degrees)
    euler = torch.empty((count, 3), device=device).uniform_(-limit, limit)
    return quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])


def _intrinsics(
    count: int,
    width: int,
    height: int,
    nominal_fov_deg: float,
    fov_jitter_deg: float,
    device: str,
) -> torch.Tensor:
    fov = torch.empty(count, device=device).uniform_(
        nominal_fov_deg - fov_jitter_deg,
        nominal_fov_deg + fov_jitter_deg,
    )
    focal = width / (2.0 * torch.tan(torch.deg2rad(fov) / 2.0))
    matrices = torch.zeros((count, 3, 3), device=device)
    matrices[:, 0, 0] = focal
    matrices[:, 1, 1] = focal
    matrices[:, 0, 2] = width * 0.5
    matrices[:, 1, 2] = height * 0.5
    matrices[:, 2, 2] = 1.0
    return matrices


def randomize_camera_calibration(
    env,
    env_ids: torch.Tensor,
    nominal: dict,
    width: int,
    height: int,
    overhead_translation_jitter_m: float = 0.010,
    wrist_translation_jitter_m: float = 0.005,
    rotation_jitter_deg: float = 3.0,
    fov_jitter_deg: float = 3.0,
) -> None:
    """Randomize all camera extrinsics and intrinsics within deployment bounds."""
    ids = _env_ids(env, env_ids)
    count = len(ids)
    origins = env.scene.env_origins[ids]

    wrist_camera = env.scene["wrist_camera"]
    wrist_pos = torch.tensor(nominal["wrist"]["pos"], device=env.device).repeat(
        count, 1
    )
    wrist_pos += torch.empty_like(wrist_pos).uniform_(
        -wrist_translation_jitter_m, wrist_translation_jitter_m
    )
    wrist_quat = torch.tensor(nominal["wrist"]["rot"], device=env.device).repeat(
        count, 1
    )
    wrist_quat = quat_mul(
        wrist_quat, _rotation_jitter(count, rotation_jitter_deg, env.device)
    )
    # Camera exposes world-pose writes only; its generic frame view is used here
    # so a reset never derives the local wrist offset from stale body kinematics.
    wrist_camera._view.set_local_poses(
        wp.from_torch(wrist_pos.contiguous(), dtype=wp.vec3f),
        wp.from_torch(wrist_quat.contiguous(), dtype=wp.vec4f),
        wp.from_torch(ids.to(dtype=torch.int32).contiguous(), dtype=wp.int32),
    )
    wrist_camera.set_intrinsic_matrices(
        _intrinsics(
            count,
            width,
            height,
            nominal["wrist"]["fov"],
            fov_jitter_deg,
            env.device,
        ),
        env_ids=ids,
    )

    for name in ("overhead_1", "overhead_2"):
        camera = env.scene[f"{name}_camera"]
        position = torch.tensor(nominal[name]["pos"], device=env.device).repeat(
            count, 1
        )
        position += origins
        position += torch.empty_like(position).uniform_(
            -overhead_translation_jitter_m, overhead_translation_jitter_m
        )
        orientation = torch.tensor(nominal[name]["rot"], device=env.device).repeat(
            count, 1
        )
        orientation = quat_mul(
            orientation, _rotation_jitter(count, rotation_jitter_deg, env.device)
        )
        camera.set_world_poses(position, orientation, ids, convention="opengl")
        camera.set_intrinsic_matrices(
            _intrinsics(
                count,
                width,
                height,
                nominal[name]["fov"],
                fov_jitter_deg,
                env.device,
            ),
            env_ids=ids,
        )


def randomize_preview_material(
    env,
    env_ids: torch.Tensor,
    asset_name: str,
    color_low: tuple[float, float, float],
    color_high: tuple[float, float, float],
    roughness_range: tuple[float, float],
) -> None:
    """Set per-environment PreviewSurface color and roughness without Replicator."""
    from pxr import Gf  # local: USD is only available after SimulationApp starts

    ids = _env_ids(env, env_ids).detach().cpu().tolist()
    stage = env.sim.stage
    for env_id in ids:
        shader_path = f"{env.scene.env_prim_paths[env_id]}/{asset_name}/material/Shader"
        shader = stage.GetPrimAtPath(shader_path)
        if not shader.IsValid():
            raise RuntimeError(f"visual randomization material is missing: {shader_path}")
        color = tuple(
            random.uniform(low, high)
            for low, high in zip(color_low, color_high, strict=True)
        )
        roughness = random.uniform(*roughness_range)
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*color))
        shader.GetAttribute("inputs:roughness").Set(roughness)


def randomize_scene_lighting(
    env,
    env_ids: torch.Tensor | None,
    dome_intensity_range: tuple[float, float],
    distant_intensity_range: tuple[float, float],
    color_range: tuple[float, float],
) -> None:
    """Randomize the shared rig lighting on a global interval."""
    del env_ids
    from pxr import Gf

    color = Gf.Vec3f(*(random.uniform(*color_range) for _ in range(3)))
    for path, intensity_range in (
        ("/World/DomeLight", dome_intensity_range),
        ("/World/DistantLight", distant_intensity_range),
    ):
        light = env.sim.stage.GetPrimAtPath(path)
        if not light.IsValid():
            raise RuntimeError(f"light randomization prim is missing: {path}")
        light.GetAttribute("inputs:intensity").Set(random.uniform(*intensity_range))
        light.GetAttribute("inputs:color").Set(color)
