"""Convert the repository's task STLs into validated, physics-ready USD assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .paths import (
    ASSET_MANIFEST_PATH,
    CUBE_STL_PATH,
    CUBE_USD_PATH,
    CUP_STL_PATH,
    CUP_USD_PATH,
    GENERATED_ASSET_DIR,
    ROBOT_USD_PATH,
)

ASSET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PreparedGeometry:
    """Normalized task meshes plus dimensions used by the environment contract."""

    cube: trimesh.Trimesh
    cup: trimesh.Trimesh
    cup_inner_radius_m: float
    cup_inner_bottom_m: float
    cup_outer_radius_m: float
    cube_horizontal_radius_m: float
    success_xy_tolerance_m: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robot_asset_digest(robot_usd: Path) -> str:
    digest = hashlib.sha256()
    root = robot_usd.parent
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_watertight_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"STL source does not exist: {path}")
    loaded = trimesh.load_mesh(path, process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"expected one mesh in {path}, got {type(loaded).__name__}")
    if loaded.is_empty or len(loaded.faces) == 0:
        raise ValueError(f"STL contains no triangles: {path}")
    if not loaded.is_watertight or not loaded.is_winding_consistent:
        raise ValueError(f"STL must be watertight with consistent winding: {path}")
    return loaded


def inspect_sources(
    cube_stl: Path, cup_stl: Path, scale: float = 0.001
) -> PreparedGeometry:
    """Load, validate, scale, and normalize the cube and cup meshes."""
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    cube = _load_watertight_mesh(cube_stl).copy()
    cup = _load_watertight_mesh(cup_stl).copy()

    # The cube's rigid-body frame is its center of mass. The cup's rigid-body frame is
    # centered in XY at the bottom surface, which makes placement tests and resets direct.
    cube.apply_translation(-cube.bounding_box.centroid)
    cube.apply_scale(scale)

    cup_center_xy = 0.5 * (cup.bounds[0, :2] + cup.bounds[1, :2])
    cup.apply_translation((-cup_center_xy[0], -cup_center_xy[1], -cup.bounds[0, 2]))
    cup.apply_scale(scale)

    cube_extents = cube.extents
    cup_extents = cup.extents
    if not np.allclose(cube_extents, (0.025, 0.025, 0.025), atol=0.002):
        raise ValueError(
            "cube dimensions are not approximately 25 mm after scaling; "
            f"got {cube_extents.tolist()} m. Check --scale."
        )
    if not np.allclose(cup_extents, (0.05, 0.05, 0.05), atol=0.003):
        raise ValueError(
            "cup dimensions are not approximately 50 mm after scaling; "
            f"got {cup_extents.tolist()} m. Check --scale."
        )

    radii = np.linalg.norm(cup.vertices[:, :2], axis=1)
    outer_radius = float(np.max(radii))
    radial_midpoint = float(0.5 * (np.min(radii) + outer_radius))
    inner_vertices = cup.vertices[radii < radial_midpoint]
    if len(inner_vertices) == 0:
        raise ValueError("could not identify the cup's inner wall")
    inner_radius = float(np.median(np.linalg.norm(inner_vertices[:, :2], axis=1)))
    inner_bottom = float(np.min(inner_vertices[:, 2]))

    cube_horizontal_radius = float(np.linalg.norm(cube_extents[:2] * 0.5))
    fit_clearance = inner_radius - cube_horizontal_radius
    if fit_clearance <= 0.001:
        raise ValueError(
            "cube does not fit inside the cup with at least 1 mm radial clearance: "
            f"inner_radius={inner_radius:.6f} m, cube_radius={cube_horizontal_radius:.6f} m"
        )

    return PreparedGeometry(
        cube=cube,
        cup=cup,
        cup_inner_radius_m=inner_radius,
        cup_inner_bottom_m=inner_bottom,
        cup_outer_radius_m=outer_radius,
        cube_horizontal_radius_m=cube_horizontal_radius,
        success_xy_tolerance_m=fit_clearance * 0.8,
    )


def _author_mesh(
    usd_mesh: Any, mesh: trimesh.Trimesh, color: tuple[float, float, float]
) -> None:
    from pxr import Gf, UsdGeom

    usd_mesh.CreatePointsAttr(
        [Gf.Vec3f(*(float(value) for value in vertex)) for vertex in mesh.vertices]
    )
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(
        mesh.faces.reshape(-1).astype(np.int32).tolist()
    )
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _write_usd(
    output: Path,
    name: str,
    visual_mesh: trimesh.Trimesh,
    collision_meshes: list[trimesh.Trimesh],
    mass_kg: float,
    color: tuple[float, float, float],
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> None:
    from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    root.GetPrim().SetMetadata("kind", Kind.Tokens.component)
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(float(mass_kg))
    density = mass_kg / float(abs(visual_mesh.volume))
    mass_api.CreateCenterOfMassAttr(
        Gf.Vec3f(*(float(value) for value in visual_mesh.center_mass))
    )
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            *(float(value * density) for value in np.diag(visual_mesh.moment_inertia))
        )
    )

    material = UsdShade.Material.Define(stage, f"/{name}/PhysicsMaterial")
    material_prim = material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(material_prim)
    material_prim.CreateAttribute(
        "physics:staticFriction", Sdf.ValueTypeNames.Float
    ).Set(float(static_friction))
    material_prim.CreateAttribute(
        "physics:dynamicFriction", Sdf.ValueTypeNames.Float
    ).Set(float(dynamic_friction))
    material_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(
        float(restitution)
    )

    visual = UsdGeom.Mesh.Define(stage, f"/{name}/Visual")
    _author_mesh(visual, visual_mesh, color)

    UsdGeom.Scope.Define(stage, f"/{name}/Collisions")
    for index, collision_mesh in enumerate(collision_meshes):
        collider = UsdGeom.Mesh.Define(stage, f"/{name}/Collisions/hull_{index:02d}")
        _author_mesh(collider, collision_mesh, color)
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        collider.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
        UsdPhysics.MeshCollisionAPI.Apply(
            collider.GetPrim()
        ).CreateApproximationAttr().Set("convexHull")
        collider.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        collider.GetPrim().CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        ).Set(0.001)
        collider.GetPrim().CreateAttribute(
            "physxCollision:restOffset", Sdf.ValueTypeNames.Float
        ).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            material,
            materialPurpose="physics",
        )

    stage.GetRootLayer().customLayerData = {
        "creator": "so101_rl.asset_prep",
        "assetFormatVersion": ASSET_FORMAT_VERSION,
    }
    stage.GetRootLayer().Save()


def _decompose_cup(
    cup: trimesh.Trimesh,
    threshold: float,
    max_hulls: int,
    seed: int,
) -> list[trimesh.Trimesh]:
    import coacd

    coacd_mesh = coacd.Mesh(cup.vertices.astype(np.float64), cup.faces.astype(np.int32))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=1000,
        mcts_nodes=10,
        mcts_iterations=80,
        mcts_max_depth=3,
        merge=True,
        max_ch_vertex=128,
        seed=seed,
    )
    hulls = [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        for vertices, faces in parts
    ]
    if not hulls or len(hulls) > max_hulls:
        raise RuntimeError(
            f"cup decomposition produced {len(hulls)} hulls; expected 1..{max_hulls}"
        )
    return hulls


def _validate_open_cavity(
    geometry: PreparedGeometry, cup_hulls: list[trimesh.Trimesh]
) -> None:
    clearance_radius = geometry.cube_horizontal_radius_m + 0.001
    height = float(geometry.cup.extents[2])
    radii = np.linspace(0.0, clearance_radius, 8)
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    z_samples = np.linspace(geometry.cup_inner_bottom_m + 0.001, height - 0.001, 28)
    clearance_samples = np.array(
        [
            (radius * np.cos(angle), radius * np.sin(angle), z)
            for z in z_samples
            for radius in radii
            for angle in angles
        ]
    )
    for index, hull in enumerate(cup_hulls):
        if np.any(hull.contains(clearance_samples)):
            raise RuntimeError(
                f"cup collision hull {index} obstructs the cube clearance cylinder; "
                "adjust decomposition settings"
            )


def _manifest_data(
    geometry: PreparedGeometry,
    cube_stl: Path,
    cup_stl: Path,
    robot_usd: Path,
    scale: float,
    hull_count: int,
    threshold: float,
    max_hulls: int,
    seed: int,
) -> dict[str, Any]:
    cube_height = float(geometry.cube.extents[2])
    cup_height = float(geometry.cup.extents[2])
    return {
        "asset_format_version": ASSET_FORMAT_VERSION,
        "source": {
            "cube_stl": str(cube_stl.resolve()),
            "cube_sha256": _sha256(cube_stl),
            "cup_stl": str(cup_stl.resolve()),
            "cup_sha256": _sha256(cup_stl),
            "robot_usd": str(robot_usd.resolve()),
            "robot_asset_sha256": _robot_asset_digest(robot_usd),
            "stl_to_meters": scale,
        },
        "geometry": {
            "cube_extents_m": geometry.cube.extents.tolist(),
            "cube_horizontal_radius_m": geometry.cube_horizontal_radius_m,
            "cup_extents_m": geometry.cup.extents.tolist(),
            "cup_inner_radius_m": geometry.cup_inner_radius_m,
            "cup_inner_bottom_m": geometry.cup_inner_bottom_m,
            "cup_outer_radius_m": geometry.cup_outer_radius_m,
            "success_xy_tolerance_m": geometry.success_xy_tolerance_m,
            "success_center_z_min_m": geometry.cup_inner_bottom_m
            + 0.5 * cube_height
            - 0.001,
            "success_center_z_max_m": cup_height - 0.5 * cube_height + 0.001,
        },
        "collision": {
            "cube": "convex_hull",
            "cup": "coacd_compound_convex",
            "cup_hull_count": hull_count,
            "coacd_threshold": threshold,
            "coacd_max_hulls": max_hulls,
            "coacd_seed": seed,
        },
    }


def prepare_assets(
    cube_stl: Path = CUBE_STL_PATH,
    cup_stl: Path = CUP_STL_PATH,
    robot_usd: Path = ROBOT_USD_PATH,
    output_dir: Path = GENERATED_ASSET_DIR,
    scale: float = 0.001,
    cup_threshold: float = 0.1,
    cup_max_hulls: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Prepare the task assets and return their manifest."""
    cube_stl = cube_stl.expanduser().resolve()
    cup_stl = cup_stl.expanduser().resolve()
    robot_usd = robot_usd.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(
            f"generated SO-101 USD does not exist: {robot_usd}\n"
            "Run scripts/isaac_sim_teleop.py once (or with --rebuild-asset) first."
        )

    geometry = inspect_sources(cube_stl, cup_stl, scale)
    cube_hulls = [geometry.cube.convex_hull]
    cup_hulls = _decompose_cup(geometry.cup, cup_threshold, cup_max_hulls, seed)
    _validate_open_cavity(geometry, cup_hulls)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="so101_assets_", dir=output_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        cube_usd = temporary_dir / CUBE_USD_PATH.name
        cup_usd = temporary_dir / CUP_USD_PATH.name
        _write_usd(
            cube_usd,
            "Cube",
            geometry.cube,
            cube_hulls,
            mass_kg=0.020,
            color=(0.12, 0.35, 0.85),
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        )
        _write_usd(
            cup_usd,
            "Cup",
            geometry.cup,
            cup_hulls,
            mass_kg=0.050,
            color=(0.85, 0.35, 0.12),
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        )
        manifest = _manifest_data(
            geometry,
            cube_stl,
            cup_stl,
            robot_usd,
            scale,
            len(cup_hulls),
            cup_threshold,
            cup_max_hulls,
            seed,
        )
        (temporary_dir / ASSET_MANIFEST_PATH.name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temporary_dir), str(output_dir))

    return manifest


def validate_generated_assets(output_dir: Path = GENERATED_ASSET_DIR) -> dict[str, Any]:
    """Validate the generated USD contract without launching Isaac Sim."""
    from pxr import Usd, UsdPhysics

    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / ASSET_MANIFEST_PATH.name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"asset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("asset_format_version") != ASSET_FORMAT_VERSION:
        raise ValueError(f"unsupported asset format version in {manifest_path}")

    for filename in (CUBE_USD_PATH.name, CUP_USD_PATH.name):
        usd_path = output_dir / filename
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise ValueError(f"USD has no valid default prim: {usd_path}")
        collision_prims = [
            prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not collision_prims:
            raise ValueError(f"USD has no collision prims: {usd_path}")
        for prim in collision_prims:
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            )
            if approximation != "convexHull":
                raise ValueError(
                    f"unexpected collision approximation {approximation!r} on {prim.GetPath()}"
                )
            contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()
            if contact_offset is None or not np.isclose(contact_offset, 0.001):
                raise ValueError(
                    f"collision contact offset is not authored on {prim.GetPath()}"
                )
            rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
            if rest_offset is None or not np.isclose(rest_offset, 0.0):
                raise ValueError(
                    f"collision rest offset is not authored on {prim.GetPath()}"
                )
            binding = prim.GetRelationship("material:binding:physics")
            targets = binding.GetTargets() if binding else []
            if len(targets) != 1:
                raise ValueError(
                    f"collision has no unique physics-material binding: {prim.GetPath()}"
                )
            material_prim = stage.GetPrimAtPath(targets[0])
            material_values = {
                name: material_prim.GetAttribute(name).Get()
                for name in (
                    "physics:staticFriction",
                    "physics:dynamicFriction",
                    "physics:restitution",
                )
            }
            if any(value is None for value in material_values.values()):
                raise ValueError(
                    f"physics material is incomplete on {material_prim.GetPath()}"
                )
        root = stage.GetDefaultPrim()
        mass_api = UsdPhysics.MassAPI(root)
        if (
            mass_api.GetMassAttr().Get() is None
            or mass_api.GetCenterOfMassAttr().Get() is None
            or mass_api.GetDiagonalInertiaAttr().Get() is None
        ):
            raise ValueError(f"USD mass properties are not fully authored: {usd_path}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-stl", type=Path, default=CUBE_STL_PATH)
    parser.add_argument("--cup-stl", type=Path, default=CUP_STL_PATH)
    parser.add_argument("--robot-usd", type=Path, default=ROBOT_USD_PATH)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_ASSET_DIR)
    parser.add_argument(
        "--scale", type=float, default=0.001, help="STL-unit to metre scale"
    )
    parser.add_argument("--cup-threshold", type=float, default=0.1)
    parser.add_argument("--cup-max-hulls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.validate_only:
        manifest = validate_generated_assets(args.output_dir)
    else:
        manifest = prepare_assets(
            cube_stl=args.cube_stl,
            cup_stl=args.cup_stl,
            robot_usd=args.robot_usd,
            output_dir=args.output_dir,
            scale=args.scale,
            cup_threshold=args.cup_threshold,
            cup_max_hulls=args.cup_max_hulls,
            seed=args.seed,
        )
        validate_generated_assets(args.output_dir)
    geometry = manifest["geometry"]
    print(
        f"Prepared assets in {args.output_dir.expanduser().resolve()} "
        f"(cup hulls={manifest['collision']['cup_hull_count']}, "
        f"fit tolerance={geometry['success_xy_tolerance_m'] * 1000.0:.2f} mm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
