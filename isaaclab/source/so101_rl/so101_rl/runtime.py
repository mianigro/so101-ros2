"""Bootstrap the local Isaac Sim runtime without mixing two OpenUSD runtimes."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


_EARLY_APP_LAUNCHER: Any | None = None


def bootstrap_local_isaac_sim(entrypoint: Path) -> None:
    """Re-exec ``entrypoint`` with the local Isaac Sim Python runtime if needed."""
    if (
        importlib.util.find_spec("isaacsim") is not None
        or os.environ.get("SO101_ISAACSIM_BOOTSTRAPPED") == "1"
    ):
        return

    isaaclab_root = (
        Path(os.environ.get("SO101_ISAACLAB_ROOT", "~/Documents/IsaacLab"))
        .expanduser()
        .resolve()
    )
    isaacsim_python = (
        Path(
            os.environ.get(
                "SO101_ISAACSIM_PYTHON",
                "~/Documents/isaacsim/_build/linux-x86_64/release/python.sh",
            )
        )
        .expanduser()
        .resolve()
    )
    isaaclab_site_packages = (
        isaaclab_root / ".venv" / "lib" / "python3.12" / "site-packages"
    )
    if not isaacsim_python.is_file() or not isaaclab_site_packages.is_dir():
        raise RuntimeError(
            "Isaac Sim is not installed in the active Python environment and the local source-build "
            "runtime could not be found. Set SO101_ISAACSIM_PYTHON and SO101_ISAACLAB_ROOT, or install "
            "Isaac Lab's isaacsim extra."
        )

    package_path = entrypoint.parent / "source" / "so101_rl"
    source_paths = sorted(
        path for path in (isaaclab_root / "source").iterdir() if path.is_dir()
    )
    inherited_paths = [
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
    ]
    python_paths = [
        package_path,
        *source_paths,
        isaaclab_site_packages,
        *map(Path, inherited_paths),
    ]

    environment = os.environ.copy()
    environment["SO101_ISAACSIM_BOOTSTRAPPED"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    os.execve(
        isaacsim_python,
        [str(isaacsim_python), str(entrypoint.resolve()), *sys.argv[1:]],
        environment,
    )


def launch_isaac_sim_before_task_imports(
    entrypoint: Path,
    *,
    arguments: list[str] | None = None,
    default_visualizer: str | None = None,
) -> None:
    """Start Kit before Isaac Lab resolves a task configuration.

    The local Isaac Lab virtual environment contains the kit-less ``usd-core``
    wheel, while the source-built Isaac Sim contains Kit's own OpenUSD runtime.
    Resolving an environment configuration before ``SimulationApp`` starts can
    load the wheel's ``pxr`` modules. Kit subsequently loading its own OpenUSD
    libraries then corrupts the process. Starting Kit first makes every later
    ``pxr`` import use the already-running Kit runtime.

    Args:
        entrypoint: Repository entry point being executed.
        arguments: Command-line arguments excluding the executable name.
        default_visualizer: Visualizer used when the caller did not select one.
            ``live`` passes ``"kit"``; other entries require an explicit
            ``--visualizer kit`` to open a native window.
    """
    global _EARLY_APP_LAUNCHER

    bootstrap_local_isaac_sim(entrypoint)
    if _EARLY_APP_LAUNCHER is not None:
        return

    requested_arguments = list(sys.argv[1:] if arguments is None else arguments)
    if any(argument in {"-h", "--help"} for argument in requested_arguments):
        return

    # Importing AppLauncher is deliberately the first Isaac Lab import in each
    # entry point. It does not import pxr; constructing it starts Kit.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task")
    AppLauncher.add_app_launcher_args(parser)
    # Isaac Lab no longer exposes --headless itself, but this external project
    # retains it as a documented compatibility flag.
    parser.add_argument("--headless", action="store_true", default=False)
    launcher_arguments, _ = parser.parse_known_args(
        AppLauncher._fuse_kit_args(requested_arguments)
    )

    if (
        default_visualizer is not None
        and launcher_arguments.visualizer is None
        and not getattr(launcher_arguments, "visualizer_explicit", False)
    ):
        launcher_arguments.visualizer = [default_visualizer]

    launcher_arguments.enable_cameras = True

    # Keep task, Hydra, and checkpoint arguments away from Kit's own argv
    # parser. AppLauncher receives every setting it needs through the namespace.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        if launcher_arguments.verbose:
            sys.argv.append("--verbose")
        if launcher_arguments.info:
            sys.argv.append("--info")
        _EARLY_APP_LAUNCHER = AppLauncher(launcher_arguments)
    finally:
        sys.argv = original_argv

    atexit.register(close_early_isaac_sim)


def close_early_isaac_sim() -> None:
    """Close the repository-owned early SimulationApp once."""
    global _EARLY_APP_LAUNCHER

    if _EARLY_APP_LAUNCHER is None:
        return
    app_launcher = _EARLY_APP_LAUNCHER
    _EARLY_APP_LAUNCHER = None
    app_launcher.app.close()
