#!/usr/bin/env python3
"""Run Easi3R's original 4D visualizer with the local environment's viser client.

This avoids viser client autobuild (nodeenv) while keeping Easi3R's original UI.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import tyro
import viser
import viser.extras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_LOADER_PATH = (
    PROJECT_ROOT
    / "src"
    / "openpi"
    / "third_party"
    / "Easi3R"
    / "viser"
    / "src"
    / "viser"
    / "extras"
    / "_record3d_customized.py"
)
EASI3R_VISUALIZER_PATH = (
    PROJECT_ROOT
    / "src"
    / "openpi"
    / "third_party"
    / "Easi3R"
    / "viser"
    / "visualizer.py"
)


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    try:
        from viser import _scene_api

        params = inspect.signature(_scene_api.SceneApi.add_camera_frustum).parameters
        if "thickness" not in params:
            _orig = _scene_api.SceneApi.add_camera_frustum

            def _patched(self, *args, thickness=None, **kwargs):
                return _orig(self, *args, **kwargs)

            _scene_api.SceneApi.add_camera_frustum = _patched
    except Exception:
        pass

    custom = _load_module_from_path("easi3r_record3d_customized", CUSTOM_LOADER_PATH)
    viser.extras.Record3dLoader_Customized = custom.Record3dLoader_Customized
    visualizer = _load_module_from_path("easi3r_visualizer", EASI3R_VISUALIZER_PATH)
    tyro.cli(visualizer.main)


if __name__ == "__main__":
    main()
