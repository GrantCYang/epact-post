"""Regression for POST ignoring the explicitly staged renderer."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class PostRendererPathTests(unittest.TestCase):
    def test_launch_passes_staged_executable_and_missing_path_fails(self):
        path = Path(__file__).resolve().parents[1] / "src/post/body.py"
        spec = importlib.util.spec_from_file_location("post_body_regression", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        controller = Mock()
        fake_controller = types.ModuleType("ai2thor.controller")
        fake_controller.Controller = controller
        fake_platform = types.ModuleType("ai2thor.platform")
        fake_platform.CloudRendering = object()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                sys.modules,
                {
                    "ai2thor": types.ModuleType("ai2thor"),
                    "ai2thor.controller": fake_controller,
                    "ai2thor.platform": fake_platform,
                },
            ),
        ):
            executable = Path(tmp) / "thor"
            executable.touch()
            with patch.dict("os.environ", {"EPACT_SIMULATOR_PATH": str(executable)}):
                module.PostBody().launch("FloorPlan1")
                self.assertEqual(
                    controller.call_args.kwargs["local_executable_path"],
                    str(executable),
                )
            controller.reset_mock()
            with patch.dict(
                "os.environ", {"EPACT_SIMULATOR_PATH": str(Path(tmp) / "missing")}
            ):
                with self.assertRaises(FileNotFoundError):
                    module.PostBody().launch("FloorPlan1")
            controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
