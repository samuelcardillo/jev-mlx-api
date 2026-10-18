"""CPU-only checks for runtime selection and managed-server handoff.

Run with: python3 -m unittest discover -s tests -p 'test_runtime.py'
No MLX import, model load, native build, network request, or real process control.
"""
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
BUILD_ID = "test-pinned-build"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeFixture(unittest.TestCase):
    def enterContext(self, context):
        # Also support macOS's system Python before unittest added enterContext.
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-runtime-test-")
        self.addCleanup(temporary.cleanup)
        # macOS /var aliases /private/var; paths.py canonicalizes JEV_HOME too.
        self.root = Path(temporary.name).resolve()
        self.install = self.root / "runtime"
        self.install.mkdir()
        self.here = self.root / "recipe"
        self.here.mkdir()
        self.base_python = self.root / "base-python"
        self.base_python.write_text("#!/bin/sh\nexit 0\n")
        self.base_python.chmod(0o755)
        self.stock_python = self.make_python(".venv")
        self.int8_python = self.make_python(".venv-int8")
        paths = types.ModuleType("paths")
        paths.HERE = self.here
        paths.INSTALL = self.install
        paths.PYTHON = self.stock_python
        paths.SOURCE = self.install / "mlx-vlm"
        paths.MODEL = self.install / "model"
        native = types.ModuleType("native_build")
        native.build_id = mock.Mock(return_value=BUILD_ID)
        native.install_native = mock.Mock(return_value={"build_id": BUILD_ID})
        native.validate_installed = mock.Mock(return_value={"build_id": BUILD_ID})
        native.support_reason = mock.Mock(return_value=None)
        self.native = native
        self.enterContext(mock.patch.dict(sys.modules, {"paths": paths, "native_build": native}))
        # Deliberately discard inherited runtime overrides, credentials, etc.
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.runtime = load_module("jev_test_runtime", "runtime.py")
        self.enterContext(mock.patch.dict(sys.modules, {"runtime": self.runtime}))

    def make_python(self, environment):
        path = self.install / environment / "bin/python"
        path.parent.mkdir(parents=True)
        path.symlink_to(self.base_python)
        return path

    def save(self, value):
        self.runtime.SELECTION.write_text(json.dumps(value))

    def marker(self, **changes):
        value = {"build_id": BUILD_ID, "python_path": str(self.int8_python)}
        value.update(changes)
        (self.install / "native-runtime.json").write_text(json.dumps(value))


class RuntimeSelectionTests(RuntimeFixture):
    def test_fresh_install_selects_stock_and_preserves_venv_symlink(self):
        selected = self.runtime.select_runtime()
        self.assertEqual(selected, {"name": "stock", "python": self.stock_python, "int8_flag": "0"})
        self.assertTrue(selected["python"].is_symlink())
        self.assertNotEqual(selected["python"], self.base_python)
        self.native.build_id.assert_not_called()
        self.native.validate_installed.assert_not_called()

    def test_saved_int8_requires_matching_build_and_interpreter(self):
        self.save({"runtime": "int8", "build_id": BUILD_ID})
        self.marker()
        selected = self.runtime.select_runtime()
        self.assertEqual(selected["name"], "int8")
        self.assertEqual(selected["python"], self.int8_python)
        self.assertEqual(selected["build_id"], BUILD_ID)
        self.assertEqual(selected["int8_flag"], "1")
        self.assertTrue(selected["python"].is_symlink())
        self.native.validate_installed.assert_called_once_with(
            self.install, self.install / "mlx-vlm", self.stock_python
        )

    def test_tampered_native_runtime_is_rejected_before_launch(self):
        self.save({"runtime": "int8"})
        self.marker()
        self.native.validate_installed.side_effect = RuntimeError("Installed native MLX files or version differ")
        with self.assertRaisesRegex(SystemExit, "INT8 runtime verification failed: Installed native MLX files or version differ"):
            self.runtime.select_runtime()

    def test_native_runtime_disappearing_during_validation_is_rejected(self):
        self.save({"runtime": "int8"})
        self.marker()
        self.native.validate_installed.return_value = None
        with self.assertRaisesRegex(SystemExit, "INT8 runtime verification failed: The native installation marker is missing"):
            self.runtime.select_runtime()

    def test_environment_can_override_saved_int8_with_stock(self):
        self.save({"runtime": "int8"})
        # No INT8 installation is required when explicitly choosing stock.
        os.environ["JEV_RUNTIME"] = "stock"
        self.assertEqual(self.runtime.select_runtime()["name"], "stock")
        self.native.build_id.assert_not_called()
        self.native.validate_installed.assert_not_called()

    def test_environment_can_override_saved_stock_with_int8(self):
        self.save({"runtime": "stock"})
        self.marker()
        os.environ["JEV_RUNTIME"] = "int8"
        self.assertEqual(self.runtime.select_runtime()["python"], self.int8_python)

    def test_stale_build_marker_is_rejected(self):
        self.save({"runtime": "int8"})
        self.marker(build_id="old-build")
        with self.assertRaisesRegex(SystemExit, "different recipe revision"):
            self.runtime.select_runtime()

    def test_relocated_int8_marker_is_rejected(self):
        self.save({"runtime": "int8"})
        self.marker(python_path=str(self.root / "another-runtime/bin/python"))
        with self.assertRaisesRegex(SystemExit, "location has changed"):
            self.runtime.select_runtime()

    def test_missing_int8_marker_is_rejected(self):
        self.save({"runtime": "int8"})
        with self.assertRaisesRegex(SystemExit, "not installed"):
            self.runtime.select_runtime()

    def test_missing_int8_interpreter_is_rejected(self):
        self.save({"runtime": "int8"})
        self.marker()
        self.int8_python.unlink()
        with self.assertRaisesRegex(SystemExit, "executable interpreter"):
            self.runtime.select_runtime()

    def test_stock_environment_forces_native_flag_off(self):
        os.environ["MLX_QMM_INT8"] = "1"
        env = self.runtime.child_environment(self.runtime.select_runtime())
        self.assertEqual(env["MLX_QMM_INT8"], "0")
        self.assertEqual(env["JEV_ACTIVE_RUNTIME"], "stock")
        self.assertEqual(env["JEV_HOME"], str(self.install))
        self.assertEqual(os.environ["MLX_QMM_INT8"], "1")

    def test_int8_environment_forces_native_flag_on(self):
        self.save({"runtime": "int8"})
        self.marker()
        os.environ["MLX_QMM_INT8"] = "0"
        env = self.runtime.child_environment(self.runtime.select_runtime())
        self.assertEqual(env["MLX_QMM_INT8"], "1")
        self.assertEqual(env["JEV_ACTIVE_RUNTIME"], "int8")

    def test_custom_interpreter_is_server_only_and_inherits_explicit_flag(self):
        os.environ.update(JEV_SERVER_PYTHON=str(self.int8_python), MLX_QMM_INT8="1")
        server = self.runtime.select_runtime(server=True)
        self.assertEqual(server, {"name": "custom", "python": self.int8_python, "int8_flag": "1"})
        self.native.validate_installed.assert_not_called()
        cli = self.runtime.select_runtime(server=False)
        self.assertEqual(cli["name"], "stock")
        self.assertEqual(cli["python"], self.stock_python)
        self.assertEqual(self.runtime.child_environment(cli)["MLX_QMM_INT8"], "0")

    def test_custom_interpreter_defaults_to_flag_off(self):
        os.environ["JEV_SERVER_PYTHON"] = str(self.int8_python)
        self.assertEqual(self.runtime.select_runtime(server=True)["int8_flag"], "0")

    def test_invalid_selection_is_rejected(self):
        for value in ("auto", "bogus", "", "INT8"):
            with self.subTest(value=value):
                os.environ["JEV_RUNTIME"] = value
                with self.assertRaisesRegex(SystemExit, "must be stock or int8"):
                    self.runtime.select_runtime()

    def test_invalid_json_selection_is_rejected(self):
        self.runtime.SELECTION.write_text("{bad json")
        with self.assertRaisesRegex(SystemExit, "Cannot read runtime selection"):
            self.runtime.select_runtime()

    def test_json_scalar_or_array_selection_is_rejected(self):
        for value in (None, [], "int8", 1):
            with self.subTest(value=value):
                self.save(value)
                with self.assertRaisesRegex(SystemExit, "Invalid runtime selection"):
                    self.runtime.select_runtime()

    def test_json_scalar_or_array_native_marker_is_rejected(self):
        self.save({"runtime": "int8"})
        for value in (None, [], "int8", 1):
            with self.subTest(value=value):
                (self.install / "native-runtime.json").write_text(json.dumps(value))
                with self.assertRaisesRegex(SystemExit, "Invalid INT8 runtime metadata"):
                    self.runtime.select_runtime()

    def test_custom_relative_and_nonexecutable_paths_are_rejected(self):
        plain = self.root / "not-executable"
        plain.write_text("not executable")
        for value in ("relative/python", str(plain), str(self.root / "missing")):
            with self.subTest(value=value):
                os.environ["JEV_SERVER_PYTHON"] = value
                with self.assertRaisesRegex(SystemExit, "executable interpreter"):
                    self.runtime.select_runtime(server=True)

    def test_saved_selection_round_trip_and_no_temporary_file(self):
        self.marker()
        self.runtime.save_selection("int8", {"build_id": BUILD_ID})
        self.assertEqual(self.runtime.select_runtime()["name"], "int8")
        self.assertEqual(json.loads(self.runtime.SELECTION.read_text())["build_id"], BUILD_ID)
        self.assertFalse(self.runtime.SELECTION.with_suffix(".tmp").exists())
        self.runtime.save_selection("stock")
        self.assertEqual(self.runtime.select_runtime()["name"], "stock")

    def test_signature_changes_with_runtime_build_or_interpreter(self):
        base = {"name": "int8", "python": self.int8_python, "int8_flag": "1", "build_id": BUILD_ID}
        original = self.runtime.signature(base)
        self.assertEqual(original, self.runtime.signature(dict(reversed(list(base.items())))))
        for field, value in (("name", "stock"), ("python", self.stock_python), ("int8_flag", "0"), ("build_id", "new")):
            with self.subTest(field=field):
                self.assertNotEqual(original, self.runtime.signature({**base, field: value}))


class SetupSelectionTests(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.setup = load_module("jev_test_setup", "setup.py")
        self.enterContext(mock.patch.object(self.setup.platform, "system", return_value="Darwin"))
        self.enterContext(mock.patch.object(self.setup.platform, "machine", return_value="arm64"))
        self.install_mock = self.enterContext(mock.patch.object(self.setup, "install"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_setup(self, choice):
        with mock.patch.object(sys, "argv", ["setup.py", "--runtime", choice]):
            self.setup.main()

    def test_auto_selects_int8_on_supported_machine(self):
        self.run_setup("auto")
        self.install_mock.assert_called_once()
        self.native.install_native.assert_called_once()
        self.assertEqual(json.loads(self.runtime.SELECTION.read_text())["runtime"], "int8")

    def test_auto_selects_stock_on_unsupported_machine(self):
        self.native.support_reason.return_value = "Native INT8 requires an M5 GPU."
        self.run_setup("auto")
        self.install_mock.assert_called_once()
        self.native.install_native.assert_not_called()
        self.assertEqual(json.loads(self.runtime.SELECTION.read_text())["runtime"], "stock")

    def test_explicit_stock_skips_native_support_checks(self):
        self.run_setup("stock")
        self.native.support_reason.assert_not_called()
        self.native.install_native.assert_not_called()
        self.assertEqual(json.loads(self.runtime.SELECTION.read_text())["runtime"], "stock")

    def test_unsupported_explicit_int8_does_not_install(self):
        self.native.support_reason.return_value = "Unsupported hardware."
        with self.assertRaisesRegex(SystemExit, "INT8 is unavailable"):
            self.run_setup("int8")
        self.install_mock.assert_not_called()
        self.native.install_native.assert_not_called()
        self.assertFalse(self.runtime.SELECTION.exists())

    def test_failed_native_build_preserves_previous_selection(self):
        self.runtime.save_selection("stock")
        self.native.install_native.side_effect = RuntimeError("test failure")
        with self.assertRaisesRegex(SystemExit, "released runtime is preserved"):
            self.run_setup("int8")
        self.assertEqual(json.loads(self.runtime.SELECTION.read_text())["runtime"], "stock")

    def test_invalid_environment_default_is_rejected_before_installation(self):
        os.environ["JEV_RUNTIME"] = "typo"
        with mock.patch.object(sys, "argv", ["setup.py"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.setup.main()
        self.install_mock.assert_not_called()
        self.native.install_native.assert_not_called()
        self.assertFalse(self.runtime.SELECTION.exists())


class LauncherTests(RuntimeFixture):
    def run_launcher(self, name, arguments, extra_environment=None):
        launcher = self.here / name
        launcher.write_text((REPO / name).read_text())
        launcher.chmod(0o755)
        environment = {"PATH": os.defpath, "JEV_HOME": str(self.install)}
        environment.update(extra_environment or {})
        completed = subprocess.run(
            ["/bin/zsh", str(launcher), *arguments], cwd=self.root,
            env=environment, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_jev_forwards_arguments_through_runtime_from_any_directory(self):
        # Use a real interpreter only for tiny temporary stubs, never run.py.
        self.stock_python.unlink()
        self.stock_python.symlink_to(sys.executable)
        (self.here / "paths.py").write_text(
            'import os; print(os.path.join(os.environ["JEV_HOME"], ".venv/bin/python"))\n'
        )
        (self.here / "runtime.py").write_text(
            'import json, os, sys; print(json.dumps({"args": sys.argv[1:], '
            '"runtime": os.environ.get("JEV_RUNTIME"), "python": sys.executable}))\n'
        )
        result = self.run_launcher("jev", ["decide", "--state", "words with spaces"], {"JEV_RUNTIME": "int8"})
        self.assertEqual(result["args"], ["decide", "--state", "words with spaces"])
        self.assertEqual(result["runtime"], "int8")
        self.assertEqual(result["python"], str(self.stock_python))

    def test_recipe_passes_choice_to_setup_then_clears_startup_overrides(self):
        capture = (
            'import json, os, sys\n'
            'value = {"args": sys.argv[1:], "env": {key: os.environ.get(key) '
            'for key in ("JEV_HOME", "JEV_RUNTIME", "JEV_SERVER_PYTHON", "MLX_QMM_INT8")}}\n'
        )
        (self.here / "setup.py").write_text(
            capture + 'open(os.path.join(os.environ["JEV_HOME"], "setup-capture.json"), "w").write(json.dumps(value))\n'
        )
        serve = self.here / "serve"
        serve.write_text("#!/usr/bin/python3\n" + capture + "print(json.dumps(value))\n")
        serve.chmod(0o755)
        result = self.run_launcher("recipe", ["--runtime", "stock"], {
            "JEV_RUNTIME": "int8", "JEV_SERVER_PYTHON": str(self.int8_python), "MLX_QMM_INT8": "1",
        })
        setup = json.loads((self.install / "setup-capture.json").read_text())
        self.assertEqual(setup["args"], ["--runtime", "stock"])
        self.assertEqual(setup["env"]["JEV_RUNTIME"], "int8")
        self.assertEqual(result["args"], ["start"])
        self.assertEqual(result["env"]["JEV_HOME"], str(self.install))
        for key in ("JEV_RUNTIME", "JEV_SERVER_PYTHON", "MLX_QMM_INT8"):
            self.assertIsNone(result["env"][key])


class ServerHandoffTests(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.server = load_module("jev_test_server_control", "server_control.py")
        self.selected = self.runtime.select_runtime()
        self.ready = {"service": "local-jev-mlx", "status": "ready"}
        self.enterContext(mock.patch.object(sys, "argv", ["server_control.py", "start"]))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(mock.patch.object(self.server.os, "kill", side_effect=AssertionError("A test must not signal a real process")))

    def pid_info(self, signature):
        self.server.PID_FILE.write_text(json.dumps({"pid": 123, "runtime_signature": signature}))

    def assert_lifecycle_lock_available(self):
        with (self.install / "api-control.lock").open("a") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def assert_lifecycle_lock_held(self, args):
        with self.assertRaises(BlockingIOError):
            self.assert_lifecycle_lock_available()

    def test_lifecycle_lock_spans_whole_action_and_releases_on_return(self):
        for action in ("start", "stop"):
            with self.subTest(action=action), \
                 mock.patch.object(sys, "argv", ["server_control.py", action]), \
                 mock.patch.object(self.server, "_run_action", side_effect=self.assert_lifecycle_lock_held):
                self.server.main()
                self.assert_lifecycle_lock_available()

    def test_lifecycle_lock_releases_on_action_error(self):
        def fail(args):
            self.assert_lifecycle_lock_held(args)
            raise SystemExit("test action failure")
        for action in ("start", "stop"):
            with self.subTest(action=action), \
                 mock.patch.object(sys, "argv", ["server_control.py", action]), \
                 mock.patch.object(self.server, "_run_action", side_effect=fail):
                with self.assertRaisesRegex(SystemExit, "test action failure"):
                    self.server.main()
                self.assert_lifecycle_lock_available()

    def test_changed_signature_restarts_only_managed_pid(self):
        self.pid_info("old-signature")
        with mock.patch.object(self.server, "owned_process", return_value=123), \
             mock.patch.object(self.server, "health", return_value=self.ready), \
             mock.patch.object(self.server, "stop_owned") as stop, \
             mock.patch.object(self.server.socket, "socket"), \
             mock.patch.object(self.server.subprocess, "Popen", return_value=types.SimpleNamespace(pid=456)) as popen:
            self.server.main()
        stop.assert_called_once_with(123)
        popen.assert_called_once()
        info = json.loads(self.server.PID_FILE.read_text())
        self.assertEqual(info["pid"], 456)
        self.assertEqual(info["runtime_signature"], self.runtime.signature(self.selected))

    def test_same_signature_keeps_ready_managed_server(self):
        self.pid_info(self.runtime.signature(self.selected))
        with mock.patch.object(self.server, "owned_process", return_value=123), \
             mock.patch.object(self.server, "health", return_value=self.ready), \
             mock.patch.object(self.server, "stop_owned") as stop, \
             mock.patch.object(self.server.subprocess, "Popen") as popen:
            self.server.main()
        stop.assert_not_called()
        popen.assert_not_called()

    def test_other_installation_is_left_untouched(self):
        self.pid_info("old-signature")
        with mock.patch.object(self.server, "owned_process", return_value=None), \
             mock.patch.object(self.server, "health", return_value=self.ready), \
             mock.patch.object(self.server, "stop_owned") as stop, \
             mock.patch.object(self.server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(SystemExit, "Another JEV installation"):
                self.server.main()
        stop.assert_not_called()
        popen.assert_not_called()

    def test_owned_process_accepts_exact_recipe_path(self):
        self.pid_info("old-signature")
        command = f"{self.stock_python} -m uvicorn api:app --app-dir {self.here} --host 127.0.0.1 --port 8765 --workers 1 --log-level info"
        with mock.patch.object(self.server.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, command, "")):
            self.assertEqual(self.server.owned_process(), 123)

    def test_owned_process_rejects_another_checkout_with_same_path_prefix(self):
        self.pid_info("old-signature")
        other = self.here.with_name(self.here.name + "-copy")
        command = f"{self.stock_python} -m uvicorn api:app --app-dir {other} --host 127.0.0.1 --port 8765 --workers 1 --log-level info"
        with mock.patch.object(self.server.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, command, "")):
            self.assertIsNone(self.server.owned_process())


if __name__ == "__main__":
    unittest.main()
