"""Tests for the macOS libomp auto-install helper and DeepDoc degradation.

These tests never touch a real package manager: subprocess and platform checks
are patched so the logic can be exercised on any OS/CI.
"""

import subprocess
from unittest.mock import patch

import pytest

from xagent.providers.pdf_parser import _libomp


class TestLooksLikeMissingLibomp:
    @pytest.mark.parametrize(
        "message",
        [
            "XGBoost Library (libxgboost.dylib) could not be loaded.",
            "dlopen(...): Library not loaded: libomp.dylib",
            "OpenMP runtime is not installed",
            "Referenced from: .../libxgboost.dylib",
        ],
    )
    def test_matches_libomp_failures(self, message: str) -> None:
        assert _libomp.looks_like_missing_libomp(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "some unrelated deepdoc init failure",
            "No module named 'pandas'",
            "",
        ],
    )
    def test_ignores_unrelated_failures(self, message: str) -> None:
        assert _libomp.looks_like_missing_libomp(RuntimeError(message)) is False


class TestTryInstallLibomp:
    def test_noop_off_macos(self) -> None:
        with patch.object(_libomp, "is_macos", return_value=False):
            assert _libomp.try_install_libomp() is False

    def test_respects_opt_out_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", "1")
        with patch.object(_libomp, "is_macos", return_value=True):
            assert _libomp.try_install_libomp() is False

    def test_conda_force_reinstalls_into_active_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/anaconda3/envs/xagent")
        monkeypatch.setenv("CONDA_EXE", "/opt/anaconda3/bin/conda")

        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(_libomp, "is_macos", return_value=True),
            patch("os.path.exists", return_value=True),
            # libomp verified present on disk after the install.
            patch.object(_libomp, "libomp_present", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            assert _libomp.try_install_libomp() is True

        assert seen, "expected an install command to run"
        cmd = seen[0]
        assert cmd[0] == "/opt/anaconda3/bin/conda"
        # Must force a reinstall and target THIS env explicitly, otherwise a
        # missing libomp.dylib is never restored.
        assert "install" in cmd
        assert "--force-reinstall" in cmd
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "/opt/anaconda3/envs/xagent"
        assert "llvm-openmp" in cmd

    def test_conda_success_code_but_file_missing_is_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # conda can exit 0 ("already installed") while the file is still gone;
        # and with no package manager fallback available, that must be a failure.
        monkeypatch.delenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/anaconda3/envs/xagent")
        monkeypatch.setenv("CONDA_EXE", "/opt/anaconda3/bin/conda")

        def ok_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(_libomp, "is_macos", return_value=True),
            patch("os.path.exists", return_value=True),
            patch.object(_libomp, "libomp_present", return_value=False),
            patch("shutil.which", return_value=None),  # no brew fallback
            patch("subprocess.run", side_effect=ok_run),
        ):
            assert _libomp.try_install_libomp() is False

    def test_non_conda_falls_back_to_brew(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)

        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(_libomp, "is_macos", return_value=True),
            patch("shutil.which", return_value="/opt/homebrew/bin/brew"),
            patch.object(_libomp, "libomp_present", return_value=True),
            patch("subprocess.run", side_effect=fake_run),
        ):
            assert _libomp.try_install_libomp() is True

        assert seen and seen[0][0] == "/opt/homebrew/bin/brew"
        assert "install" in seen[0] and "libomp" in seen[0]

    def test_returns_false_when_install_command_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)

        def failing_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        with (
            patch.object(_libomp, "is_macos", return_value=True),
            patch("shutil.which", return_value="/opt/homebrew/bin/brew"),
            patch.object(_libomp, "libomp_present", return_value=False),
            patch("subprocess.run", side_effect=failing_run),
        ):
            assert _libomp.try_install_libomp() is False

    def test_returns_false_when_no_package_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAGENT_DISABLE_LIBOMP_AUTOINSTALL", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        with (
            patch.object(_libomp, "is_macos", return_value=True),
            patch("shutil.which", return_value=None),
        ):
            assert _libomp.try_install_libomp() is False


class TestManualFixHint:
    def test_conda_hint_targets_active_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONDA_PREFIX", "/opt/anaconda3/envs/xagent")
        assert _libomp.manual_fix_hint() == (
            "conda install -y --force-reinstall "
            "-p /opt/anaconda3/envs/xagent llvm-openmp"
        )

    def test_brew_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        assert _libomp.manual_fix_hint() == "brew install libomp"
