from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool import chrome_daemon_runner


def test_validate_server_args_requires_isolation_and_blocks_managed_overrides():
    assert chrome_daemon_runner._validate_server_args(
        ["--headless", "--isolated", "--no-usage-statistics"]
    ) == ["--headless", "--isolated", "--no-usage-statistics"]
    for args in (
        ["--headless"],
        ["--isolated"],
        ["--headless", "--isolated", "--sessionId=other"],
        ["--headless", "--isolated", "--user-data-dir", "/shared"],
    ):
        with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError):
            chrome_daemon_runner._validate_server_args(args)


def test_socket_request_uses_nul_framing_and_parses_one_response():
    client = MagicMock()
    client.__enter__.return_value = client
    client.recv.return_value = b'{"success":true,"result":"{}"}\0'
    with patch.object(socket, "socket", return_value=client) as constructor:
        response = chrome_daemon_runner._socket_request(
            Path("/tmp/daemon.sock"),
            {"method": "status"},
        )

    assert response == {"success": True, "result": "{}"}
    constructor.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect.assert_called_once_with("/tmp/daemon.sock")
    client.sendall.assert_called_once_with(b'{"method":"status"}\0')


def test_start_uses_exact_pin_private_environment_and_health_check(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    runtime.mkdir()
    profile.mkdir()
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    healthy = {
        "version": "1.6.0",
        "pid": 42,
        "args": [
            "--headless",
            "--isolated",
            "--viaCli",
            "--experimentalStructuredContent",
        ],
    }
    completed = MagicMock(returncode=0)

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=(
                {
                    "NPM_CONFIG_CACHE": "/opt/npm-cache",
                    "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
                },
                runtime,
                profile,
            ),
        ),
        patch.object(
            chrome_daemon_runner, "_runtime_paths", return_value=(socket_path, pid_file)
        ),
        patch.object(chrome_daemon_runner, "_status", side_effect=[None, healthy]),
        patch.object(
            chrome_daemon_runner.subprocess, "run", return_value=completed
        ) as run,
    ):
        assert (
            chrome_daemon_runner._start("a" * 32, ["--headless", "--isolated"])
            == healthy
        )

    args = run.call_args.args[0]
    assert args[:6] == [
        "npx",
        "-y",
        "--prefer-offline",
        "--package",
        "chrome-devtools-mcp@1.6.0",
        "chrome-devtools",
    ]
    assert args[-2:] == ["--headless", "--isolated"]
    assert run.call_args.kwargs["env"]["CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS"] == "1"


def test_start_fails_closed_for_healthy_daemon_with_wrong_launch_spec(tmp_path):
    runtime = tmp_path / "runtime"
    profile = tmp_path / "profile"
    daemon_home = runtime / f"chrome-devtools-mcp-{'a' * 32}"
    daemon_home.mkdir(parents=True)
    profile.mkdir()
    socket_path, pid_file = chrome_daemon_runner._runtime_paths("a" * 32, runtime)
    pid_file.write_text("42")
    wrong = {"version": "1.6.0", "args": ["--headless"]}

    with (
        patch.object(
            chrome_daemon_runner,
            "_session_environment",
            return_value=({}, runtime, profile),
        ),
        patch.object(chrome_daemon_runner, "_status", return_value=wrong),
        patch.object(chrome_daemon_runner, "_terminate_expected_daemon") as terminate,
        patch.object(
            chrome_daemon_runner.subprocess,
            "run",
            return_value=MagicMock(returncode=0),
        ) as run,
    ):
        with pytest.raises(
            chrome_daemon_runner.ChromeDaemonRunnerError,
            match="does not match",
        ):
            chrome_daemon_runner._start("a" * 32, ["--headless", "--isolated"])

    terminate.assert_not_called()
    run.assert_not_called()


def test_private_dir_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(chrome_daemon_runner.ChromeDaemonRunnerError, match="symlink"):
        chrome_daemon_runner._private_dir(link)


def test_terminate_refuses_unverified_pid(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(os.getpid()))
    with pytest.raises(
        chrome_daemon_runner.ChromeDaemonRunnerError, match="unverified"
    ):
        chrome_daemon_runner._terminate_expected_daemon(pid_file)
