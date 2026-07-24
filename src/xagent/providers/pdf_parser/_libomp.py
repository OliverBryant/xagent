"""macOS OpenMP runtime (libomp) auto-repair for the DeepDoc parser.

``deepdoc`` transitively depends on ``xgboost``, whose native library
(``libxgboost.dylib``) fails to load on macOS unless the OpenMP runtime
(``libomp.dylib``) is present. Linux ships ``libgomp.so`` as part of the
toolchain, so this problem is macOS-only.

When the DeepDoc import fails on macOS because of a missing ``libomp``, we try
to install it automatically. Installation strategy, in order:

1. In a conda environment:
   ``conda install -y --force-reinstall -p $CONDA_PREFIX llvm-openmp``. This is
   the most reliable option because conda drops ``libomp.dylib`` into
   ``$CONDA_PREFIX/lib``, which is on xgboost's dlopen fallback search path.
   ``--force-reinstall`` and an explicit ``-p`` are required: a plain
   ``conda install`` targets the base env and no-ops when the package DB already
   lists llvm-openmp, so it would not restore a deleted ``libomp.dylib``.
2. Otherwise, best-effort ``brew install libomp`` (Homebrew's default prefix is
   where xgboost's baked-in rpath points, e.g.
   ``/opt/homebrew/opt/libomp/lib``).

After installing we verify ``libomp.dylib`` actually exists on disk; a
zero-exit package manager is not trusted as proof.

Everything here is best-effort: any failure is logged and swallowed so the
caller can fall back to degrading the DeepDoc parser to unavailable rather than
crashing startup. The install cannot fix the *current* process (xgboost caches
its failed native load at first import), so a successful install takes effect
on the next start.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# Time budget for the install command. libomp is small, but conda/brew metadata
# resolution can be slow; keep this generous but bounded so a hung package
# manager can't wedge startup indefinitely.
_INSTALL_TIMEOUT_SECONDS = 600


def is_macos() -> bool:
    return sys.platform == "darwin"


def looks_like_missing_libomp(exc: BaseException) -> bool:
    """Heuristic: does this import failure look like the missing-libomp case?

    We match on the exception text rather than the type because xgboost raises
    its own ``XGBoostError`` (a ``ValueError`` subclass) while the underlying
    ``dlopen`` failure surfaces as an ``OSError`` in other paths. Both mention
    the OpenMP runtime / libomp / libxgboost, which is what we key on.
    """
    text = str(exc).lower()
    needles = ("libomp", "libxgboost", "openmp runtime", "libxgboost.dylib")
    return any(n in text for n in needles)


def _in_conda_env() -> bool:
    # CONDA_PREFIX is set for both `conda activate`d and `conda run` contexts.
    return bool(os.environ.get("CONDA_PREFIX"))


def _run(cmd: list[str]) -> bool:
    """Run an install command, returning True on success. Never raises."""
    logger.info("Attempting to install OpenMP runtime: %s", " ".join(cmd))
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        logger.debug("Install command not found: %s", cmd[0])
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Timed out installing OpenMP runtime via '%s'.", cmd[0])
        return False
    except Exception as exc:  # noqa: BLE001 - stay best-effort
        logger.warning("Failed to run '%s': %s", cmd[0], exc)
        return False

    if proc.returncode == 0:
        logger.info("OpenMP runtime installed via '%s'.", cmd[0])
        return True

    logger.warning(
        "'%s' exited with code %s while installing OpenMP runtime.\nstdout: %s\nstderr: %s",
        cmd[0],
        proc.returncode,
        (proc.stdout or "").strip(),
        (proc.stderr or "").strip(),
    )
    return False


def _conda_executable() -> str | None:
    """Best path to the conda executable for the active environment."""
    exe = os.environ.get("CONDA_EXE")
    if exe and os.path.exists(exe):
        return exe
    import shutil

    return shutil.which("conda") or shutil.which("mamba")


def libomp_present() -> bool:
    """Best-effort check for a loadable libomp.dylib on disk.

    xgboost reports success/failure by attempting to ``dlopen`` its native
    library; a package manager reporting "already installed" is not proof the
    file exists (a conda package can be recorded as installed while its files
    were removed). We therefore verify the actual ``libomp.dylib`` on the paths
    xgboost's loader searches: the active conda env's ``lib`` directory and
    Homebrew's libomp prefix (xgboost's baked-in rpath).
    """
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib", "libomp.dylib"))
    # Homebrew default prefixes (Apple Silicon and Intel).
    candidates.append("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    candidates.append("/usr/local/opt/libomp/lib/libomp.dylib")
    return any(os.path.exists(p) for p in candidates)


def try_install_libomp() -> bool:
    """Attempt to install libomp on macOS.

    Returns True only if, after the install, ``libomp.dylib`` is actually
    present on disk where xgboost's loader can find it. A zero exit code alone
    is not trusted: conda reports "already installed" (and does nothing) when
    its package DB records llvm-openmp even if the file is missing, so we force
    a reinstall into the *active* environment and then verify the file.

    Note: installing here does not fix the *current* process. xgboost's native
    extension is loaded (and cached as failed) at first import; a subprocess
    install cannot be picked up by an already-failed C extension in-process.
    The caller should treat a True result as "fixed on next start".

    Only runs on macOS. Prefers conda (installs libomp into the env's own
    ``lib`` dir, which is on xgboost's dlopen fallback path), falling back to
    Homebrew.
    """
    if not is_macos():
        return False

    # Allow operators to opt out of any automatic package installation.
    if os.environ.get("XAGENT_DISABLE_LIBOMP_AUTOINSTALL"):
        logger.info(
            "Skipping automatic libomp install "
            "(XAGENT_DISABLE_LIBOMP_AUTOINSTALL is set)."
        )
        return False

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if _in_conda_env():
        conda = _conda_executable()
        if conda:
            # Force a reinstall into THIS env explicitly (-p). A plain
            # `conda install` targets the base env and/or no-ops when the
            # package DB already lists llvm-openmp, neither of which restores a
            # missing libomp.dylib in the active env.
            cmd = [conda, "install", "-y", "--force-reinstall"]
            if conda_prefix:
                cmd += ["-p", conda_prefix]
            cmd.append("llvm-openmp")
            _run(cmd)
            if libomp_present():
                return True
            logger.warning(
                "conda reported success but libomp.dylib is still missing in "
                "the active environment."
            )
        else:
            logger.warning(
                "In a conda environment but no conda/mamba executable found; "
                "cannot auto-install llvm-openmp."
            )

    # Fallback (or non-conda envs): Homebrew. xgboost's baked-in rpath points at
    # Homebrew's libomp prefix, so this works for Homebrew-based Python too.
    import shutil

    brew = shutil.which("brew")
    if brew:
        _run([brew, "install", "libomp"])
        if libomp_present():
            return True
        logger.warning(
            "brew install libomp did not produce a loadable libomp.dylib on a "
            "path xgboost searches. On Apple Silicon, ensure you are using the "
            "arm64 Homebrew at /opt/homebrew."
        )
    else:
        logger.debug("Homebrew not found; cannot auto-install libomp via brew.")

    return False


def manual_fix_hint() -> str:
    """A copy-pasteable hint tailored to the current environment."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if _in_conda_env():
        if conda_prefix:
            return f"conda install -y --force-reinstall -p {conda_prefix} llvm-openmp"
        return "conda install -y --force-reinstall llvm-openmp"
    return "brew install libomp"
