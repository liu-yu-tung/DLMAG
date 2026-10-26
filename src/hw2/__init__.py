import subprocess

from hw2.remi import HW2_ROOT


def code_version() -> str:
    """Short commit hash, '+dirty' when code under hw2/ differs from it; 'unknown' outside a git checkout."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=HW2_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", "src", "scripts", "Makefile", "pyproject.toml")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")
