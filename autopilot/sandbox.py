"""Docker sandbox: one locked-down Linux container per task."""

from __future__ import annotations

import base64
import io
import shlex
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

from autopilot.config import SandboxConfig, load_sandbox

REPO_DIR = "/workspace/repo"
DEFAULT_TEST_CMD = "python -m pytest -q --tb=short"
MAX_OUTPUT_CHARS = 12_000
CONTAINER_UID = 1000
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
BUILD_CMD = "docker build -t autopilot-sandbox:latest -f docker/Dockerfile.sandbox ."


class SandboxError(RuntimeError):
    """Raised for sandbox setup problems (Docker down, image missing, ...)."""


@dataclass
class CmdResult:
    """Result of a command run inside the sandbox."""

    exit_code: int
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True if the command exited 0 without timing out."""
        return self.exit_code == 0 and not self.timed_out


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output so it fits in LLM context."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n\n... [{dropped} chars truncated] ...\n\n{text[-half:]}"


def _normalise_text(data: bytes) -> bytes:
    """Convert CRLF to LF for text files; leave binary files untouched."""
    if b"\0" in data[:8192]:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n")


class Sandbox:
    """A per-task Docker container. Use as a context manager."""

    def __init__(self, config: SandboxConfig | None = None, test_cmd: str | None = None) -> None:
        self.config = config or load_sandbox()
        self.test_cmd = test_cmd or DEFAULT_TEST_CMD
        self.container = None
        self.container_id: str | None = None
        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as exc:
            raise SandboxError(
                "Cannot reach Docker. Start Docker Desktop and wait until it says "
                f"'Engine running', then try again.\n(details: {exc})"
            ) from exc

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> "Sandbox":
        """Start the container (sleep infinity) with resource limits."""
        try:
            self.client.images.get(self.config.image)
        except ImageNotFound as exc:
            raise SandboxError(
                f"Sandbox image '{self.config.image}' not found. Build it from the project root:\n"
                f"  {BUILD_CMD}"
            ) from exc
        self.container = self.client.containers.run(
            self.config.image,
            command=["sleep", "infinity"],
            detach=True,
            user=f"{CONTAINER_UID}:{CONTAINER_UID}",
            working_dir="/workspace",
            mem_limit=self.config.memory,
            nano_cpus=int(self.config.cpus * 1e9),
            pids_limit=self.config.pids_limit,
            security_opt=["no-new-privileges"],
            labels={"app": "autopilot"},
        )
        self.container_id = self.container.id
        return self

    def close(self) -> None:
        """Force-remove the container. Safe to call more than once."""
        if self.container is None:
            return
        try:
            self.container.remove(force=True)
        except NotFound:
            pass
        finally:
            self.container = None

    def is_removed(self) -> bool:
        """True if the container no longer exists."""
        if self.container_id is None:
            return True
        try:
            self.client.containers.get(self.container_id)
            return False
        except NotFound:
            return True

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- commands --------------------------------------------------------

    def exec(self, cmd: str, workdir: str = REPO_DIR, timeout: int = 120) -> CmdResult:
        """Run a bash command with a hard timeout (coreutils `timeout`)."""
        if self.container is None:
            raise SandboxError("Sandbox is not started.")
        wrapped = ["timeout", "-k", "5", str(timeout), "bash", "-c", cmd]
        start = time.monotonic()
        res = self.container.exec_run(wrapped, workdir=workdir, demux=False)
        elapsed = time.monotonic() - start
        code = res.exit_code if res.exit_code is not None else -1
        output = (res.output or b"").decode("utf-8", errors="replace")
        # 124 = timeout fired; 137 = SIGKILL after the -k grace period.
        timed_out = code == 124 or (code == 137 and elapsed >= timeout)
        if timed_out:
            output += f"\n[timed out after {timeout}s]"
        return CmdResult(code, truncate(output), timed_out)

    def _git_init_commit(self, message: str) -> CmdResult:
        return self.exec(
            "git init -q && git config core.autocrlf false && git add -A && "
            f"git commit -q -m {shlex.quote(message)}"
        )

    def clone(self, repo_url: str, ref: str | None = None) -> CmdResult:
        """Clone a git repo into /workspace/repo, optionally checking out a ref."""
        cmd = f"rm -rf {REPO_DIR} && git clone -q {shlex.quote(repo_url)} {REPO_DIR}"
        if ref:
            cmd += f" && cd {REPO_DIR} && git checkout -q {shlex.quote(ref)}"
        return self.exec(cmd, workdir="/workspace", timeout=300)

    def copy_in(self, local_dir: str | Path) -> CmdResult:
        """Copy a local folder into /workspace/repo (LF endings, uid 1000) and commit it."""
        src = Path(local_dir).resolve()
        if not src.is_dir():
            raise SandboxError(f"Not a directory: {src}")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            self._add_dir(tar, "repo")
            for path in sorted(src.rglob("*")):
                rel = path.relative_to(src)
                if any(part in SKIP_DIRS for part in rel.parts):
                    continue
                arcname = str(PurePosixPath("repo", *rel.parts))
                if path.is_dir():
                    self._add_dir(tar, arcname)
                elif path.is_file():
                    data = _normalise_text(path.read_bytes())
                    info = tarfile.TarInfo(arcname)
                    info.size = len(data)
                    info.mode = 0o644
                    info.uid = info.gid = CONTAINER_UID
                    info.uname = info.gname = "agent"
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(data))
        self.exec(f"rm -rf {REPO_DIR}", workdir="/workspace")
        self.container.put_archive("/workspace", buf.getvalue())
        return self._git_init_commit("initial import")

    @staticmethod
    def _add_dir(tar: tarfile.TarFile, arcname: str) -> None:
        info = tarfile.TarInfo(arcname)
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.uid = info.gid = CONTAINER_UID
        info.uname = info.gname = "agent"
        info.mtime = int(time.time())
        tar.addfile(info)

    def install(self, timeout: int = 600) -> CmdResult:
        """Best-effort dependency install, plus pytest."""
        script = (
            "set -x; "
            "for f in requirements*.txt; do [ -f \"$f\" ] && pip install -q -r \"$f\"; done; "
            "if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install -q -e . || pip install -q .; fi; "
            "pip install -q pytest; true"
        )
        return self.exec(script, timeout=timeout)

    def run_tests(self, target: str = "", timeout: int = 300) -> CmdResult:
        """Run the test command (pytest by default), optionally on a specific target."""
        args = shlex.quote(target) if target else ""
        return self.exec(f"{self.test_cmd} {args}".strip(), timeout=timeout)

    # ---- files -----------------------------------------------------------

    def read_file(self, path: str) -> CmdResult:
        """Return file contents with line numbers."""
        return self.exec(f"cat -n -- {shlex.quote(path)}")

    def read_text(self, path: str) -> str:
        """Return a repo file's full text (not truncated), via the Docker archive API."""
        if self.container is None:
            raise SandboxError("Sandbox is not started.")
        try:
            stream, _ = self.container.get_archive(str(PurePosixPath(REPO_DIR, path)))
        except NotFound as exc:
            raise FileNotFoundError(path) from exc
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as tar:
            member = tar.next()
            if member is None or not member.isfile():
                raise IsADirectoryError(f"Not a regular file: {path}")
            handle = tar.extractfile(member)
            return handle.read().decode("utf-8", errors="replace") if handle else ""

    def exists(self, path: str) -> bool:
        """True if the path exists in the repo."""
        return self.exec(f"test -e {shlex.quote(path)}").ok

    def write_file(self, path: str, content: str) -> CmdResult:
        """Write a file safely via base64 (no heredoc/quoting issues)."""
        data = base64.b64encode(content.replace("\r\n", "\n").encode("utf-8")).decode("ascii")
        qpath = shlex.quote(path)
        res = self.exec(f"mkdir -p -- \"$(dirname -- {qpath})\" && : > {qpath}")
        if not res.ok:
            return res
        chunk = 60_000  # stay well below Linux's per-argument length limit
        for i in range(0, len(data), chunk):
            res = self.exec(f"printf '%s' {data[i:i + chunk]} | base64 -d >> {qpath}")
            if not res.ok:
                return res
        return res

    def diff(self) -> CmdResult:
        """Show the diff against the initial commit, including new files."""
        return self.exec("git add -A -N && git diff")
