"""Git-backed persistence helper for the SQLite DB (data.sqlite3).

Implements a minimal subset of the skill_18_gitstore_data_persistence playbook,
adapted for this project: when Git credentials are provided via env vars, keep a
workspace clone under data/gitstore, mirror the SQLite DB there via SQLite
backup, and periodically force-push a single-commit snapshot. Falls back to
local-only mode when Git is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GIT_TIMEOUT = 60  # seconds


class GitStoreMode:
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    LOCAL = "LOCAL"


@dataclass
class GitStoreStatus:
    mode: str
    pending: bool
    error: Optional[str]


class GitStoreSQLiteSync:
    """Handle Git persistence for the SQLite database file."""

    def __init__(self, base_dir: Path, db_filename: str = "data.sqlite3", branch: str = "main", interval: int = 30) -> None:
        self.base_dir = base_dir
        self.db_filename = db_filename
        self.branch = branch
        self.interval = interval

        self.local_db_path = base_dir / db_filename
        self.mirror_dir = base_dir / "data"
        self.workdir = self.mirror_dir / "gitstore"
        self.git_db_path = self.workdir / db_filename

        self.git_url = os.getenv("GITSTORE_GIT_URL", "").strip()
        self.git_username = os.getenv("GITSTORE_GIT_USERNAME", "").strip()
        self.git_token = os.getenv("GITSTORE_GIT_TOKEN", "").strip()
        self.git_email = f"{self.git_username or 'gitstore'}@localhost"

        self.mode: str = GitStoreMode.LOCAL
        self.pending: bool = False
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._askpass_path: Optional[Path] = None

    def _configured(self) -> bool:
        return bool(self.git_url and self.git_username and self.git_token)

    def status(self) -> GitStoreStatus:
        return GitStoreStatus(mode=self.mode, pending=self.pending, error=self.error)

    async def prepare(self) -> None:
        """Prepare gitstore workspace and restore local DB if possible."""
        if not self._configured():
            self.mode = GitStoreMode.LOCAL
            self.error = "GITSTORE_GIT_URL/GITSTORE_GIT_USERNAME/GITSTORE_GIT_TOKEN 未配置"
            return

        try:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            self.workdir.mkdir(parents=True, exist_ok=True)
            await self._ensure_askpass()
            await self._ensure_repo()
            await self._checkout_branch()
            await self._pull_latest()
            await self._ensure_identity()

            if self.git_db_path.exists():
                self._copy_file(self.git_db_path, self.local_db_path)
            elif self.local_db_path.exists():
                # Bootstrap gitstore from existing local DB
                self._copy_file(self.local_db_path, self.git_db_path)
                await self._git_add_commit_push(force=True, message="chore: init sqlite snapshot")

            self.mode = GitStoreMode.ACTIVE
            self.pending = False
            self.error = None
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.mode = GitStoreMode.LOCAL
            self.pending = False
            self.error = str(exc)

    def start_background(self) -> None:
        if self.mode == GitStoreMode.LOCAL or self._task:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop_background(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.snapshot_and_push()
            except Exception as exc:  # pragma: no cover - keep loop alive
                self.mode = GitStoreMode.DEGRADED
                self.pending = True
                self.error = str(exc)
            await asyncio.sleep(self.interval)

    async def snapshot_and_push(self) -> None:
        if self.mode == GitStoreMode.LOCAL:
            return

        try:
            self._backup_sqlite(self.local_db_path, self.git_db_path)
        except Exception as exc:
            self.mode = GitStoreMode.DEGRADED
            self.pending = True
            self.error = f"backup failed: {exc}"
            return

        changed = await self._git_has_changes()
        if not changed:
            self.pending = False
            self.error = None
            return

        try:
            await self._git_add_commit_push(force=True, message="chore: sync sqlite snapshot")
            self.mode = GitStoreMode.ACTIVE
            self.pending = False
            self.error = None
        except Exception as exc:
            self.mode = GitStoreMode.DEGRADED
            self.pending = True
            self.error = str(exc)

    # --- repo helpers -----------------------------------------------------

    async def _ensure_repo(self) -> None:
        git_dir = self.workdir / ".git"
        if git_dir.exists():
            await self._run_git(["remote", "set-url", "origin", self.git_url])
            return

        # Try clone first
        clone_proc = await self._run_git(
            ["clone", "--branch", self.branch, "--single-branch", self.git_url, str(self.workdir)],
            cwd=self.base_dir,
            allow_failure=True,
        )
        if clone_proc == 0:
            return

        # Fallback: init in-place
        await self._run_git(["init"], cwd=self.workdir)
        await self._run_git(["remote", "add", "origin", self.git_url])

    async def _checkout_branch(self) -> None:
        await self._run_git(["fetch", "origin"], cwd=self.workdir, allow_failure=True)
        # checkout -B to align with remote if present
        await self._run_git(["checkout", "-B", self.branch], cwd=self.workdir)

    async def _pull_latest(self) -> None:
        await self._run_git(["pull", "--rebase", "origin", self.branch], cwd=self.workdir, allow_failure=True)

    async def _ensure_identity(self) -> None:
        await self._run_git(["config", "user.name", self.git_username or "gitstore"], cwd=self.workdir, allow_failure=True)
        await self._run_git(["config", "user.email", self.git_email], cwd=self.workdir, allow_failure=True)

    async def _git_has_changes(self) -> bool:
        code, out, _ = await self._run_git_capture(["status", "--porcelain", "--", self.db_filename], cwd=self.workdir)
        return code == 0 and bool(out.strip())

    async def _git_add_commit_push(self, *, force: bool, message: str) -> None:
        await self._run_git(["add", self.db_filename], cwd=self.workdir)

        # Commit: amend if HEAD exists
        head_exists = await self._has_head_commit()
        commit_args = ["commit", "-m", message, "--allow-empty"]
        if head_exists:
            commit_args = ["commit", "--amend", "-m", message, "--allow-empty"]
        await self._run_git(commit_args, cwd=self.workdir, allow_failure=False)

        push_args = ["push", "origin", f"+HEAD:{self.branch}"] if force else ["push", "origin", self.branch]
        await self._run_git(push_args, cwd=self.workdir)

    async def _has_head_commit(self) -> bool:
        code = await self._run_git(["rev-parse", "--verify", "HEAD"], cwd=self.workdir, allow_failure=True)
        return code == 0

    async def _run_git(self, args: list[str], *, cwd: Optional[Path] = None, allow_failure: bool = False) -> int:
        env = os.environ.copy()
        env.update({
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": str(self._askpass_path) if self._askpass_path else "",
            "GITSTORE_USER": self.git_username,
            "GITSTORE_TOKEN": self.git_token,
        })

        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd or self.workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"git {' '.join(args)} timed out")

        if proc.returncode != 0 and not allow_failure:
            raise RuntimeError(f"git {' '.join(args)} failed: {stderr.decode().strip()}")
        return proc.returncode

    async def _run_git_capture(self, args: list[str], *, cwd: Optional[Path] = None) -> tuple[int, str, str]:
        env = os.environ.copy()
        env.update({
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": str(self._askpass_path) if self._askpass_path else "",
            "GITSTORE_USER": self.git_username,
            "GITSTORE_TOKEN": self.git_token,
        })

        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd or self.workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode()

    async def _ensure_askpass(self) -> None:
        if self._askpass_path and self._askpass_path.exists():
            return
        script_dir = self.workdir if self.workdir.exists() else self.base_dir
        fd, tmp_path = tempfile.mkstemp(prefix="gitstore_askpass_", dir=script_dir)
        os.close(fd)
        script_path = Path(tmp_path)
        # GIT calls askpass twice (user then password). Return username for the first prompt and token for password.
        script_path.write_text(
            "#!/bin/sh\n"
            "prompt=\"$1\"\n"
            "if echo \"$prompt\" | grep -qi 'username'; then\n"
            "  printf '%s' \"$GITSTORE_USER\"\n"
            "else\n"
            "  printf '%s' \"$GITSTORE_TOKEN\"\n"
            "fi\n",
            encoding="utf-8",
        )
        script_path.chmod(0o700)
        self._askpass_path = script_path

    # --- file helpers -----------------------------------------------------

    def _copy_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _backup_sqlite(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        # Use sqlite backup API for a consistent snapshot
        with sqlite3.connect(src) as src_conn:
            with sqlite3.connect(tmp) as dst_conn:
                src_conn.backup(dst_conn)
        tmp.replace(dst)


def build_gitstore_sync(base_dir: Path) -> Optional[GitStoreSQLiteSync]:
    """Factory to create sync helper only when SQLite backend is in use."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return None
    return GitStoreSQLiteSync(base_dir)
