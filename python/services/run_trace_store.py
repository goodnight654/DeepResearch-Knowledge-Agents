"""
Run Trace 存储服务 — 持久化 QA / DeepResearch 执行轨迹
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import settings


class RunTraceStore:
    """将执行轨迹保存为 JSON 文件，便于回放与审计。"""

    SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{6,80}$")

    def __init__(self, base_dir: str | None = None) -> None:
        path = base_dir or settings.run_trace_dir
        self.base_dir = Path(path).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        workflow: str,
        input_payload: dict[str, Any],
        trace: list[dict[str, Any]],
        output: dict[str, Any],
        status: str = "success",
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        run_id = self._generate_run_id(workflow)
        record = {
            "run_id": run_id,
            "workflow": workflow,
            "status": status,
            "created_at": self._now_iso(),
            "input": input_payload,
            "trace": trace,
            "output": output,
            "metadata": metadata or {},
            "error": error,
        }
        file_path = self.base_dir / f"{run_id}.json"
        temporary = self.base_dir / f".{run_id}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(file_path)
        finally:
            temporary.unlink(missing_ok=True)
        return run_id

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.SAFE_RUN_ID.match(run_id):
            return None
        file_path = self.base_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def list_runs(
        self,
        limit: int = 20,
        workflow: str | None = None,
    ) -> list[dict[str, Any]]:
        max_items = min(max(1, int(limit)), 200)
        files = sorted(
            self.base_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        rows: list[dict[str, Any]] = []
        for path in files:
            if len(rows) >= max_items:
                break
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            wf = str(raw.get("workflow", ""))
            if workflow and wf != workflow:
                continue
            output = raw.get("output", {}) if isinstance(raw.get("output"), dict) else {}
            payload = raw.get("input", {}) if isinstance(raw.get("input"), dict) else {}
            rows.append(
                {
                    "run_id": raw.get("run_id", path.stem),
                    "workflow": wf or "unknown",
                    "status": raw.get("status", "unknown"),
                    "created_at": raw.get("created_at", ""),
                    "question": payload.get("question") or output.get("question") or "",
                    "confidence": output.get("confidence"),
                    "iterations": output.get("iterations"),
                }
            )
        return rows

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_run_id(workflow: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token = uuid4().hex[:10]
        safe_workflow = re.sub(r"[^a-zA-Z0-9_-]", "", workflow)[:24] or "run"
        return f"{safe_workflow}_{ts}_{token}"
