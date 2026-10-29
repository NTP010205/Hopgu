from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.llm_service import OllamaService


def main() -> int:
    service = OllamaService(timeout=5.0)
    status = service.status()
    if not status.ready:
        raise RuntimeError(status.detail)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "check": "real configured Ollama endpoint after controlled fallback test",
        "base_url": service.base_url,
        "model": service.model,
        "status": {
            "ready": status.ready,
            "mode": status.mode,
            "detail": status.detail,
        },
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_ollama_ready_after_fallback.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
