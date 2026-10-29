from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def main() -> int:
    files = [PROJECT_ROOT / "app.py", *sorted((PROJECT_ROOT / "pages").glob("*.py"))]
    rows = []
    for path in files:
        app = AppTest.from_file(str(path)).run(timeout=60)
        row = {
            "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "exceptions": [str(item.value) for item in app.exception],
            "titles": [str(item.value) for item in app.title],
            "warnings": [str(item.value) for item in app.warning],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    if any(row["exceptions"] for row in rows):
        raise AssertionError("Streamlit AppTest có exception")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_tested": len(rows),
        "files_without_exceptions": sum(not row["exceptions"] for row in rows),
        "results": rows,
    }
    output = PROJECT_ROOT / "data" / "verification" / "phase9_streamlit_apptest.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
