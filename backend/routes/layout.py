from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from typing import Any, Dict
from threading import RLock

router = APIRouter()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True, parents=True)
_LAYOUT_FILE = _DATA_DIR / "layout_state.json"
_lock = RLock()


def _read_layout() -> Dict[str, Any]:
    if not _LAYOUT_FILE.exists():
        return {"layout": None, "version": None}
    try:
        with _LAYOUT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"layout": None, "version": None}


def _write_layout(payload: Dict[str, Any]) -> None:
    tmp = _LAYOUT_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(_LAYOUT_FILE)


@router.get("/layout")
def get_layout() -> Dict[str, Any]:
    with _lock:
        return _read_layout()


@router.post("/layout")
def save_layout(data: Dict[str, Any]):
    if "layout" not in data:
        raise HTTPException(status_code=400, detail="Missing 'layout' key")
    record = {
        "layout": data["layout"],
        "version": data.get("version"),
        "_schemaVersion": data.get("_schemaVersion"),
    }
    with _lock:
        _write_layout(record)
    return {"status": "ok"}
