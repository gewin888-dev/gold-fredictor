from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.fred_collector import collect_fred_data
from app.database import SessionLocal, init_db


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        counts = collect_fred_data(db)
    except Exception as exc:
        raise SystemExit(f"FRED collection failed: {exc}") from exc
    finally:
        db.close()
    print("FRED collection complete.")
    for series_id, count in counts.items():
        print(f"{series_id}: {count} rows")
