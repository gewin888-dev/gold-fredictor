from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.cftc_collector import collect_cftc_gold_position
from app.database import SessionLocal, init_db


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        record = collect_cftc_gold_position(db)
    except Exception as exc:
        raise SystemExit(f"CFTC collection failed: {exc}") from exc
    finally:
        db.close()
    print("CFTC collection complete.")
    print(f"market={record.market_name}")
    print(f"timestamp={record.timestamp}")
    print(f"open_interest={record.open_interest}")
    print(f"noncommercial_net={record.noncommercial_net}")
