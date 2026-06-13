from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db
from app.scoring.gold_score import compute_and_store_gold_score


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        snapshot = compute_and_store_gold_score(db)
    except Exception as exc:
        raise SystemExit(f"Gold score computation failed: {exc}") from exc
    finally:
        db.close()
    print("Gold score computed.")
    print(f"timestamp={snapshot.timestamp}")
    print(f"total_score={snapshot.total_score}")
    print(f"direction={snapshot.direction}")
    print(f"summary={snapshot.summary}")
