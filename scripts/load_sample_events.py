from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db
from app.events.calendar import load_sample_macro_events


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        count = load_sample_macro_events(db)
    finally:
        db.close()
    print(f"Loaded {count} sample macro events.")
