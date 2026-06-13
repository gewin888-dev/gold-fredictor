from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.database import SessionLocal, init_db
from app.events.calendar import list_macro_events
from app.models import GoldScoreSnapshot
from app.monitoring.health import get_data_health
from app.notifications.feishu import send_score_alert_with_health


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        snapshot = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
        if not snapshot:
            raise SystemExit("No score snapshot found. Run scripts/compute_score.py first.")
        events = [
            {
                "timestamp": row.timestamp,
                "name": row.name,
                "importance": row.importance,
            }
            for row in list_macro_events(db, days_ahead=30)
        ]
        result = send_score_alert_with_health(snapshot, get_data_health(db), events)
    finally:
        db.close()

    if result.get("skipped"):
        print(f"Feishu alert skipped: {result['reason']}")
    else:
        print("Feishu alert sent.")
