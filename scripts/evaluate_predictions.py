from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db, serialized_write
from app.scoring.gold_predictor import evaluate_due_predictions, prediction_evaluation_summary


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        with serialized_write():
            result = evaluate_due_predictions(db)
        print(result)
        print(prediction_evaluation_summary(db))
    finally:
        db.close()
