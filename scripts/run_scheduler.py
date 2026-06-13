from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import init_db
from app.scheduler import create_scheduler


def main() -> None:
    init_db()
    scheduler = create_scheduler()
    scheduler.start()
    print("Scheduler started. Press Ctrl+C to stop.", flush=True)
    print("Registered jobs:", flush=True)
    for job in scheduler.get_jobs():
        print(f"- {job.id}: next_run_time={job.next_run_time}", flush=True)

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    try:
        while not stop:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
        print("Scheduler stopped.", flush=True)


if __name__ == "__main__":
    main()
