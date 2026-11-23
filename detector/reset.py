"""Reset the app's data without touching the student photos, then re-enroll everyone.

Deletes (inside the data folder):
  PCs and assignments, watchlist, incidents + snapshots, usage, room (CCTV) and alert logs,
  face embeddings and the enrollment report, lighting and CCTV test results.
Keeps:
  enrollment/<name>/ photos  (they are enrolled again automatically)

Usage (stop the app first):
  python reset.py               # asks for confirmation
  python reset.py --keep-pcs    # keep PCs, assignments and the watchlist
  python reset.py --yes         # no confirmation
  python reset.py --no-enroll   # don't re-enroll afterwards
Docker:
  docker compose down
  docker compose run --rm pc-usage-monitor python reset.py
  docker compose up -d
"""
import argparse
import os
import shutil
import sys
import urllib.request

import config

KEEP_WITH_PCS = {config.ASSIGNMENTS_FILE.name, config.WATCHLIST_FILE.name}


def app_is_running():
    """True if the web app answers on its port (it would write the old data back from memory)."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('PCMON_PORT', 8765)}/api/state", timeout=1)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-pcs", action="store_true", help="keep PCs, assignments and the watchlist")
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    ap.add_argument("--no-enroll", action="store_true", help="don't re-enroll the photos afterwards")
    ap.add_argument("--force", action="store_true", help="run even if the app seems to be running")
    args = ap.parse_args()

    if app_is_running() and not args.force:
        sys.exit("The app is still running. Stop it first (docker compose down, or Ctrl+C / pkill -f web.py), "
                 "otherwise it writes the old data back.")

    data = config.DATA_DIR
    items = sorted(p for p in data.iterdir() if not (args.keep_pcs and p.name in KEEP_WITH_PCS)) if data.exists() else []
    people = sorted(p.name for p in config.ENROLLMENT_DIR.iterdir() if p.is_dir()) if config.ENROLLMENT_DIR.exists() else []

    print(f"Data folder:      {data}")
    print("Will delete:      " + (", ".join(p.name for p in items) if items else "(nothing)"))
    print("Will keep:        photos of " + (", ".join(people) if people else "nobody")
          + (" + PCs/assignments/watchlist" if args.keep_pcs else ""))
    print("Then:             " + ("nothing" if args.no_enroll else "re-enroll everyone from the photos"))
    if not args.yes and input("Continue? Type 'yes': ").strip().lower() != "yes":
        sys.exit("Cancelled, nothing deleted.")

    for p in items:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"Deleted {len(items)} item(s).")

    if not args.no_enroll and people:
        from enroll import enroll
        from faces import load_face_app
        print("Re-enrolling (this can take a minute)...")
        report = enroll(load_face_app())
        for e in report:
            print(f"  {e['name']}: {e['used']}/{len(e['photos'])} photos usable")
    print("Done. Start the app again.")


if __name__ == "__main__":
    main()
