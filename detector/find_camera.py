"""Lists which camera indexes work, so you know what to pass to --source (e.g. Iriun)."""
import sys

import cv2

backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
for i in range(5):
    cap = cv2.VideoCapture(i, backend)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    print(f"Camera {i}: " + (f"OK {frame.shape[1]}x{frame.shape[0]}" if ok else "not available"))
    cap.release()
