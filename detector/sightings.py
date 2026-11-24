"""Shared between all cameras: where and how each student was last seen BY THEIR FACE.

Each camera keeps its own short-term re-ID memory (reid.py), because the same clothes look different
from another angle and in other light. This board lets the cameras help each other a little:

  - Hints: a person whose face no camera has seen yet, whose clothes clearly match a student whom
    ANOTHER camera recognized by face in the last HINT_SEC, is shown as "maybe <name>?". A hint is only
    shown, never logged or used for alerts; the name becomes real only when this camera sees the face.
  - Where is a student: the last camera that saw them, used for the "assigned PC is empty, but the
    student is in the room" note.
"""
from reid import clothes_similarity

import config


class Sightings:
    def __init__(self):
        self.last = {}    # name -> {"cam", "t", "sig"}

    def record(self, name, cam, sig, t):
        """A face-confirmed sighting (not a re-ID or a hint)."""
        self.last[name] = {"cam": cam, "t": t, "sig": sig}

    def hint(self, cam, sig, t):
        """{"name", "camera", "sim", "ago"} or None: a student recently recognized on another camera
        whose clothes match clearly better than anyone else's."""
        if sig.get("upper") is None:
            return None    # no shirt visible: nothing to compare
        scores = sorted(((clothes_similarity(s["sig"], sig), name, s) for name, s in self.last.items()
                         if s["cam"] != cam and t - s["t"] <= config.HINT_SEC and s["sig"].get("upper") is not None),
                        key=lambda x: -x[0])
        if not scores or scores[0][0] < config.HINT_SIM:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < config.HINT_MARGIN:
            return None    # two students look alike: no hint rather than a wrong one
        sim, name, s = scores[0]
        return {"name": name, "camera": s["cam"], "sim": round(sim, 2), "ago": round(t - s["t"])}

    def where(self, name, t, within):
        s = self.last.get(name)
        if s is None or t - s["t"] > within:
            return None
        return {"camera": s["cam"], "ago": round(t - s["t"])}
