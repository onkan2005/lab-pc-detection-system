"""Stable small numbers for unknown people ("Unknown person 1", "Unknown person 2", ...).

The number belongs to the PERSON while they are in view, not to their position in the frame, so
people walking past each other do not swap numbers. When someone leaves, their number is kept for
HOLD_SEC seconds and then reused by the next new person, so numbers stay as small as the number of
people in view instead of growing (10, 11, 12 ...).

When YOLO loses someone behind another person and they reappear as a "new" person, they get the
number of the person who vanished closest to where they reappear (within HOLD_SEC), so the label
does not change just because the tracker blinked.
"""
HOLD_SEC = 2.0


class StableNumbers:
    def __init__(self):
        self.slots = {}   # (camera, person id) -> {"n", "last", "box"}

    @staticmethod
    def _center(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def assign(self, cam, people, t):
        """people: the unknown people in view on this camera (dicts with "id" and "box").
        Returns {person id: number}."""
        ids = {p["id"] for p in people}
        mine = {k: v for k, v in self.slots.items() if k[0] == cam}
        for k, v in mine.items():                       # forget people gone longer than HOLD_SEC
            if k[1] not in ids and t - v["last"] > HOLD_SEC:
                del self.slots[k]
        used = {v["n"] for k, v in self.slots.items() if k[0] == cam}
        out = {}
        for p in sorted(people, key=lambda q: q["box"][0]):   # new people numbered left to right
            key = (cam, p["id"])
            slot = self.slots.get(key)
            if slot is None:
                slot = self._take_over(cam, p, ids, t) or {"n": self._lowest_free(used)}
                used.add(slot["n"])
            slot.update(last=t, box=p["box"])
            self.slots[key] = slot
            out[p["id"]] = slot["n"]
        return out

    def _take_over(self, cam, p, ids, t):
        """The number of a person who just vanished near this spot (the tracker lost and re-found them)."""
        cx, cy = self._center(p["box"])
        reach = 1.5 * max(1, p["box"][2] - p["box"][0])
        best, best_d = None, reach
        for k, v in self.slots.items():
            if k[0] != cam or k[1] in ids or t - v["last"] > HOLD_SEC:
                continue
            vx, vy = self._center(v["box"])
            d = ((cx - vx) ** 2 + (cy - vy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = k, d
        return self.slots.pop(best) if best else None

    @staticmethod
    def _lowest_free(used):
        n = 1
        while n in used:
            n += 1
        return n
