"""Image-based tracking plans, independent of detection and network I/O."""

import math


class PositionController:
    """Plan bounded destinations at the camera's own speed.

    The firmware controls acceleration and braking. Detection continues while
    a move runs; later corrections use frames received after it completed.
    """

    max_age = 0.35

    def __init__(self, mapping):
        self.mapping = mapping
        self.reset()

    def reset(self):
        self.goal = None
        self.started_at = None
        self.ready_after = 0.0
        self.last_observed_at = None

    def plan(self, point, captured_at, now, position):
        valid = (point is not None and captured_at is not None
                 and 0 <= now - captured_at <= self.max_age
                 and len(point) == 2 and all(math.isfinite(p) and 0 <= p <= 1 for p in point))
        if not valid:
            return ("stop", None) if self.goal is not None else None
        if self.goal is not None:
            reached = all(abs(a-b) <= tolerance for a,b,tolerance in zip(position,self.goal,(0.003,0.006)))
            if reached:
                self.goal = None
                self.ready_after = now + 0.15
                return None
            if now - self.started_at > 3:
                raise TimeoutError("The camera did not complete its tracking movement.")
            return None
        if captured_at < self.ready_after or captured_at == self.last_observed_at:
            return None
        self.last_observed_at = captured_at
        errors = (point[0]-0.5, 0.45-point[1])
        degrees = tuple(0.0 if abs(error) < zone else max(-limit,min(limit,error*gain))
                        for error,zone,limit,gain in zip(errors,(0.065,0.08),(24,12),(60,35)))
        target = []
        for p,delta,scale,direction in zip(position,degrees,(170,35),
                                          (self.mapping.pan_right,self.mapping.tilt_up)):
            candidate = p + delta/scale*direction
            if not delta or p > 0.97 and candidate >= p or p < -0.97 and candidate <= p:
                target.append(p)
            else:
                target.append(max(-0.97,min(0.97,candidate)))
        target = tuple(target)
        if all(abs(a-b)*scale < threshold for a,b,scale,threshold in
               zip(position,target,(170,35),(1.5,1.0))):
            return None
        self.goal = target
        self.started_at = now
        return ("move", target)

