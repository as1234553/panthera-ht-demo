"""The task program: pick the cable housing out of the J-hook and plug it
into the port, using the wrist camera and the two ArUco tags.

Sequence
  1. Hook survey: look where the hook tag should be (searching other view
     directions if it is not seen), then centre the tag in the image from a
     30 deg oblique view (sharper rotation estimates than square-on) and
     re-measure until the estimate stops moving. Locate the housing in the
     slot from its green front flange (it may have rolled along the channel
     if the hook is tilted).
  2. Port survey: turn round and do the same for the port tag.
  3. Plan the whole job on the measured poses before touching anything:
     grasp + lift-out (IK along the J, grasp roll searched), the 180 deg
     swing, the port view with the housing in hand, and the insertion
     (roll about the bore axis searched). If any part is unreachable or
     collides, abort safely with the arm empty.
  4. Execute: approach, grasp, lift out along the J, rotate 180 deg, centre on
     the port tag again (housing in hand), re-plan the insertion on the
     refined estimate, compliant insertion with the compliance centre at the
     plug tip (remote-centre-compliance style), release, retreat, home.

The program only uses: joint readings, camera images, the robot model, the
hand-eye calibration, and the known geometry of the hook, port and housing.
"""

from __future__ import annotations

import math
import time

import numpy as np

from geometry import (P, Params, Pose, camera_intrinsics, hook_nominal,
                      neck_free_region, port_nominal, rot_angle, slot_in_hook,
                      tcp_to_cam, tilted, tip_in_tcp)
from kin import GRIP_CLOSED_ON_GROOVE, GRIP_OPEN, Kin, interp_poses
from planner import Planner
from vision import TagDetector, average_poses, locate_housing

INSERT_GAINS = dict(kp=[900.0, 250.0, 250.0], kr=8.0, dp=[45.0, 25.0, 25.0],
                    dr=0.6)


class Abort(Exception):
    def __init__(self, kind, reason):
        super().__init__(f"{kind}: {reason}")
        self.kind, self.reason = kind, reason


class TaskProgram:
    def __init__(self, robot, xml, p: Params = P, verbose=True):
        self.r = robot
        self.p = p
        self.k = Kin(xml, p)
        self.pl = Planner(self.k, p)
        self.det = TagDetector(p)
        self.K = camera_intrinsics(p)
        self.T_tcp_cam = tcp_to_cam(p)
        self.verbose = verbose
        self.res = dict(outcome=None, kind=None, reason="", est={}, plan={},
                        timing={})
        self._last_images = []

    def log(self, *a):
        if self.verbose:
            print("  [program]", *a, flush=True)

    # ================================================================ sensing
    def T_base_cam(self):
        return self.k.fk(self.r.q()) @ self.T_tcp_cam

    def measure(self, which, n=3):
        """Tag pose from n images at the current pose (averaged)."""
        tag_id = self.p.hook_tag_id if which == "hook" else self.p.port_tag_id
        T_cam = self.T_base_cam()
        ests, diags, imgs = [], [], []
        for _ in range(n):
            img = self.r.capture()
            est, dg = self.det.fixture_pose(img, tag_id, T_cam)
            if est is not None:
                ests.append(est)
                diags.append(dg)
                imgs.append((img, T_cam))
        if not ests:
            return None, None
        self._last_images = imgs
        dg = {key: np.mean([dd[key] for dd in diags], axis=0) for key in diags[0]}
        return average_poses(ests), dg

    def find_tag(self, which, holding):
        """Look where the tag should be; if it is not visible, try views from
        other directions (the tag centre is known, its tilt is not)."""
        T_nom = hook_nominal(self.p) if which == "hook" else port_nominal(self.p)
        guesses = [T_nom] + [tilted(T_nom, 0, math.radians(a), math.radians(b))
                             for a, b in ((0, 40), (0, -40), (40, 0), (-40, 0),
                                          (35, 35), (35, -35), (-35, 35),
                                          (-35, -35))]
        for i, g in enumerate(guesses):
            v = self.pl.plan_view(g, self.r.q(), holding)
            if not v.ok:
                continue
            self.r.move(v.data["vias"], label=f"{which} survey")
            est, dg = self.measure(which, n=1)
            if est is not None:
                self.log(f"{which} tag found on look {i + 1}")
                self.res.setdefault("looks", {})[which] = i + 1
                return est
        raise Abort("tag not found", f"{which} tag not seen from any survey view")

    def center(self, which, T_est, holding, max_iter=3):
        """Centre the tag in the image (oblique view) and re-measure until the
        estimate settles."""
        diag = None
        for it in range(max_iter):
            v = self.pl.plan_view(T_est, self.r.q(), holding)
            if not v.ok:
                if it == 0:
                    self.log(f"cannot centre on the {which} tag ({v.reason});"
                             " keeping the survey estimate")
                    est, diag = self.measure(which, n=3)
                    if est is not None:
                        T_est = est
                break
            self.r.move(v.data["vias"], label=f"{which} centre")
            est, diag = self.measure(which, n=3)
            if est is None:
                raise Abort("tag lost", f"{which} tag not seen after centring")
            dp = float(np.linalg.norm(est.t - T_est.t))
            dr = math.degrees(rot_angle(est.R @ T_est.R.T))
            c = diag["centre_px"] - self.K[:2, 2]
            self.log(f"{which} centre #{it + 1}: tag {np.linalg.norm(c):.0f} px "
                     f"off centre, estimate moved {dp * 1e3:.2f} mm / {dr:.2f} deg")
            T_est = est
            if dp < 0.001 and dr < 0.3 and np.linalg.norm(c) < 40:
                break
        return T_est, diag

    def locate_housing(self, T_hook):
        T_slot = T_hook @ slot_in_hook(self.p)
        ys = []
        for img, T_cam in self._last_images:
            y, _ = locate_housing(img, self.K, T_cam, T_slot, self.p)
            if y is not None:
                ys.append(y)
        reg = neck_free_region(self.p)[1]          # the channel
        if ys:
            y = float(np.clip(np.median(ys), reg[0], reg[1]))
            src = "camera"
        else:
            # Fallback: it rolls to the low end of the channel under gravity.
            g = T_slot.R.T @ np.array([0, 0, -1.0])
            y = reg[1] if g[1] > 0.02 else 0.0
            src = "gravity"
        self.log(f"housing neck at slot y = {y * 1e3:.1f} mm ({src})")
        self.res["est"]["y_neck_src"] = src
        return y

    # ================================================================ run
    def run(self):
        t0 = time.time()
        try:
            self._run()
            self.res["outcome"] = "success" if self.res.get("seated") else "failure"
        except Abort as a:
            self.res["kind"], self.res["reason"] = a.kind, a.reason
            safe = a.kind in ("unreachable", "tag not found", "tag lost")
            self.res["outcome"] = "safe_abort" if safe else "failure"
            self.log(f"ABORT ({a.kind}): {a.reason}")
            try:
                self._go_home(holding=self.res.get("holding", False))
            except Exception as e:                          # noqa: BLE001
                self.log(f"could not return home: {e}")
        self.res["timing"]["wall_s"] = time.time() - t0
        return self.res

    def _go_home(self, holding):
        if holding:
            return                      # stay put with the housing in hand
        self.k.grip = GRIP_OPEN
        vias, _ = self.pl.connect(self.r.q(), np.array(self.p.home_q))
        if vias is not None:
            self.r.move(vias, label="home")

    def _run(self):
        r, k, pl, p = self.r, self.k, self.pl, self.p
        k.set_fixture("hook", hook_nominal(p))
        k.set_fixture("port", port_nominal(p))
        k.set_peg_free(pl.peg_rest_pose(hook_nominal(p) @ slot_in_hook(p), 0.0))

        # ---------------- 1. hook survey
        T_hook = self.find_tag("hook", holding=False)
        k.set_fixture("hook", T_hook)
        T_hook, _ = self.center("hook", T_hook, holding=False)
        k.set_fixture("hook", T_hook)
        y_neck = self.locate_housing(T_hook)
        T_slot = T_hook @ slot_in_hook(p)
        k.set_peg_free(pl.peg_rest_pose(T_slot, y_neck))
        self.res["est"].update(T_hook=T_hook, y_neck=y_neck)
        q_hook_view = r.q()

        # ---------------- 2. port survey (empty-handed)
        T_port = self.find_tag("port", holding=False)
        k.set_fixture("port", T_port)
        T_port, _ = self.center("port", T_port, holding=False)
        k.set_fixture("port", T_port)
        self.res["est"]["T_port_survey"] = T_port

        # ---------------- 3. plan everything before moving the housing
        tp = time.time()
        hook = pl.plan_hook(T_hook, y_neck, q_ref=q_hook_view)
        if not hook.ok:
            raise Abort("unreachable", f"hook: {hook.reason}")
        rot = pl.plan_rotate(hook.data["q_end"])
        if not rot.ok:
            raise Abort("unreachable", f"swing: {rot.reason}")
        pv = pl.plan_view(T_port, rot.data["q"], holding=True)
        if not pv.ok:
            raise Abort("unreachable", f"port view with housing: {pv.reason}")
        port = pl.plan_port(T_port, pv.data["q"])
        if not port.ok:
            raise Abort("unreachable", f"port: {port.reason}")
        k.set_peg_in_hand()
        k.grip = GRIP_CLOSED_ON_GROOVE
        to_pre, info = pl.connect(pv.data["q"], port.data["q_pre"])
        if to_pre is None:
            raise Abort("unreachable", f"port view -> pre-insert blocked {info}")
        k.set_peg_free(pl.peg_rest_pose(T_slot, y_neck))
        k.grip = GRIP_OPEN
        to_hook, info = pl.connect(r.q(), hook.data["q_pre"])
        if to_hook is None:
            raise Abort("unreachable", f"path back to the hook blocked {info}")
        self.res["timing"]["plan_s"] = time.time() - tp
        self.res["plan"].update(phi=hook.data["phi"], psi=port.data["psi"],
                                margin_hook=hook.margin,
                                margin_port=port.margin,
                                margin_rotate=rot.margin)
        self.log(f"plan ok: grasp roll {hook.data['phi']} deg (margin "
                 f"{hook.margin:.2f} rad), insertion roll {port.data['psi']} deg"
                 f" (margin {port.margin:.2f} rad)")

        # ---------------- 4. execute: pick
        r.move(to_hook, label="to hook")
        r.move(hook.data["approach"], v_peak=0.25, label="approach")
        r.grip(True, T=0.6, label="grasp")
        r.wait(0.3)
        if r.finger_opening() < -(p.finger_stroke - 0.003):
            raise Abort("missed grasp", "fingers closed on nothing")
        r.set_payload(True)
        self.res["holding"] = True
        k.set_peg_in_hand()
        k.grip = GRIP_CLOSED_ON_GROOVE
        for name, qs in hook.data["lift"]:
            r.move(qs, v_peak=0.25, label=f"lift out: {name}")

        # ---------------- rotate 180 deg and re-centre on the port tag
        r.move([rot.data["q"]], v_peak=1.0, label="rotate 180")
        r.move(pv.data["vias"], label="port view")
        T_port2, _ = self.center("port", T_port, holding=True)
        k.set_fixture("port", T_port2)
        self.res["est"]["T_port"] = T_port2
        port2 = pl.plan_port(T_port2, r.q(), psis=(port.data["psi"],)
                             + tuple(range(0, 360, 20)))
        if not port2.ok:
            raise Abort("unreachable", f"port after refinement: {port2.reason}")
        vias, info = pl.connect(r.q(), port2.data["q_pre"])
        if vias is None:
            raise Abort("unreachable", f"port view -> pre-insert blocked {info}")
        r.move(vias, label="pre-insert")
        self.res["plan"]["psi_final"] = port2.data["psi"]

        # ---------------- insert
        outcome = self.insert(port2.data)
        self.res["insert"] = outcome
        if outcome != "seated":
            self.back_off(port2.data)
            raise Abort("insertion failed", outcome)
        self.res["seated"] = True
        self.release_and_retreat()

    # ================================================================ insertion
    def insert(self, plan, speed=0.015):
        """Compliant push along the estimated bore axis. The compliance centre
        is at the plug tip: soft sideways, stiff along the axis, moderately
        soft in rotation, so the chamfer can slide the tip onto the axis and
        the bore walls can square the housing up."""
        r, p = self.r, self.p
        H = plan["H"]
        a = H.R[:, 0]
        tip_off = float(tip_in_tcp(p)[0])
        r.phase = "insert: entry"
        r.cart_begin(tip_off, **INSERT_GAINS)
        r.cart_set(sp_R=plan["T_pre"].R)
        Tc, _ = r.cart_state()
        sp = Tc.t.copy()
        dt = 0.001
        target = p.hole_depth + 0.008
        t = stall = 0.0
        while True:
            Tc, vel = r.cart_state()
            tip = Tc.t
            depth = float((tip - H.t) @ a)
            if depth > p.chamfer + 0.002:
                r.phase = "insert: bore"
                e = tip - sp
                sp += (dt / 0.25) * (e - (e @ a) * a)   # shed side load
            sp_depth = float((sp - H.t) @ a)
            v = speed if depth < p.hole_depth - 0.010 else 0.005
            if sp_depth < target:
                sp += a * min(v * dt, target - sp_depth)
                sp_vel = a * v
            else:
                sp_vel = np.zeros(3)
            lead = float((sp - tip) @ a)
            stalled = abs(float(vel @ a)) < 0.002 and lead > 0.003
            stall = stall + dt if stalled else 0.0
            if stall > 0.3 and depth > p.hole_depth - 0.008:
                r.phase = "seated"
                return "seated"
            if lead > 0.012 and depth < 0.004:
                r.phase = "JAMMED"
                return "jammed at the entry"
            if stall > 1.0:
                r.phase = "STUCK"
                return f"stuck at {depth * 1e3:.1f} mm"
            if t > 15.0:
                r.phase = "TIMEOUT"
                return "timeout"
            r.cart_set(sp_pos=sp, sp_vel=sp_vel)
            r.cart_step()
            t += dt

    def back_off(self, plan):
        r = self.r
        Tc, _ = r.cart_state()
        a = plan["H"].R[:, 0]
        r.cart_set(sp_pos=Tc.t, sp_vel=np.zeros(3))
        r.cart_step(300)
        for i in range(800):
            r.cart_set(sp_pos=Tc.t - a * 0.03 * min(1.0, i / 600))
            r.cart_step()
        r.cart_end()

    def release_and_retreat(self):
        r, k, pl = self.r, self.k, self.pl
        Tc, _ = r.cart_state()
        r.cart_set(sp_pos=Tc.t, sp_vel=np.zeros(3))   # drop the preload
        r.phase = "unload"
        r.cart_step(200)
        r.set_payload(False)
        r.grip(False, T=0.6, label="release")
        self.res["holding"] = False
        r.cart_end()
        k.grip = GRIP_OPEN
        q = r.q()
        T = k.fk(q)
        k.set_peg_free(T @ k.T_tcp_peg)       # housing now sits in the port
        T_up = Pose(T.R, T.t + T.R[:, 2] * 0.06)
        T_away = Pose(T.R, T_up.t + T.R[:, 0] * 0.06)
        qs = k.cartesian_path(q, interp_poses(T, T_up) + interp_poses(T_up, T_away))
        if qs is not None:
            r.move(qs, v_peak=0.4, label="retreat")
        vias, _ = pl.connect(r.q(), np.array(self.p.home_q))
        if vias is not None:
            r.move(vias, label="home")
        r.phase = "done"
