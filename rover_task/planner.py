"""Motion planning for the task program.

Everything here works on the program's own world model (kin.Kin), with the
hook and port placed where the camera measured them. A plan is only returned
if every configuration along it has an IK solution inside the joint limits,
the joint path between them is continuous, and nothing collides (with a
safety margin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geometry import (P, Params, Pose, exp_rot, hole_in_port, housing_offsets,
                      lift_out_path, rot_x, slot_in_hook, tip_in_tcp)
from kin import GRIP_CLOSED_ON_GROOVE, GRIP_OPEN, Kin, interp_joints, interp_poses

# TCP orientation relative to the slot frame for the grasp with roll phi = 0:
# tcp x into the plate (the plug points out of it, along tcp -x), tcp z = slot
# z, so the gripper comes straight down the J's stem direction.
TCP_IN_SLOT0 = np.column_stack([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
# TCP orientation relative to the hole frame (x into the bore) for roll 0:
# tcp x out of the bore (plug along tcp -x points in), tcp z = hole z.
TCP_IN_HOLE0 = TCP_IN_SLOT0
R_SLOT_PEG = np.column_stack([[0.0, 0, 1], [0, 1.0, 0], [-1.0, 0, 0]])


@dataclass
class Plan:
    ok: bool
    reason: str = ""
    margin: float = -1.0
    data: dict = field(default_factory=dict)


class Planner:
    def __init__(self, kin: Kin, p: Params = P, approach=0.07,
                 view_dist=0.20, view_obl=math.radians(30)):
        self.k = kin
        self.p = p
        self.approach = approach
        self.view_dist = view_dist
        self.view_obl = view_obl
        off = housing_offsets(p)
        self.groove_in_slot = np.array([-off["neck"], 0.0, 0.0])
        self.tip = tip_in_tcp(p)
        self.stats = dict(ik_calls=0)

    # ================================================================ helpers
    def _ik_from(self, T, seeds, target="tcp"):
        for s in seeds:
            if s is None:
                continue
            q, ok = self.k.ik(T, s, target=target)
            self.stats["ik_calls"] += 1
            if ok:
                return q
        return None

    def solutions(self, T, ref=None, target="tcp", n_keep=3, max_seeds=40):
        return self.k.ik_multi(T, target=target, ref=ref, n_keep=n_keep,
                               max_seeds=max_seeds)

    def check_joint_path(self, qs, check_peg_env=True):
        ok, info = self.k.path_ok(qs, check_peg_env=check_peg_env)
        return ok, info

    def connect(self, q0, q1, check_peg_env=True):
        """Collision-free joint-space path q0 -> q1: straight in joint space,
        else through a raised, tucked via posture."""
        direct = interp_joints(q0, q1)
        ok, info = self.check_joint_path(direct, check_peg_env)
        if ok:
            return [np.asarray(q1)], None
        lo, hi = self.k.lo, self.k.hi
        tucks = [(1.9, 2.5, 0.3), (2.2, 2.9, 0.6), (1.6, 2.0, 0.0)]
        for q2, q3, q4 in tucks:
            for base in (q0, q1):
                via0 = np.array(q0, float)
                via0[1:4] = (q2, q3, q4)
                via1 = np.array(q1, float)
                via1[1:4] = (q2, q3, q4)
                vias = [np.clip(via0, lo, hi), np.clip(via1, lo, hi)]
                seg = []
                prev = q0
                good = True
                for v in vias + [np.asarray(q1)]:
                    s = interp_joints(prev, v)
                    if not self.check_joint_path(s, check_peg_env)[0]:
                        good = False
                        break
                    seg.append(v)
                    prev = v
                if good:
                    return seg, None
        return None, info

    def cart(self, q0, T0, T1, check_peg_env=True, step=0.004):
        poses = interp_poses(T0, T1, step=step)
        qs = self.k.cartesian_path(q0, poses)
        if qs is None:
            return None, "ik"
        ok, info = self.check_joint_path(qs, check_peg_env)
        if not ok:
            return None, f"collision {info[1][:2]}"
        return qs, None

    # ================================================================ viewing
    def view_candidates(self, T_fix: Pose, dist=None, obl=None):
        """Camera poses that put the tag in the middle of the image, viewed
        `obl` off its normal (a square-on view gives poor rotation estimates),
        in order of preference: from the fixture's up side first. The camera
        roll about its optical axis is free."""
        dist = dist or self.view_dist
        obl = self.view_obl if obl is None else obl
        c, n, u = T_fix.t, T_fix.R[:, 0], T_fix.R[:, 2]
        w = np.cross(n, u)
        out = []
        for d_scale, o_scale in ((1.0, 1.0), (1.15, 1.0), (1.0, 0.6),
                                 (1.3, 1.2), (0.9, 1.4)):
            for az in (0, 30, -30, 60, -60, 90, -90, 120, -120):
                a = math.radians(az)
                ob = obl * o_scale
                dirv = math.cos(ob) * n + math.sin(ob) * (
                    math.cos(a) * u + math.sin(a) * w)
                pos = c + dist * d_scale * dirv
                zc = -dirv                                  # optical axis
                # image 'down' (OpenCV y) roughly along -fixture up
                yd = -u - (-u @ zc) * zc
                if np.linalg.norm(yd) < 1e-6:
                    yd = w - (w @ zc) * zc
                yd /= np.linalg.norm(yd)
                for roll in (0, 60, -60, 120, -120, 180):
                    Rr = exp_rot(zc * math.radians(roll))
                    y = Rr @ yd
                    x = np.cross(y, zc)
                    out.append(Pose(np.column_stack([x, y, zc]), pos))
        return out

    def plan_view(self, T_fix: Pose, q_from, holding: bool, max_cands=270,
                  dist=None, obl=None, check_path=True, n_good=6):
        """IK + collision-checked joint path from q_from to a camera pose
        centred on the fixture's tag."""
        k = self.k
        k.grip = GRIP_CLOSED_ON_GROOVE if holding else GRIP_OPEN
        if holding:
            k.set_peg_in_hand()
        cands = self.view_candidates(T_fix, dist, obl)[:max_cands]
        last = "no IK"
        found = []
        for i, Tc in enumerate(cands):
            # stop after a few good ones once past the preferred directions
            if found and (len(found) >= n_good or i >= 54):
                break
            q = self._ik_from(Tc, [q_from], target="cam")
            sols = [q] if q is not None else []
            sols += self.solutions(Tc, target="cam", ref=q_from, n_keep=2,
                                   max_seeds=12)
            for q in sols:
                if k.collisions(q):
                    last = f"collision at view {k.collisions(q)[:1]}"
                    continue
                if not check_path:
                    found.append(Plan(True, margin=k.margin(q),
                                      data=dict(q=q, T_cam=Tc, vias=[q])))
                    break
                vias, info = self.connect(q_from, q)
                if vias is None:
                    last = f"path to view blocked {info}"
                    continue
                found.append(Plan(True, margin=k.margin(q),
                                  data=dict(q=q, T_cam=Tc, vias=vias)))
                break
        if not found:
            return Plan(False, reason=last)
        return max(found, key=lambda pl: pl.margin)

    # ================================================================ hook
    def grasp_pose(self, T_slot: Pose, y_neck, phi):
        R = T_slot.R @ rot_x(phi) @ TCP_IN_SLOT0
        return Pose(R, T_slot @ (self.groove_in_slot + [0, y_neck, 0]))

    def peg_rest_pose(self, T_slot: Pose, y_neck):
        return Pose(T_slot.R @ R_SLOT_PEG, T_slot @ np.array([0, y_neck, 0.0]))

    def plan_hook(self, T_hook: Pose, y_neck, q_ref,
                  phis=tuple(range(0, 181, 15)) + tuple(range(-15, -181, -15))):
        """Grasp roll phi (degrees about the housing axis; 0 = from the J's
        open side) and joint paths for: approach (fingers open), grasp,
        lift-out along the J. Picks the roll with the largest joint-limit
        margin."""
        k = self.k
        T_slot = T_hook @ slot_in_hook(self.p)
        k.set_fixture("hook", T_hook)
        rest = self.peg_rest_pose(T_slot, y_neck)
        best, reasons = None, {}
        seed_prev = q_ref
        for phi_deg in phis:
            phi = math.radians(phi_deg)
            Tg = self.grasp_pose(T_slot, y_neck, phi)
            k.set_peg_free(rest)
            k.grip = GRIP_OPEN
            sols = []
            q = self._ik_from(Tg, [seed_prev, q_ref])
            if q is not None:
                sols.append(q)
            sols += [s for s in self.solutions(Tg, ref=q_ref, n_keep=3,
                                               max_seeds=30)
                     if not any(np.max(np.abs(s - o)) < 0.05 for o in sols)]
            if not sols:
                reasons[phi_deg] = "grasp IK"
                continue
            for qg in sols[:3]:
                r = self._hook_from_grasp(T_slot, Tg, qg, y_neck, rest)
                if not r.ok:
                    reasons[phi_deg] = r.reason
                    continue
                seed_prev = qg
                r.data["phi"] = phi_deg
                if best is None or r.margin > best.margin:
                    best = r
                break
        if best is None:
            k.set_peg_free(rest)
            return Plan(False, reason=f"no grasp roll works: {reasons}")
        k.set_peg_free(rest)
        return best

    def _hook_from_grasp(self, T_slot, Tg, qg, y_neck, rest):
        k = self.k
        # approach: from `approach` m back along +z_tcp, fingers open
        k.set_peg_free(rest)
        k.grip = GRIP_OPEN
        if k.collisions(qg):
            return Plan(False, reason=f"collision at grasp {k.collisions(qg)[:1]}")
        T_pre = Pose(Tg.R, Tg.t + Tg.R[:, 2] * self.approach)
        back, why = self.cart(qg, Tg, T_pre)
        if back is None:
            return Plan(False, reason=f"approach {why}")
        approach = back[::-1][1:] + [qg]
        q_pre = back[-1]
        # lift-out, holding the housing
        k.set_peg_in_hand()
        k.grip = GRIP_CLOSED_ON_GROOVE
        segs, q, T = [], qg, Tg
        margin = min(k.margin(x) for x in approach)
        for name, p_neck in lift_out_path(y_neck, self.p):
            T1 = Pose(Tg.R, T_slot @ (self.groove_in_slot + p_neck))
            in_slot = name != "clear"
            qs, why = self.cart(q, T, T1, check_peg_env=not in_slot)
            if qs is None:
                return Plan(False, reason=f"lift-out '{name}': {why}")
            segs.append((name, qs))
            margin = min(margin, min(k.margin(x) for x in qs))
            q, T = qs[-1], T1
        return Plan(True, margin=margin, data=dict(
            q_pre=q_pre, approach=approach, q_grasp=qg, T_grasp=Tg,
            lift=segs, q_end=q, T_end=T))

    # ================================================================ swing
    def plan_rotate(self, q_end):
        """'Rotate 180 deg': joint 1 turns by pi (whichever way stays inside
        its limits), every other joint held, housing in hand."""
        k = self.k
        k.set_peg_in_hand()
        k.grip = GRIP_CLOSED_ON_GROOVE
        for sgn in (+1, -1):
            q = np.array(q_end, float)
            q[0] += sgn * math.pi
            if not (k.lo[0] <= q[0] <= k.hi[0]):
                continue
            path = interp_joints(q_end, q, step=0.03)
            ok, info = self.check_joint_path(path)
            if ok:
                return Plan(True, margin=k.margin(q), data=dict(q=q, sign=sgn))
            return Plan(False, reason=f"swing collides {info[1][:2]}")
        return Plan(False, reason="joint 1 cannot turn 180 deg from here")

    # ================================================================ port
    def insert_pose(self, H: Pose, depth, psi):
        R = H.R @ rot_x(psi) @ TCP_IN_HOLE0
        return Pose(R, H @ np.array([depth, 0, 0]) - R @ self.tip)

    def plan_port(self, T_port: Pose, q_ref, pre_depth=-0.012,
                  psis=tuple(range(0, 360, 20)), sink=0.003):
        """Insertion roll psi (about the bore axis; free, the plug is round)
        and joint paths for: pre-insert, the straight push to 3 mm past the
        floor, release and retreat. Picks the largest joint-limit margin."""
        k = self.k
        k.set_fixture("port", T_port)
        H = T_port @ hole_in_port(self.p)
        best, reasons = None, {}
        seed_prev = q_ref
        for psi_deg in psis:
            psi = math.radians(psi_deg)
            k.set_peg_in_hand()
            k.grip = GRIP_CLOSED_ON_GROOVE
            T_pre = self.insert_pose(H, pre_depth, psi)
            sols = []
            q = self._ik_from(T_pre, [seed_prev, q_ref])
            if q is not None:
                sols.append(q)
            sols += [s for s in self.solutions(T_pre, ref=q_ref, n_keep=2,
                                               max_seeds=20)
                     if not any(np.max(np.abs(s - o)) < 0.05 for o in sols)]
            if not sols:
                reasons[psi_deg] = "pre-insert IK"
                continue
            for qp in sols[:2]:
                r = self._port_from_pre(H, T_pre, qp, psi, sink)
                if not r.ok:
                    reasons[psi_deg] = r.reason
                    continue
                seed_prev = qp
                r.data["psi"] = psi_deg
                if best is None or r.margin > best.margin:
                    best = r
                break
        if best is None:
            return Plan(False, reason=f"no insertion roll works: {reasons}")
        return best

    def _port_from_pre(self, H, T_pre, qp, psi, sink):
        k = self.k
        k.set_peg_in_hand()
        k.grip = GRIP_CLOSED_ON_GROOVE
        if k.collisions(qp):
            return Plan(False, reason=f"collision at pre-insert {k.collisions(qp)[:1]}")
        # the push: tip from pre-insert to sink past the floor. Before the tip
        # reaches the opening check the housing too; once inside, only the arm.
        T_open = self.insert_pose(H, 0.0, psi)
        q1, why = self.cart(qp, T_pre, T_open, check_peg_env=True)
        if q1 is None:
            return Plan(False, reason=f"entry {why}")
        T_seat = self.insert_pose(H, self.p.hole_depth + sink, psi)
        q2, why = self.cart(q1[-1], T_open, T_seat, check_peg_env=False)
        if q2 is None:
            return Plan(False, reason=f"push {why}")
        # release: housing left in the bore, fingers open, back off up the
        # gripper, then away from the port along the bore axis
        T_rest = self.insert_pose(H, self.p.hole_depth, psi)
        peg_T = T_rest @ k.T_tcp_peg
        k.set_peg_free(peg_T)
        k.grip = GRIP_OPEN
        q_seat = q2[-1]
        T_up = Pose(T_seat.R, T_seat.t + T_seat.R[:, 2] * 0.06)
        q3, why = self.cart(q_seat, T_seat, T_up)
        if q3 is None:
            return Plan(False, reason=f"retreat {why}")
        T_away = Pose(T_up.R, T_up.t + T_up.R[:, 0] * 0.06)
        q4, why = self.cart(q3[-1], T_up, T_away)
        if q4 is None:
            return Plan(False, reason=f"retreat away {why}")
        allq = [qp] + q1 + q2 + q3 + q4
        margin = min(k.margin(q) for q in allq)
        k.set_peg_in_hand()
        return Plan(True, margin=margin, data=dict(
            q_pre=qp, T_pre=T_pre, push=q1 + q2, q_seat=q_seat, T_seat=T_seat,
            retreat=q3 + q4, H=H))
