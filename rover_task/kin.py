"""Kinematics for the task program: forward kinematics, damped-least-squares
IK with joint limits, and collision checking against the program's world model.

The program owns a private copy of the MuJoCo model. It moves the fixture
bodies to where perception says they are and attaches the housing to the
gripper when it is holding it, then uses MuJoCo's collision detection
(with a safety margin) to check candidate arm configurations.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from geometry import Pose, Params, P, rotvec, tcp_to_cam, tcp_to_peg

ARM = [f"joint{i}" for i in range(1, 7)]
GRIP_OPEN = -0.012           # finger joint: inner faces 29 mm from centre
GRIP_CLOSED_ON_GROOVE = -(0.041 - 0.008)


class Kin:
    def __init__(self, xml_path, p: Params = P, margin=0.004, peg_margin=0.002):
        self.p = p
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        m = self.m
        self.d = mujoco.MjData(m)
        jids = [m.joint(j).id for j in ARM]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.lo = m.jnt_range[jids, 0].copy()
        self.hi = m.jnt_range[jids, 1].copy()
        self.tcp = m.site("tcp").id
        self.gl = m.jnt_qposadr[m.joint("gripper_L_joint").id]
        self.gr = m.jnt_qposadr[m.joint("gripper_R_joint").id]
        self.peg_q = m.jnt_qposadr[m.joint("peg_free").id]
        self.peg_body = m.body("peg").id
        self.hook_body = m.body("hook").id
        self.port_body = m.body("port").id
        self.T_tcp_cam = tcp_to_cam(p)
        self.T_tcp_peg = tcp_to_peg(p)
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        # body classes for collision filtering
        robot_bodies = set()
        for name in ("base_link", "link1", "link2", "link3", "link4", "link5",
                     "link6", "gripper_center_link", "gripper_L_link",
                     "gripper_R_link"):
            robot_bodies.add(m.body(name).id)
        self.robot_bodies = robot_bodies
        self.env_bodies = {0, self.hook_body, self.port_body,
                           m.body("bore").id}
        self.pad_geoms = {m.geom("pad_L").id, m.geom("pad_R").id}
        # safety margins: robot/finger/camera geoms and housing geoms
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            if b in robot_bodies and g not in self.pad_geoms:
                m.geom_margin[g] = margin
            elif b == self.peg_body:
                m.geom_margin[g] = peg_margin
        # self-collision between non-adjacent robot links, and the held
        # housing against the arm
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in robot_bodies and m.geom_contype[g] == 2:
                m.geom_conaffinity[g] |= 2
        self.self_pairs_ignore = self._always_touching_pairs()
        self.peg_in_hand = False
        self.peg_free_pose = None   # (pos, quat) when the housing is not held
        self.grip = GRIP_OPEN

    # ------------------------------------------------------------ world model
    def set_fixture(self, name, T: Pose):
        b = self.hook_body if name == "hook" else self.port_body
        self.m.body_pos[b] = T.t
        self.m.body_quat[b] = T.quat()

    def set_peg_free(self, T: Pose | None):
        """Housing resting somewhere in the world (None: park it far away)."""
        self.peg_in_hand = False
        if T is None:
            T = Pose(np.eye(3), np.array([5.0, 5.0, -5.0]))
        self.peg_free_pose = T

    def set_peg_in_hand(self):
        self.peg_in_hand = True

    def _always_touching_pairs(self):
        """Robot link pairs whose collision hulls overlap at home and in a few
        neutral poses (adjacent-ish links on this arm); ignored as self-hits."""
        pairs = None
        for q in ([0, 1.2, 1.2, 0, 0, 0], [0.5, 1.6, 2.0, -0.5, 0.3, 1.0],
                  [-1.0, 0.8, 0.6, 0.6, -0.4, -1.5], list(self.p.home_q)):
            self.d.qpos[:] = self.m.qpos0
            self.d.qpos[self.qadr] = q
            self.d.qpos[self.peg_q:self.peg_q + 3] = [5, 5, -5]
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_collision(self.m, self.d)
            s = set()
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
                if b1 in self.robot_bodies and b2 in self.robot_bodies:
                    s.add(frozenset((b1, b2)))
            pairs = s if pairs is None else pairs | s
        return pairs

    # ------------------------------------------------------------ FK
    def _set_q(self, q):
        d = self.d
        d.qpos[self.qadr] = q
        d.qpos[self.gl] = self.grip
        d.qpos[self.gr] = -self.grip

    def fk(self, q) -> Pose:
        self._set_q(q)
        mujoco.mj_kinematics(self.m, self.d)
        return Pose(self.d.site_xmat[self.tcp].reshape(3, 3).copy(),
                    self.d.site_xpos[self.tcp].copy())

    def fk_cam(self, q) -> Pose:
        return self.fk(q) @ self.T_tcp_cam

    # ------------------------------------------------------------ IK
    def ik(self, T: Pose, seed, iters=150, tol_p=2e-5, tol_r=2e-4,
           target="tcp"):
        """DLS IK from one seed. `target` 'cam' means T is a camera pose.
        Returns (q, ok)."""
        if target == "cam":
            T = T @ self.T_tcp_cam.inv()
        m, d = self.m, self.d
        q = np.clip(np.asarray(seed, float), self.lo, self.hi)
        lam = 1e-2
        e_prev = None
        best_e, stall = np.inf, 0
        for _ in range(iters):
            self._set_q(q)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.tcp]
            R = d.site_xmat[self.tcp].reshape(3, 3)
            ep = T.t - p
            er = rotvec(T.R @ R.T)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                return q, True
            e = np.r_[ep, 0.2 * er]               # 0.2 m per rad weighting
            ne = float(np.linalg.norm(e))
            if ne < best_e * 0.995:
                best_e, stall = ne, 0
            else:
                stall += 1
                if stall > 15:
                    return q, False           # stuck (joint limit / singular)
            if e_prev is not None:
                lam = max(1e-4, lam * 0.5) if ne < e_prev else min(1.0, lam * 4)
            e_prev = ne
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.tcp)
            J = np.vstack([self._jp[:, self.dadr], 0.2 * self._jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False

    def seeds(self, T: Pose, n_random=24, rng=None):
        yaw = math.atan2(T.t[1], T.t[0])
        out = []
        for q1 in (yaw, yaw - 0.4, yaw + 0.4):
            for q2, q3 in ((1.0, 1.0), (1.6, 1.6), (1.2, 2.2), (2.0, 2.6),
                           (0.6, 0.6)):
                for q4 in (-0.8, 0.0, 0.8):
                    for q6 in (-1.6, 0.0, 1.6):
                        out.append(np.array([q1, q2, q3, q4, 0.0, q6]))
        rng = rng or np.random.default_rng(0)
        out += [rng.uniform(self.lo, self.hi) for _ in range(n_random)]
        # interleave so that any prefix of the list is diverse
        order = np.random.default_rng(1).permutation(len(out))
        return [out[i] for i in order]

    def ik_multi(self, T: Pose, target="tcp", ref=None, n_keep=6, rng=None,
                 max_seeds=None):
        """Solve from many seeds; return distinct solutions sorted by
        closeness to `ref` (if given) else by joint-limit margin."""
        Tt = T @ self.T_tcp_cam.inv() if target == "cam" else T
        sols = []
        seeds = ([ref] if ref is not None else []) + self.seeds(Tt, rng=rng)
        if max_seeds:
            seeds = seeds[:max_seeds]
        for s in seeds:
            q, ok = self.ik(Tt, s)
            if not ok:
                continue
            if any(np.max(np.abs(q - o)) < 1e-2 for o in sols):
                continue
            sols.append(q)
            if len(sols) >= 3 * n_keep:
                break
        if ref is not None:
            sols.sort(key=lambda q: float(np.max(np.abs(q - ref))))
        else:
            sols.sort(key=lambda q: -self.margin(q))
        return sols[:n_keep]

    def margin(self, q):
        q = np.asarray(q)
        return float(np.min(np.r_[q - self.lo, self.hi - q]))

    # ------------------------------------------------------------ collisions
    def collisions(self, q, check_peg_env=True, check_self=True):
        """List of (body1, body2) name pairs in contact (within margin) for arm
        configuration q, with the current gripper opening and housing state."""
        m, d = self.m, self.d
        self._set_q(q)
        if self.peg_in_hand:
            mujoco.mj_kinematics(m, d)
            T = Pose(d.site_xmat[self.tcp].reshape(3, 3).copy(),
                     d.site_xpos[self.tcp].copy()) @ self.T_tcp_peg
        else:
            T = self.peg_free_pose or Pose(np.eye(3), np.array([5, 5, -5.0]))
        d.qpos[self.peg_q:self.peg_q + 3] = T.t
        d.qpos[self.peg_q + 3:self.peg_q + 7] = T.quat()
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hits = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if g1 in self.pad_geoms or g2 in self.pad_geoms:
                if self.peg_body in (b1, b2):
                    continue           # the grasp itself
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            pe1, pe2 = b1 == self.peg_body, b2 == self.peg_body
            if r1 and r2:
                if not check_self or frozenset((b1, b2)) in self.self_pairs_ignore:
                    continue
            elif (pe1 or pe2) and not (r1 or r2):
                if not check_peg_env or not self.peg_in_hand:
                    continue           # resting housing on its fixture
            elif (pe1 or pe2) and (r1 or r2) and not self.peg_in_hand:
                pass                   # robot touching the resting housing
            elif (pe1 or pe2) and (r1 or r2) and self.peg_in_hand:
                # held housing vs. gripper links that it sits next to
                other = b2 if pe1 else b1
                if other in (m.body("gripper_L_link").id,
                             m.body("gripper_R_link").id):
                    continue
            hits.append((m.body(b1).name, m.body(b2).name))
        return hits

    def path_ok(self, qs, check_peg_env=True, skip_first=0):
        for i, q in enumerate(qs):
            if i < skip_first:
                continue
            h = self.collisions(q, check_peg_env=check_peg_env)
            if h:
                return False, (i, h)
        return True, None

    # ------------------------------------------------------------ paths
    def cartesian_path(self, q0, poses, max_jump=0.15):
        """IK along a dense sequence of TCP poses by continuation from q0.
        Returns joint path or None if it fails or jumps."""
        out = []
        q = np.asarray(q0, float)
        for T in poses:
            qn, ok = self.ik(T, q, iters=60)
            if not ok or np.max(np.abs(qn - q)) > max_jump:
                return None
            out.append(qn)
            q = qn
        return out


def interp_poses(T0: Pose, T1: Pose, step=0.004, rot_step=math.radians(2)):
    """Straight-line poses from T0 to T1 (excluding T0)."""
    dp = T1.t - T0.t
    dr = rotvec(T1.R @ T0.R.T)
    n = max(1, int(math.ceil(max(np.linalg.norm(dp) / step,
                                 np.linalg.norm(dr) / rot_step))))
    from geometry import exp_rot
    return [Pose(exp_rot(dr * k / n) @ T0.R, T0.t + dp * k / n)
            for k in range(1, n + 1)]


def interp_joints(q0, q1, step=0.03):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(1, int(math.ceil(np.max(np.abs(q1 - q0)) / step)))
    return [q0 + (q1 - q0) * k / n for k in range(1, n + 1)]
