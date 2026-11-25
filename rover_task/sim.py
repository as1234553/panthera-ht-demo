"""MuJoCo simulation backend for the task program.

SimRobot plays the part of the real Panthera-HT + wrist camera. The program
talks to it only through the Robot interface (joint readings, camera images,
joint trajectories, a Cartesian impedance mode for the insertion, gripper,
payload compensation). Everything else here (true fixture poses, contact
forces, the housing's true pose) is ground truth used to score a trial.

Realism knobs (all randomised per trial by the Monte Carlo):
  * image noise and brightness/gain,
  * hand-eye calibration error (true camera mount vs. the program's value),
  * focal-length calibration error,
  * joint encoder offsets (the program's joint readings are slightly off).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from geometry import (P, Params, Pose, exp_rot, hole_in_port, quat_to_mat,
                      rotvec, slot_in_hook, tcp_to_peg)

GRIP_OPEN = -0.012
GRIP_CLOSE = -0.041          # command fully closed; the actuator is force-limited

ARM = [f"joint{i}" for i in range(1, 7)]


def min_jerk(tau):
    tau = min(max(tau, 0.0), 1.0)
    return (tau ** 3 * (10 - 15 * tau + 6 * tau ** 2),
            30 * tau ** 2 * (1 - tau) ** 2,
            60 * tau * (1 - tau) * (1 - 2 * tau))


class SimRobot:
    def __init__(self, xml, T_hook: Pose, T_port: Pose, p: Params = P,
                 errors=None, rng=None, video=None):
        self.p = p
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.rng = rng or np.random.default_rng(0)
        e = dict(img_noise=0.0, img_gain=1.0, cam_pos=np.zeros(3),
                 cam_rot=np.zeros(3), fovy_scale=1.0, enc=np.zeros(6))
        e.update(errors or {})
        self.err = e
        jids = [m.joint(j).id for j in ARM]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.lo, self.hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        self.tau_lim = m.actuator_ctrlrange[:6, 1].copy()
        self.damping = m.dof_damping[self.dadr].copy()
        self.tcp = m.site("tcp").id
        self.gc_body = m.body("gripper_center_link").id
        self.peg_body = m.body("peg").id
        self.peg_jq = m.jnt_qposadr[m.joint("peg_free").id]
        self.peg_jv = m.jnt_dofadr[m.joint("peg_free").id]
        self.hook_body = m.body("hook").id
        self.port_body = m.body("port").id
        self.bore_body = m.body("bore").id
        self.peg_mass = float(m.body_subtreemass[self.peg_body])
        self.T_tcp_peg = tcp_to_peg(p)
        self.robot_bodies = {m.body(n).id for n in (
            "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
            "gripper_center_link", "gripper_L_link", "gripper_R_link")}
        self.pad_geoms = {m.geom("pad_L").id, m.geom("pad_R").id}

        # true fixture poses
        self.T_hook, self.T_port = T_hook, T_port
        m.body_pos[self.hook_body] = T_hook.t
        m.body_quat[self.hook_body] = T_hook.quat()
        m.body_pos[self.port_body] = T_port.t
        m.body_quat[self.port_body] = T_port.quat()
        self.H_true = T_port @ hole_in_port(p)

        # true camera mount = nominal + calibration error
        cid = m.camera("wrist").id
        self.cam_id = cid
        Rc = quat_to_mat(m.cam_quat[cid])
        m.cam_pos[cid] = m.cam_pos[cid] + e["cam_pos"]
        m.cam_quat[cid] = Pose(exp_rot(e["cam_rot"]) @ Rc).quat()
        m.cam_fovy[cid] = math.degrees(2 * math.atan(
            math.tan(math.radians(p.cam_fovy) / 2) / e["fovy_scale"]))

        # start: arm at home, fingers open, housing resting in the hook
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[m.jnt_qposadr[m.joint("gripper_L_joint").id]] = GRIP_OPEN
        d.qpos[m.jnt_qposadr[m.joint("gripper_R_joint").id]] = -GRIP_OPEN
        d.ctrl[6] = GRIP_OPEN
        T_slot = T_hook @ slot_in_hook(p)
        R_slot_peg = np.column_stack([[0.0, 0, 1], [0, 1.0, 0], [-1.0, 0, 0]])
        d.qpos[self.peg_jq:self.peg_jq + 3] = T_slot.t
        d.qpos[self.peg_jq + 3:self.peg_jq + 7] = Pose(T_slot.R @ R_slot_peg).quat()
        self.T_slot = T_slot
        mujoco.mj_forward(m, d)
        self.settle_info = self._settle_housing()

        # control state
        self.mode = "joint"
        self.q_des = self._q_meas()
        self.qd_des = np.zeros(6)
        self.qdd_des = np.zeros(6)
        self.payload = False
        self.cart = None
        self.phase = "start"
        self.log = []
        self.events = []
        self.max_robot_env_force = 0.0
        self.robot_env_contacts = set()
        self.max_peg_hook_force = {}
        self.max_peg_port_force = 0.0
        self.video = video
        self.renderer = None
        self._Mfull = np.zeros((m.nv, m.nv))
        self._dk = mujoco.MjData(m)
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self.n_captures = 0

    # ================================================================ setup
    def _settle_housing(self, T=1.5):
        """Let the housing settle in the (possibly tilted) hook under gravity
        with the arm frozen at home."""
        m, d = self.m, self.d
        q0 = d.qpos.copy()
        jq, jv = self.peg_jq, self.peg_jv
        fq = np.ones(m.nq, bool)
        fq[jq:jq + 7] = False
        fv = np.ones(m.nv, bool)
        fv[jv:jv + 6] = False
        for _ in range(int(T / m.opt.timestep)):
            mujoco.mj_step(m, d)
            d.qpos[fq], d.qvel[fv] = q0[fq], 0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        # where did the neck end up, in the slot frame?
        rel = self.T_slot.inv() @ d.xpos[self.peg_body]
        return dict(neck_slot=rel, speed=float(np.linalg.norm(d.qvel[jv:jv + 3])))

    # ================================================================ sensors
    def _q_meas(self):
        return self.d.qpos[self.qadr] + self.err["enc"]

    def q(self):
        return self._q_meas().copy()

    def capture(self):
        """Wrist camera image (RGB uint8) with sensor noise."""
        p = self.p
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, p.cam_height, p.cam_width)
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, camera="wrist")
        img = self.renderer.render().astype(np.float32)
        img = img * self.err["img_gain"]
        if self.err["img_noise"] > 0:
            img += self.rng.normal(0, self.err["img_noise"], img.shape)
        self.n_captures += 1
        return np.clip(img, 0, 255).astype(np.uint8)

    # ================================================================ control
    def _tcp_meas(self):
        """TCP pose from the true kinematics at the true joint angles. (The
        program computes its own FK from the joint readings.)"""
        d = self.d
        return (d.site_xpos[self.tcp].copy(),
                d.site_xmat[self.tcp].reshape(3, 3).copy())

    def _jac(self, point):
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr, point, self.gc_body)
        return np.vstack([self._jp, self._jr])[:, self.dadr]

    def _payload_tau(self):
        if not self.payload:
            return 0.0
        p, R = self._tcp_meas()
        pc = p + R @ self.T_tcp_peg.t
        return self._jac(pc)[:3].T @ np.array([0, 0, self.peg_mass * 9.81])

    def _torque(self):
        m, d = self.m, self.d
        qd = d.qvel[self.dadr]
        bias = d.qfrc_bias[self.dadr] + 0.9 * self.damping * qd + self._payload_tau()
        if self.mode == "joint":
            mujoco.mj_fullM(m, d, self._Mfull)
            M = self._Mfull[np.ix_(self.dadr, self.dadr)]
            w = 25.0
            e = self.q_des - self._q_meas()
            ed = self.qd_des - qd
            tau = M @ (w * w * e + 2 * w * ed + self.qdd_des) + bias
        else:
            tau = self._cart_torque(qd) + bias
        return np.clip(tau, -self.tau_lim, self.tau_lim)

    def _step(self):
        m, d = self.m, self.d
        mujoco.mj_step1(m, d)
        d.ctrl[:6] = self._torque()
        mujoco.mj_step2(m, d)
        k = int(round(d.time / m.opt.timestep))
        if k % 10 == 0:
            self._record()
        if self.video is not None and k % self.video.every == 0:
            self.video(self)

    # ================================================================ motions
    def move(self, vias, v_peak=1.0, t_min=0.6, settle=True, label=None,
             duration=None):
        """Follow joint waypoints (a dense IK path or a few via postures)
        with a minimum-jerk profile along the path, then settle."""
        if label:
            self.phase = label
            if label == "rotate 180" and not hasattr(self, "in_hand"):
                self.in_hand = self.peg_in_hand_offset()
        self.mode = "joint"
        q0 = self.q_des.copy()
        pts = [q0] + [np.asarray(v, float) for v in vias]
        seg = np.array([np.max(np.abs(b - a)) for a, b in zip(pts[:-1], pts[1:])])
        L = float(seg.sum())
        if L < 1e-9:
            return self.wait(0.05) if settle else None
        cum = np.r_[0, np.cumsum(seg)]
        T = duration or max(t_min, 1.875 * L / v_peak)
        n = int(round(T / self.m.opt.timestep))
        P_ = np.array(pts)
        for i in range(1, n + 1):
            s, ds, dds = min_jerk(i / n)
            sl = s * L
            j = min(int(np.searchsorted(cum, sl, side="right")) - 1, len(seg) - 1)
            f = 0.0 if seg[j] < 1e-12 else (sl - cum[j]) / seg[j]
            dq_ds = (P_[j + 1] - P_[j]) / max(seg[j], 1e-12)
            self.q_des = P_[j] + f * (P_[j + 1] - P_[j])
            self.qd_des = dq_ds * ds * L / T
            self.qdd_des = dq_ds * dds * L / T ** 2
            self._step()
        self.q_des = P_[-1].copy()
        self.qd_des[:] = 0
        self.qdd_des[:] = 0
        if settle:
            self.settle()

    def settle(self, tol=2e-4, vtol=5e-3, hold=0.1, t_max=1.5):
        dt, t, ok = self.m.opt.timestep, 0.0, 0.0
        while t < t_max:
            e = np.max(np.abs(self.q_des - self._q_meas()))
            v = np.max(np.abs(self.d.qvel[self.dadr]))
            ok = ok + dt if (e < tol and v < vtol) else 0.0
            if ok >= hold:
                return True
            t += dt
            self._step()
        return False

    def wait(self, T, label=None):
        if label:
            self.phase = label
        for _ in range(int(round(T / self.m.opt.timestep))):
            self._step()

    def grip(self, close: bool, T=0.6, label=None):
        if label:
            self.phase = label
        c0 = self.d.ctrl[6]
        c1 = GRIP_CLOSE if close else GRIP_OPEN
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            self.d.ctrl[6] = c0 + (c1 - c0) * min(1.0, 2 * k / n)
            self._step()

    def set_payload(self, on: bool):
        self.payload = on

    def finger_opening(self):
        """Gripper joint reading (the SDK reports the finger position)."""
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint("gripper_L_joint").id]])

    # ---------------- Cartesian impedance (insertion) ----------------
    # The arm's own impedance mode: works from the encoder readings (like the
    # program), with the compliance centre `offset` m along tcp x and
    # stiffness/damping given in the setpoint's tool frame.
    def _fk_meas(self, q, point_offset):
        dk, m = self._dk, self.m
        dk.qpos[:] = self.d.qpos
        dk.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, dk)
        mujoco.mj_comPos(m, dk)
        R = dk.site_xmat[self.tcp].reshape(3, 3).copy()
        pc = dk.site_xpos[self.tcp] + R[:, 0] * point_offset
        mujoco.mj_jac(m, dk, self._jp, self._jr, pc, self.gc_body)
        J = np.vstack([self._jp, self._jr])[:, self.dadr]
        return pc, R, J

    def cart_begin(self, offset, kp, kr, dp, dr):
        """Switch to Cartesian impedance; the setpoint starts at the current
        (measured) compliance-centre pose. Returns that pose."""
        self.cart = dict(offset=offset, kp=np.asarray(kp, float), kr=kr,
                         dp=np.asarray(dp, float), dr=dr)
        pc, R, _ = self._fk_meas(self._q_meas(), offset)
        self.cart.update(sp_pos=pc.copy(), sp_R=R.copy(), sp_vel=np.zeros(3))
        self.mode = "cart"
        return Pose(R, pc)

    def cart_set(self, sp_pos=None, sp_R=None, sp_vel=None, gains=None):
        c = self.cart
        if sp_pos is not None:
            c["sp_pos"] = np.asarray(sp_pos, float)
        if sp_R is not None:
            c["sp_R"] = np.asarray(sp_R, float)
        if sp_vel is not None:
            c["sp_vel"] = np.asarray(sp_vel, float)
        if gains:
            c.update(gains)

    def cart_state(self):
        """Measured compliance-centre pose and linear velocity."""
        pc, R, J = self._fk_meas(self._q_meas(), self.cart["offset"])
        v = J @ self.d.qvel[self.dadr]
        return Pose(R, pc), v[:3]

    def _cart_torque(self, qd):
        c = self.cart
        pc, R, J = self._fk_meas(self._q_meas(), c["offset"])
        v = J @ qd
        Rt = c["sp_R"]
        K = Rt @ np.diag(c["kp"]) @ Rt.T
        D = Rt @ np.diag(c["dp"]) @ Rt.T
        F = K @ (c["sp_pos"] - pc) + D @ (c["sp_vel"] - v[:3])
        M = c["kr"] * rotvec(Rt @ R.T) - c["dr"] * v[3:]
        return J.T @ np.r_[F, M]

    def cart_step(self, n=1):
        for _ in range(n):
            self._step()

    def cart_end(self):
        self.mode = "joint"
        self.q_des = self._q_meas().copy()
        self.qd_des[:] = 0
        self.qdd_des[:] = 0
        self.cart = None

    # ================================================================ truth
    def contact_force(self, body_a, bodies_b):
        m, d = self.m, self.d
        total = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == body_a and b2 in bodies_b:
                sgn = -1
            elif b2 == body_a and b1 in bodies_b:
                sgn = 1
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            total += sgn * (c.frame.reshape(3, 3).T @ f6[:3])
        return total

    def peg_state(self):
        """True tip depth along the bore axis, lateral offset and tilt."""
        d, m = self.d, self.m
        tip = d.site_xpos[m.site("peg_tip").id].copy()
        rear = d.site_xpos[m.site("peg_rear").id].copy()
        H = self.H_true
        a = H.R[:, 0]
        rel = tip - H.t
        depth = float(rel @ a)
        lat = float(np.linalg.norm(rel - depth * a))
        u = (tip - rear) / np.linalg.norm(tip - rear)
        tilt = math.degrees(math.acos(min(1.0, float(u @ a))))
        return tip, depth, lat, tilt

    def _record(self):
        m, d = self.m, self.d
        # robot touching the environment (fixtures, deck): any contact force
        f6 = np.zeros(6)
        env = {0, self.hook_body, self.port_body, self.bore_body}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if (r1 and b2 in env) or (r2 and b1 in env):
                mujoco.mj_contactForce(m, d, i, f6)
                f = float(np.linalg.norm(f6[:3]))
                if f > 0.5:
                    self.max_robot_env_force = max(self.max_robot_env_force, f)
                    self.robot_env_contacts.add(
                        (m.body(b1).name, m.body(b2).name, self.phase))
        fh = float(np.linalg.norm(self.contact_force(self.peg_body, {self.hook_body})))
        self.max_peg_hook_force[self.phase] = max(
            self.max_peg_hook_force.get(self.phase, 0.0), fh)
        fp = self.contact_force(self.peg_body, {self.port_body, self.bore_body})
        self.max_peg_port_force = max(self.max_peg_port_force,
                                      float(np.linalg.norm(fp)))
        tip, depth, lat, tilt = self.peg_state()
        tcp = d.site_xpos[self.tcp]
        a = self.H_true.R[:, 0]
        self.log.append(dict(
            t=d.time, phase=self.phase, depth=depth, lat=lat, tilt=tilt,
            f_hook=fh, f_port_ax=float(fp @ a),
            f_port_lat=float(np.linalg.norm(fp - (fp @ a) * a)),
            tcp_x=tcp[0], tcp_y=tcp[1], tcp_z=tcp[2],
            grip=float(d.ctrl[6])))

    def peg_in_hand_offset(self):
        """True housing pose relative to the nominal in-hand pose (TCP
        frame): translation [m] and rotation vector [rad]."""
        p, R = self._tcp_meas()
        T_tcp = Pose(R, p)
        T_peg = Pose(self.d.xmat[self.peg_body].reshape(3, 3).copy(),
                     self.d.xpos[self.peg_body].copy())
        # translation of the housing origin, and the tilt of its axis from the
        # nominal (tcp x); its roll about its own axis does not matter
        dt = T_tcp.R.T @ (T_peg.t - (T_tcp @ self.T_tcp_peg).t)
        axis = T_tcp.R.T @ T_peg.R[:, 2]
        return dt, np.cross([1.0, 0, 0], axis) * 1.0

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
