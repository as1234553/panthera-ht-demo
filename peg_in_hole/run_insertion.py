"""Panthera-HT peg-in-hole: pick a cable housing (the grip adapter on the end of
a long cable) out of a J-hook, turn, and plug it into a horizontally mounted
hole with a chamfered opening (MuJoCo).

The housing starts resting in a J-shaped slot in a vertical plate,
perpendicular to the plate: its neck sits on the floor of the slot with a
flange either side of the plate. The cable comes out of the back, so the arm
grabs the housing from the side: the gripper comes down from above onto the
grip groove and closes across it. It then lifts the housing out along the J
(up off the floor of the slot, sideways to the stem, up the stem until the
flanges clear the top), swings round to the hole and plugs it in.
With --start hand the arm instead starts already holding the housing.

Control: torque-level Cartesian impedance control on the 6 arm joints
(the SDK model's actuators are joint torque motors), with gravity/Coriolis
compensation from MuJoCo's bias forces and feed-forward for the held peg's
weight. Stiffness is specified in the tool frame (axial / lateral).

Insertion strategy: the arm is deliberately aimed a few millimetres off the
hole axis (a stand-in for perception/calibration error). It then switches to a
compliant mode whose compliance centre sits at the peg tip (like a
remote-centre-compliance device): low lateral stiffness, high rotational
stiffness. Pushing forward, the peg lands on the 45 deg chamfer, which turns
the axial push into a lateral force that slides the peg onto the hole axis; it
then slides into the bore and seats against the bottom.

Usage:
    python peg_in_hole/run_insertion.py                 # sim + video + plot
    python peg_in_hole/run_insertion.py --offset 0.003 0.002
    python peg_in_hole/run_insertion.py --start hand    # skip the J-hook
    python peg_in_hole/run_insertion.py --no-video
    mjpython peg_in_hole/run_insertion.py --viewer      # live viewer (macOS)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kinematics import axis_angle_quat, quat_err, site_pose, slerp  # noqa: E402

SCENE = os.path.join(HERE, "scene.xml")
OUT = os.path.join(HERE, "out")

GRIP_OPEN = -0.012           # finger joint: inner faces 29 mm from centre
GRIP_CLOSE = -0.041          # command fully closed; force-limited actuator

# Stiffness/damping in the TOOL frame: [axial (tool x), lateral y, lateral z]
GAINS = {
    "stiff":     dict(kp=[1500, 1500, 1500], kr=40.0, dp=[70, 70, 70], dr=1.0),
    "compliant": dict(kp=[900, 250, 250], kr=40.0, dp=[45, 25, 25], dr=1.0),
}


def min_jerk(tau):
    tau = min(max(tau, 0.0), 1.0)
    s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
    ds = 30 * tau ** 2 * (1 - tau) ** 2
    return s, ds


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class ImpedanceController:
    """tau = J_c^T [K e - D (v - v_d)] + bias + J_m^T (m g)  at a compliance
    centre c fixed to the gripper, `offset` metres along the tool axis."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, com_offset):
        self.m, self.d = m, d
        self.dofs = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id]
                              for i in range(1, 7)])
        self.tcp = m.site("tcp").id
        self.body = m.body("gripper_center_link").id
        self.lim = m.actuator_ctrlrange[:6, 1].copy()
        # The SDK model has 1 Nms/rad viscous damping in every joint; cancel
        # most of it (as a real controller compensates joint friction) so it
        # doesn't drag the tool off its setpoint during large joint motions.
        self.damp_comp = 0.9 * m.dof_damping[self.dofs]
        self.offset = 0.0
        self.payload = 0.0
        self.com_offset = com_offset
        self.set_gains("stiff")
        self.sp_pos, self.sp_quat = self.center_pose()
        self.sp_vel = np.zeros(3)
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def set_gains(self, name):
        g = GAINS[name]
        self.mode = name
        self.kp, self.dp = np.array(g["kp"], float), np.array(g["dp"], float)
        self.kr, self.dr = g["kr"], g["dr"]

    def center_pose(self, offset=None):
        off = self.offset if offset is None else offset
        R = self.d.site_xmat[self.tcp].reshape(3, 3)
        p = self.d.site_xpos[self.tcp] + R[:, 0] * off
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.tcp])
        return p.copy(), q

    def set_center(self, offset):
        """Move the compliance centre without a jump in commanded wrench."""
        e_p = self.sp_pos - self.center_pose()[0]
        self.offset = offset
        self.sp_pos = self.center_pose()[0] + e_p
        self.sp_vel[:] = 0

    def hold_here(self):
        self.sp_pos, self.sp_quat = self.center_pose()
        self.sp_vel[:] = 0

    def error(self):
        p, q = self.center_pose()
        return (float(np.linalg.norm(self.sp_pos - p)),
                float(np.linalg.norm(quat_err(self.sp_quat, q))))

    def jac(self, point):
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr, point, self.body)
        return np.vstack([self._jp, self._jr])[:, self.dofs]

    def torque(self):
        d = self.d
        p, q = self.center_pose()
        J = self.jac(p)
        v = J @ d.qvel[self.dofs]
        Rt = quat_to_mat(self.sp_quat)            # tool-frame gains -> world
        K = Rt @ np.diag(self.kp) @ Rt.T
        D = Rt @ np.diag(self.dp) @ Rt.T
        F = K @ (self.sp_pos - p) + D @ (self.sp_vel - v[:3])
        M = self.kr * quat_err(self.sp_quat, q) - self.dr * v[3:]
        tau = (J.T @ np.r_[F, M] + d.qfrc_bias[self.dofs]
               + self.damp_comp * d.qvel[self.dofs])
        if self.payload > 0:
            R = d.site_xmat[self.tcp].reshape(3, 3)
            pc = d.site_xpos[self.tcp] + R[:, 0] * self.com_offset
            tau += self.jac(pc)[:3].T @ np.array([0, 0, self.payload * 9.81])
        return np.clip(tau, -self.lim, self.lim)


class PegInHole:
    def __init__(self, offset=(0.003, 0.002), insert_speed=0.015,
                 scene=SCENE, start="hook"):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.geo = {k: float(m.numeric(k).data[0]) for k in
                    ("chamfer", "hole_radius", "hole_depth", "peg_radius",
                     "flange_radius", "hook_slot", "hook_slide", "hook_stem",
                     "hook_margin")}
        self.geo["hook_tip"] = tuple(m.numeric("hook_tip").data)
        # Nominal in-hand geometry along the TCP x axis (the housing axis,
        # towards the plug), gripped at the middle of the grip groove. The
        # housing's local z points away from the plug tip.
        grip_z = m.site_pos[m.site("peg_grip").id][2]
        self.tip_offset = grip_z - m.site_pos[m.site("peg_tip").id][2]
        self.com_offset = grip_z - m.body_ipos[m.body("peg").id][2]

        assert start in ("hook", "hand")
        self.start = start
        self.at_hook = start == "hook"
        mujoco.mj_resetDataKeyframe(m, d, m.key(
            "hook" if start == "hook" else "ready").id)
        # Hook task: fingers start open. Otherwise start already gripping.
        d.ctrl[6] = GRIP_OPEN if start == "hook" else GRIP_CLOSE
        mujoco.mj_forward(m, d)
        self.ctl = ImpedanceController(m, d, self.com_offset)
        self.peg_body = m.body("peg").id
        self.fixture_body = m.body("fixture").id
        self.hook_body = m.body("hook").id
        self.peg_mass = float(m.body_subtreemass[self.peg_body])
        self.ctl.payload = 0.0 if start == "hook" else self.peg_mass
        self.peg_dof = m.jnt_dofadr[m.joint("peg_free").id]

        # Hook frame (x along the resting cable, y to the robot's left, z up)
        # and the cable-centre waypoints that lead out along the J.
        self.hook_pos = d.xpos[self.hook_body].copy()
        self.hook_R = d.xmat[self.hook_body].reshape(3, 3).copy()
        self.hook_pts = {k: d.site_xpos[m.site(k).id].copy() for k in
                         ("hook_rest", "hook_lift", "hook_corner",
                          "hook_exit")}

        # Hole frame: site z axis points out of the opening.
        sid = m.site("hole_top").id
        Rh = d.site_xmat[sid].reshape(3, 3).copy()
        self.hole_top = d.site_xpos[sid].copy()
        self.axis = -Rh[:, 2]                     # unit vector into the hole
        # Lateral aiming offset (dy, dz): fixture local y = world y,
        # local x = world z for the horizontal bore.
        self.aim_offset = np.column_stack([Rh[:, 1], Rh[:, 0]]) @ np.asarray(
            offset, float)
        self.insert_speed = insert_speed
        self.phase = "start"
        self.log = []

    # ------------------------------------------------------------ sensing
    def peg_state(self):
        """tip, rear, depth along bore axis, lateral offset, tilt [deg]."""
        d, m = self.d, self.m
        tip = d.site_xpos[m.site("peg_tip").id].copy()
        rear = d.site_xpos[m.site("peg_rear").id].copy()
        rel = tip - self.hole_top
        depth = float(rel @ self.axis)
        lat = float(np.linalg.norm(rel - depth * self.axis))
        u = (tip - rear) / np.linalg.norm(tip - rear)
        tilt = math.degrees(math.acos(min(1.0, float(u @ self.axis))))
        return tip, rear, depth, lat, tilt

    def contact_force(self, body_a, body_b):
        """Net world-frame force on body_a from body_b."""
        m, d = self.m, self.d
        total = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} != {body_a, body_b}:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]   # force on geom2 from geom1
            total += fw if b2 == body_a else -fw
        return total

    # ------------------------------------------------------------ motions
    def move_tcp(self, pos, quat, T, label):
        """Min-jerk move of the TCP (compliance centre must be at the TCP)."""
        assert self.ctl.offset == 0.0
        self.phase = label
        p0, q0 = self.ctl.sp_pos.copy(), self.ctl.sp_quat.copy()
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            s, ds = min_jerk(k / n)
            self.ctl.sp_pos = p0 + s * (pos - p0)
            self.ctl.sp_vel = (pos - p0) * ds / T
            self.ctl.sp_quat = slerp(q0, quat, s)
            yield
        self.ctl.sp_vel[:] = 0

    def move_arc(self, pos, quat, T, label):
        """Min-jerk move that swings the TCP round the base (world z) axis:
        yaw, radius and height are interpolated instead of position, so the
        arm turns rather than cutting straight across the workspace."""
        assert self.ctl.offset == 0.0
        self.phase = label
        p0, q0 = self.ctl.sp_pos.copy(), self.ctl.sp_quat.copy()

        def cyl(p):
            return np.array([math.atan2(p[1], p[0]), math.hypot(p[0], p[1]),
                             p[2]])
        c0 = cyl(p0)
        dc = cyl(pos) - c0
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            s, ds = min_jerk(k / n)
            psi, r, z = c0 + s * dc
            c, sn = math.cos(psi), math.sin(psi)
            self.ctl.sp_pos = np.array([r * c, r * sn, z])
            dp_ds = np.array([dc[1] * c - r * sn * dc[0],
                              dc[1] * sn + r * c * dc[0], dc[2]])
            self.ctl.sp_vel = dp_ds * ds / T
            self.ctl.sp_quat = slerp(q0, quat, s)
            yield
        self.ctl.sp_vel[:] = 0

    def wait(self, T, label=None):
        if label:
            self.phase = label
        for _ in range(int(round(T / self.m.opt.timestep))):
            yield

    def settle(self, label, pos_tol=5e-5, rot_tol=math.radians(0.05),
               hold=0.15, t_max=3.0):
        """Hold until the pose error has stayed within tolerance for `hold`
        seconds, so a slow residual swing can't pass at a zero crossing."""
        self.phase = label
        dt, t, ok = self.m.opt.timestep, 0.0, 0.0
        while t < t_max:
            ep, er = self.ctl.error()
            ok = ok + dt if (ep < pos_tol and er < rot_tol) else 0.0
            if ok >= hold:
                return
            t += dt
            yield

    def grip(self, cmd, T, label):
        self.phase = label
        c0 = self.d.ctrl[6]
        n = int(round(T / self.m.opt.timestep))
        for k in range(1, n + 1):
            self.d.ctrl[6] = c0 + (cmd - c0) * min(1.0, 2 * k / n)
            yield

    def pick_from_hook(self):
        """Grasp the housing by its grip groove where it rests in the J-hook
        and lift it out along the J: up off the slot floor, sideways to the
        stem, up and out."""
        ctl, m, d = self.ctl, self.m, self.d
        # The housing's pose is known (from the scene, standing in for
        # perception). Grip it from the side (from above), across the groove.
        rear = d.site_xpos[m.site("peg_rear").id].copy()
        tip = d.site_xpos[m.site("peg_tip").id].copy()
        u = (tip - rear) / np.linalg.norm(tip - rear)
        q = axis_angle_quat([0, 0, 1], math.atan2(u[1], u[0]))
        grasp = d.site_xpos[m.site("peg_grip").id].copy()

        yield from self.wait(0.5, "start")
        # Line up above the groove, then come straight down over it.
        yield from self.move_tcp(grasp + [0, 0, 0.06], q, 1.5, "approach")
        yield from self.settle("approach")
        yield from self.move_tcp(grasp, q, 1.0, "reach in")
        yield from self.settle("reach in")
        yield from self.grip(GRIP_CLOSE, 0.5, "grasp")
        yield from self.wait(0.3)
        ctl.payload = self.peg_mass
        rest = self.hook_pts["hook_rest"]
        for label, site, T in (("lift out: lift", "hook_lift", 0.5),
                               ("lift out: slide", "hook_corner", 1.5),
                               ("lift out: up", "hook_exit", 1.5)):
            yield from self.move_tcp(grasp + self.hook_pts[site] - rest, q, T,
                                     label)
        yield from self.wait(0.2)

    def insert(self):
        """Compliant push along the bore axis, compliance centre at the tip."""
        ctl, dt, a = self.ctl, self.m.opt.timestep, self.axis
        ctl.set_center(self.tip_offset)
        ctl.set_gains("compliant")
        self.phase = "insert: entry"
        target = self.geo["hole_depth"] + 0.003   # 3 mm past the floor: preload
        lat_relax_tau = 0.25                      # s, lateral setpoint follows
        seated_t = t = 0.0
        while True:
            t += dt
            tip, _, depth, _, _ = self.peg_state()
            if depth > self.geo["chamfer"] + 0.002:
                self.phase = "insert: bore"
                # Inside the bore the walls locate the peg; let the lateral
                # setpoint drift to the measured position to shed side load.
                e = ctl.center_pose()[0] - ctl.sp_pos
                ctl.sp_pos += (dt / lat_relax_tau) * (e - (e @ a) * a)
            sp_depth = (ctl.sp_pos - self.hole_top) @ a
            # Slow down for the last 5 mm to soften the impact on the floor.
            v = (self.insert_speed if depth < self.geo["hole_depth"] - 0.005
                 else min(self.insert_speed, 0.005))
            if sp_depth < target:
                ctl.sp_pos += a * min(v * dt, target - sp_depth)
                ctl.sp_vel = a * v
            else:
                ctl.sp_vel[:] = 0
            v_ax = self.d.qvel[self.peg_dof:self.peg_dof + 3] @ a
            seated = (depth > self.geo["hole_depth"] - 0.001
                      and abs(v_ax) < 0.002)
            seated_t = seated_t + dt if seated else 0.0
            if seated_t > 0.3:
                self.phase = "seated"
                return
            # Jam guard: commanded tip far past the real one (pushing hard,
            # ~K_axial * 12 mm) before the tip has entered the hole.
            if (ctl.sp_pos - tip) @ a > 0.012 and depth < 0.002:
                self.phase = "JAMMED"
                return
            if t > 12.0:
                self.phase = "TIMEOUT"
                return
            yield

    def program(self):
        ctl = self.ctl
        a = self.axis
        # TCP pose that puts the (nominal) tip 8 mm in front of the opening,
        # deliberately off-axis by aim_offset.
        pre_tcp = self.hole_top + self.aim_offset - a * (0.008 + self.tip_offset)
        transfer = pre_tcp - a * 0.04

        if self.start == "hook":
            # Tool along the bore axis, fingers horizontal (as in the hook).
            q_tool = axis_angle_quat([0, 0, 1], math.atan2(a[1], a[0]))
            yield from self.pick_from_hook()
            self.at_hook = False
            # Swing round to the hole at the exit height (clear of the hook
            # plate), then come down in front of it.
            yield from self.move_arc(np.r_[transfer[:2], ctl.sp_pos[2]],
                                     q_tool, 3.0, "turn")
            yield from self.move_tcp(transfer, q_tool, 1.2, "lower")
        else:
            q_tool = ctl.sp_quat.copy()           # keep the start orientation
            yield from self.wait(0.5, "settle (holding cable)")
            yield from self.move_tcp(transfer, q_tool, 2.0, "transfer")
        yield from self.move_tcp(pre_tcp, q_tool, 0.8, "pre-insert")
        yield from self.settle("pre-insert")
        self.pre_insert = self.peg_state()
        yield from self.insert()
        if self.phase != "seated":
            # Abort: back off along the axis with the peg still in hand.
            fail = self.phase
            ctl.set_center(0.0)
            ctl.set_gains("stiff")
            ctl.hold_here()
            p_tcp, q_tcp = ctl.center_pose()
            yield from self.move_tcp(p_tcp - a * 0.03, q_tcp, 0.8,
                                     f"abort ({fail})")
            yield from self.wait(0.5)
            self.phase = fail
            return
        yield from self.wait(0.3)
        # Unload: drop the axial preload and side force, so the peg doesn't
        # spring back when the fingers open; then let go.
        ctl.hold_here()
        yield from self.wait(0.2, "unload")
        ctl.payload = 0.0
        yield from self.grip(GRIP_OPEN, 0.6, "release")
        ctl.set_center(0.0)
        ctl.set_gains("stiff")
        ctl.hold_here()
        p_tcp, q_tcp = ctl.center_pose()
        up = p_tcp + [0, 0, 0.03]                # lift the open fingers clear
        yield from self.move_tcp(up, q_tcp, 0.8, "retract")
        yield from self.move_tcp(up - a * 0.08, q_tcp, 1.0, "retract")
        yield from self.wait(1.0, "done")

    # ------------------------------------------------------------ main loop
    def run(self, on_frame=None, fps=30, realtime_viewer=None):
        m, d = self.m, self.d
        prog = self.program()
        dt = m.opt.timestep
        frame_every = max(1, int(round(1.0 / (fps * dt))))
        step = 0
        wall0 = time.time()
        jids = [m.joint(f"joint{i}").id for i in range(1, 7)]
        lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        self.min_joint_margin = np.inf
        while True:
            mujoco.mj_step1(m, d)
            try:
                next(prog)
            except StopIteration:
                break
            d.ctrl[:6] = self.ctl.torque()
            mujoco.mj_step2(m, d)
            step += 1
            q = d.qpos[:6]
            self.min_joint_margin = min(self.min_joint_margin,
                                        float(np.min(np.r_[q - lo, hi - q])))
            if step % 10 == 0:
                self.record()
            if on_frame and step % frame_every == 0:
                on_frame(self)
            if realtime_viewer is not None:
                if not realtime_viewer.is_running():
                    break
                if step % 16 == 0:
                    realtime_viewer.sync()
                    lag = d.time - (time.time() - wall0)
                    if lag > 0:
                        time.sleep(lag)
        return self.result()

    def record(self):
        tip, rear, depth, lat, tilt = self.peg_state()
        f = self.contact_force(self.peg_body, self.fixture_body)
        f_ax = float(f @ self.axis)
        f_lat = float(np.linalg.norm(f - f_ax * self.axis))
        f_hook = float(np.linalg.norm(
            self.contact_force(self.peg_body, self.hook_body)))
        # Cable centre in the hook frame (y: robot's left, z: up).
        hc = self.hook_R.T @ (self.d.xpos[self.peg_body] - self.hook_pos)
        tcp, _ = site_pose(self.m, self.d, "tcp")
        self.log.append(dict(
            t=self.d.time, phase=self.phase,
            tip_x=tip[0], tip_y=tip[1], tip_z=tip[2],
            depth=depth, lat=lat, tilt=tilt,
            f_axial=f_ax, f_lateral=f_lat,
            f_hook=f_hook, hook_y=hc[1], hook_z=hc[2],
            tcp_x=tcp[0], tcp_y=tcp[1], tcp_z=tcp[2],
            sp_x=self.ctl.sp_pos[0], sp_y=self.ctl.sp_pos[1],
            sp_z=self.ctl.sp_pos[2], grip=self.d.ctrl[6]))

    def result(self):
        tip, rear, depth, lat, tilt = self.peg_state()
        g = self.geo
        ins = [r for r in self.log if r["phase"].startswith("insert")]
        res = dict(
            initial_offset_mm=self.pre_insert[3] * 1e3,
            final_tip_offset_mm=lat * 1e3,
            radial_clearance_mm=(g["hole_radius"] - g["peg_radius"]) * 1e3,
            insertion_depth_mm=depth * 1e3,
            hole_depth_mm=g["hole_depth"] * 1e3,
            chamfer_mm=g["chamfer"] * 1e3,
            tilt_deg=tilt,
            max_contact_force_N=max((math.hypot(r["f_axial"], r["f_lateral"])
                                     for r in ins), default=0.0),
            insert_time_s=(ins[-1]["t"] - ins[0]["t"]) if ins else float("nan"),
            min_joint_limit_margin_rad=self.min_joint_margin,
            phase=self.phase,
        )
        if self.start == "hook":
            out = [r for r in self.log if r["phase"].startswith("lift out")]
            res["max_hook_contact_force_N"] = max(
                (r["f_hook"] for r in out), default=0.0)
            res["lift_out_time_s"] = (out[-1]["t"] - out[0]["t"]) if out \
                else float("nan")
        clearance = g["hole_radius"] - g["peg_radius"]
        res["success"] = bool(depth > g["hole_depth"] - 0.0015
                              and lat < clearance + 3e-4
                              and tilt < 2.0 and self.phase == "done")
        return res


# ---------------------------------------------------------------- outputs
class VideoWriter:
    # camera, label, visible geom groups (2 = fixtures, 5 = fixture section)
    HOLE_CAMS = (("overview", "overview", (0, 1, 2)),
                 ("closeup", "close-up", (0, 1, 2)),
                 ("section", "section view (fixture cut through bore axis)",
                  (0, 1, 5)))
    HOOK_CAMS = (("overview", "overview", (0, 1, 2)),
                 ("hook_closeup", "J-hook close-up", (0, 1, 2)),
                 ("hook_back", "J-hook from behind", (0, 1, 2)))
    STORY = {"hand": ("settle (holding cable)", "pre-insert", "on chamfer",
                      "insert: bore", "release", "done"),
             "hook": ("start", "grasp", "lift out: slide", "lift out: up",
                      "turn", "on chamfer", "insert: bore", "done")}

    def __init__(self, sim: PegInHole, path, w=640, h=480, fps=30):
        import imageio.v2 as imageio
        from PIL import ImageFont
        self.r = mujoco.Renderer(sim.m, h, w)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                         quality=8, macro_block_size=8)
        self.opts = {}
        for _, _, groups in self.HOLE_CAMS + self.HOOK_CAMS:
            o = mujoco.MjvOption()
            o.geomgroup[:] = [1 if g in groups else 0 for g in range(6)]
            self.opts[groups] = o
        self.story = self.STORY[sim.start]
        self.w = w
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        except OSError:
            self.font = ImageFont.load_default()
        self.stills = {}

    def __call__(self, sim: PegInHole):
        from PIL import Image, ImageDraw
        cams = self.HOOK_CAMS if sim.at_hook else self.HOLE_CAMS
        panels = []
        for cam, _, groups in cams:
            self.r.update_scene(sim.d, camera=cam,
                                scene_option=self.opts[groups])
            panels.append(self.r.render())
        img = Image.fromarray(np.concatenate(panels, axis=1))
        _, _, depth, lat, tilt = sim.peg_state()
        draw = ImageDraw.Draw(img)
        if sim.at_hook:
            hc = sim.hook_R.T @ (sim.d.xpos[sim.peg_body] - sim.hook_pos)
            txt = (f"t={sim.d.time:5.2f}s  {sim.phase:<23s} "
                   f"cable in J-hook: {-hc[1] * 1e3:5.1f} mm right, "
                   f"{hc[2] * 1e3:5.1f} mm up from rest")
        else:
            txt = (f"t={sim.d.time:5.2f}s  {sim.phase:<23s} "
                   f"tip depth {depth * 1e3:6.1f} mm   "
                   f"tip off-axis {lat * 1e3:5.1f} mm   tilt {tilt:4.1f} deg")
        draw.rectangle([0, 0, img.width, 28], fill=(0, 0, 0))
        draw.text((10, 4), txt, fill=(255, 255, 255), font=self.font)
        for i, (_, label, _) in enumerate(cams):
            draw.text((i * self.w + 10, img.height - 28), label,
                      fill=(255, 255, 255), font=self.font)
        frame = np.asarray(img)
        self.writer.append_data(frame)
        self.stills[sim.phase] = frame          # last frame of each phase
        if (sim.phase.startswith("insert") and depth >= 0.0015
                and "on chamfer" not in self.stills):
            self.stills["on chamfer"] = frame   # peg sliding on the chamfer

    def storyboard(self, path):
        from PIL import Image
        keys = [k for k in self.story if k in self.stills]
        keys += [k for k in self.stills if k.startswith(("JAMMED", "abort"))]
        rows = [self.stills[k] for k in keys]
        Image.fromarray(np.concatenate(rows, axis=0)).save(path)

    def close(self):
        self.writer.close()


def save_csv(log, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)


def plot_hook(ax, sim: PegInHole):
    """Cable centre path through the J slot, seen from the robot side."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    g = sim.geo
    r, s, L, H, M = (g[k] * 1e3 for k in ("peg_radius", "hook_slot",
                                          "hook_slide", "hook_stem",
                                          "hook_margin"))
    tw, th = (v * 1e3 for v in g["hook_tip"])
    zb, zc = -r, -r + s
    zt, ztip = zc + H, zc + th
    x0, x1 = -tw / 2 - M, L + s / 2 + M
    ax.add_patch(Rectangle((x0, zb - 30), x1 - x0, zt - zb + 30,
                           color="#d6d6d1", zorder=0))
    ax.add_patch(Polygon([(-tw / 2, zb), (L + s / 2, zb), (L + s / 2, zt),
                          (L - s / 2, zt), (L - s / 2, zc), (tw / 2, zc),
                          (tw / 2, ztip), (-tw / 2, ztip)],
                         color="#55575a", zorder=1, label="J slot"))
    rows = [row for row in sim.log
            if row["phase"].startswith(("grasp", "lift out", "turn"))]
    x = np.array([-row["hook_y"] for row in rows]) * 1e3
    z = np.array([row["hook_z"] for row in rows]) * 1e3
    ax.plot(x, z, color="#d9480f", lw=2, zorder=3, label="neck centre")
    rf = g["flange_radius"] * 1e3
    i_exit = int(np.argmax(z > zt + rf))
    for i in (0, int(np.argmax(x > L - 1)), i_exit):
        ax.add_patch(Circle((x[i], z[i]), r, fill=False, ec="#d9480f",
                            ls="--", lw=1, zorder=3))
        ax.add_patch(Circle((x[i], z[i]), rf, fill=False, ec="#d9480f",
                            ls=":", lw=1, zorder=3,
                            label="flange" if i == 0 else None))
    ax.set_xlim(x0 - 5, x1 + 5)
    ax.set_ylim(zb - 30, z[i_exit] + rf + 8)
    ax.set_aspect("equal")
    ax.set_xlabel("to the robot's right [mm]")
    ax.set_ylabel("up [mm]")
    ax.set_title("Lifting the housing out of the J-hook")
    ax.legend(loc="lower right", fontsize=8)


def save_plot(sim: PegInHole, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, geo = sim.log, sim.geo
    t = np.array([r["t"] for r in L])
    ins = np.array([r["phase"].startswith(("insert", "seated", "JAM",
                                            "TIME")) for r in L])
    t0 = t[ins][0]
    sel = (t >= t0 - 0.3) & (t <= t[ins][-1] + 0.5)
    g = lambda k: np.array([r[k] for r in L])[sel]   # noqa: E731
    ts = t[sel] - t0
    hook = sim.start == "hook"
    fig, axs = plt.subplots(1, 3 + hook, figsize=(15 + 4 * hook, 4.2),
                            gridspec_kw=dict(width_ratios=[0.8] * hook
                                             + [1, 1, 1]))
    if hook:
        plot_hook(axs[0], sim)
    ax = axs[hook:]
    ax[0].plot(g("depth") * 1e3, g("lat") * 1e3, color="#d9480f", lw=2)
    if geo["chamfer"] > 0:
        ax[0].axvspan(0, geo["chamfer"] * 1e3, color="#4c6ef5", alpha=0.15,
                      label="chamfer zone")
    ax[0].axhline((geo["hole_radius"] - geo["peg_radius"]) * 1e3, color="k",
                  ls="--", lw=1, label="radial clearance")
    ax[0].set_xlabel("peg tip depth past hole opening [mm]")
    ax[0].set_ylabel("peg tip offset from bore axis [mm]")
    ax[0].set_title("Chamfer guidance")
    ax[0].legend()
    ax[0].set_xlim(left=-8)
    ax[1].plot(ts, g("depth") * 1e3, lw=2, label="peg tip depth")
    ax[1].axhline(geo["hole_depth"] * 1e3, color="k", ls="--", lw=1,
                  label="hole bottom")
    ax[1].set_xlabel("time since insertion start [s]")
    ax[1].set_ylabel("depth [mm]")
    ax[1].set_title("Insertion depth")
    ax[1].legend()
    ax[2].plot(ts, -g("f_axial"), lw=1.5, label="axial (resisting push)")
    ax[2].plot(ts, g("f_lateral"), lw=1.5, label="lateral")
    ax[2].set_xlabel("time since insertion start [s]")
    ax[2].set_ylabel("contact force on peg from fixture [N]")
    ax[2].set_title("Peg-fixture contact force")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--offset", nargs=2, type=float, default=[0.003, 0.002],
                    metavar=("DY", "DZ"),
                    help="deliberate sideways/vertical aiming error [m]")
    ap.add_argument("--speed", type=float, default=0.015,
                    help="insertion speed [m/s]")
    ap.add_argument("--start", choices=("hook", "hand"), default="hook",
                    help="hook: pick the cable out of the J-hook first; "
                         "hand: start already holding it")
    ap.add_argument("--scene", default=SCENE, help="scene XML to load")
    ap.add_argument("--tag", default="",
                    help="suffix for output file names, e.g. _nochamfer")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true",
                    help="live MuJoCo viewer (use mjpython on macOS)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sim = PegInHole(offset=args.offset, insert_speed=args.speed,
                    scene=args.scene, start=args.start)
    tag = args.tag
    t0 = time.time()
    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
            res = sim.run(realtime_viewer=v)
    elif args.no_video:
        res = sim.run()
    else:
        vw = VideoWriter(sim, os.path.join(OUT, f"peg_in_hole{tag}.mp4"))
        res = sim.run(on_frame=vw)
        vw.close()
        vw.storyboard(os.path.join(OUT, f"storyboard{tag}.png"))
    print(f"sim {sim.d.time:.1f}s in {time.time() - t0:.1f}s wall")
    save_csv(sim.log, os.path.join(OUT, f"log{tag}.csv"))
    save_plot(sim, os.path.join(OUT, f"insertion_plot{tag}.png"))
    for k, v in res.items():
        print(f"  {k:28s} {v:.3f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print("RESULT:", "SUCCESS" if res["success"] else "FAILED")
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
