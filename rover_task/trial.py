"""One closed-loop trial: build the true world (tilted hook and port,
calibration errors), run the task program against the simulated robot, and
score the result against ground truth."""

from __future__ import annotations

import math
import os
import time

import numpy as np

from geometry import (P, Pose, hook_nominal, port_nominal, rot_angle,
                      slot_in_hook, tilted)
from program import TaskProgram
from sim import SimRobot

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "scene.xml")


def sample_errors(rng, level="realistic"):
    """Per-trial sensing / calibration errors."""
    if level == "none":
        return {}
    return dict(
        img_noise=2.0,                                     # grey levels (sigma)
        img_gain=float(rng.uniform(0.85, 1.15)),
        cam_pos=rng.normal(0, 0.0005, 3),                  # hand-eye, 0.5 mm
        cam_rot=rng.normal(0, math.radians(0.15), 3),      # hand-eye, 0.15 deg
        fovy_scale=float(1 + rng.normal(0, 0.002)),        # focal length 0.2 %
        enc=rng.normal(0, math.radians(0.03), 6),          # joint offsets
    )


class Video:
    """Three panels: overview, the fixture being worked on, wrist camera."""

    def __init__(self, sim_m, path, fps=30, w=640, h=480):
        import imageio.v2 as imageio
        import mujoco
        from PIL import ImageFont
        self.every = int(round(1.0 / (fps * sim_m.opt.timestep)))
        self.r = mujoco.Renderer(sim_m, h, w)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                         quality=7, macro_block_size=8)
        self.w, self.h = w, h
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
        except OSError:
            self.font = ImageFont.load_default()
        self.stills = {}

    def __call__(self, sim):
        from PIL import Image, ImageDraw
        side = "port_view" if any(s in sim.phase for s in (
            "rotate", "port", "insert", "seated", "unload", "release",
            "retreat", "pre-insert", "JAM", "STUCK", "TIME", "done", "home")) \
            and "hook" not in sim.phase else "hook_view"
        panels = []
        for cam in ("overview", side, "wrist"):
            self.r.update_scene(sim.d, camera=cam)
            panels.append(self.r.render())
        img = Image.fromarray(np.concatenate(panels, axis=1))
        dr = ImageDraw.Draw(img)
        _, depth, lat, tilt = sim.peg_state()
        txt = f"t={sim.d.time:6.2f}s  {sim.phase:<22s}"
        if any(s in sim.phase for s in ("insert", "seated", "unload", "release")):
            txt += f"  tip depth {depth * 1e3:5.1f} mm  off-axis {lat * 1e3:4.1f} mm  tilt {tilt:4.1f} deg"
        dr.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
        dr.text((8, 3), txt, fill=(255, 255, 255), font=self.font)
        for i, lab in enumerate(("overview", side.replace("_", " "), "wrist camera")):
            dr.text((i * self.w + 8, self.h - 22), lab, fill=(255, 255, 255),
                    font=self.font)
        frame = np.asarray(img)
        self.writer.append_data(frame)
        self.stills[sim.phase] = frame

    def storyboard(self, path, keys):
        from PIL import Image
        rows = [self.stills[k] for k in keys if k in self.stills]
        if rows:
            Image.fromarray(np.concatenate(rows, axis=0)).save(path)

    def close(self):
        self.writer.close()
        self.r.close()


def kin_trial(hook_rpy_deg=(0, 0, 0), port_rpy_deg=(0, 0, 0), seed=0):
    """Is the job feasible at all with PERFECT perception? Runs the program's
    planning steps on the true poses (housing where physics left it) and
    reports the first stage that fails. Used to check that the closed-loop
    program's safe aborts are justified."""
    import numpy as np  # noqa: F811
    from kin import Kin
    from planner import Planner
    T_hook = tilted(hook_nominal(P), *np.radians(hook_rpy_deg))
    T_port = tilted(port_nominal(P), *np.radians(port_rpy_deg))
    sim = SimRobot(XML, T_hook, T_port, P)          # only to settle the housing
    y = float(sim.settle_info["neck_slot"][1])
    k = Kin(XML, P)
    pl = Planner(k, P)
    k.set_fixture("hook", T_hook)
    k.set_fixture("port", T_port)
    T_slot = T_hook @ slot_in_hook(P)
    k.set_peg_free(pl.peg_rest_pose(T_slot, y))
    home = np.array(P.home_q)
    out = dict(seed=seed, hook_roll=hook_rpy_deg[0], hook_pitch=hook_rpy_deg[1],
               hook_yaw=hook_rpy_deg[2], port_roll=port_rpy_deg[0],
               port_pitch=port_rpy_deg[1], port_yaw=port_rpy_deg[2],
               neck_true_y_mm=y * 1e3)

    def fail(stage, why):
        out.update(feasible=False, stage=stage, reason=str(why)[:300])
        return out

    v1 = pl.plan_view(T_hook, home, holding=False)
    if not v1.ok:
        return fail("hook view", v1.reason)
    v2 = pl.plan_view(T_port, v1.data["q"], holding=False)
    if not v2.ok:
        return fail("port view", v2.reason)
    h = pl.plan_hook(T_hook, y, v1.data["q"])
    if not h.ok:
        return fail("hook grasp/lift-out", h.reason)
    r = pl.plan_rotate(h.data["q_end"])
    if not r.ok:
        return fail("rotate 180", r.reason)
    pv = pl.plan_view(T_port, r.data["q"], holding=True)
    if not pv.ok:
        return fail("port view holding", pv.reason)
    pp = pl.plan_port(T_port, pv.data["q"])
    if not pp.ok:
        return fail("port insertion", pp.reason)
    out.update(feasible=True, stage="", reason="", phi=h.data["phi"],
               psi=pp.data["psi"], margin_hook=h.margin, margin_port=pp.margin)
    return out


def pose_err(T_est: Pose, T_true: Pose):
    return (float(np.linalg.norm(T_est.t - T_true.t)) * 1e3,
            math.degrees(rot_angle(T_est.R @ T_true.R.T)))


def run_trial(hook_rpy_deg=(0, 0, 0), port_rpy_deg=(0, 0, 0), errors="none",
              seed=0, video_path=None, storyboard_path=None, verbose=False):
    rng = np.random.default_rng(seed)
    T_hook = tilted(hook_nominal(P), *np.radians(hook_rpy_deg))
    T_port = tilted(port_nominal(P), *np.radians(port_rpy_deg))
    err = sample_errors(rng, errors) if isinstance(errors, str) else errors
    t0 = time.time()
    sim = SimRobot(XML, T_hook, T_port, P, errors=err, rng=rng)
    vid = None
    if video_path:
        vid = Video(sim.m, video_path)
        sim.video = vid
    prog = TaskProgram(sim, XML, P, verbose=verbose)
    res = prog.run()
    if vid:
        vid.close()
        if storyboard_path:
            vid.storyboard(storyboard_path, [
                "hook survey", "hook centre", "grasp", "lift out: slide",
                "lift out: up the stem", "rotate 180", "port centre",
                "insert: bore", "release", "done"])

    # ---------------- ground-truth scoring
    tip, depth, lat, tilt = sim.peg_state()
    clearance = P.hole_radius - P.plug_radius
    out = dict(seed=seed, hook_roll=hook_rpy_deg[0], hook_pitch=hook_rpy_deg[1],
               hook_yaw=hook_rpy_deg[2], port_roll=port_rpy_deg[0],
               port_pitch=port_rpy_deg[1], port_yaw=port_rpy_deg[2],
               outcome=res["outcome"], kind=res.get("kind"),
               reason=res.get("reason", "")[:300],
               final_depth_mm=depth * 1e3, final_lat_mm=lat * 1e3,
               final_tilt_deg=tilt, sim_time_s=sim.d.time,
               wall_s=time.time() - t0,
               neck_true_y_mm=float(sim.settle_info["neck_slot"][1]) * 1e3,
               max_robot_env_N=sim.max_robot_env_force,
               robot_env_contacts=";".join(sorted({f"{a}-{b}@{ph}" for a, b, ph in sim.robot_env_contacts}))[:300],
               max_peg_port_N=sim.max_peg_port_force,
               max_peg_hook_lift_N=max([v for k, v in sim.max_peg_hook_force.items()
                                        if k.startswith("lift out")] or [0.0]),
               captures=sim.n_captures)
    out.update({f"plan_{k}": v for k, v in res.get("plan", {}).items()})
    est = res.get("est", {})
    if "T_hook" in est:
        out["hook_pos_err_mm"], out["hook_rot_err_deg"] = pose_err(est["T_hook"], T_hook)
    if "T_port" in est:
        out["port_pos_err_mm"], out["port_rot_err_deg"] = pose_err(est["T_port"], T_port)
    elif "T_port_survey" in est:
        out["port_pos_err_mm"], out["port_rot_err_deg"] = pose_err(est["T_port_survey"], T_port)
    if "y_neck" in est:
        out["neck_est_err_mm"] = (est["y_neck"] * 1e3) - out["neck_true_y_mm"]
    if hasattr(sim, "in_hand"):
        out["in_hand_mm"] = float(np.linalg.norm(sim.in_hand[0])) * 1e3
        out["in_hand_deg"] = math.degrees(float(np.linalg.norm(sim.in_hand[1])))
    seated_truth = (depth > P.hole_depth - 0.0015 and lat < clearance + 0.0003
                    and tilt < 2.0)
    collided = sim.max_robot_env_force > 2.0
    out["seated_truth"] = bool(seated_truth)
    out["collision"] = bool(collided)
    if res["outcome"] == "success" and seated_truth and not collided:
        out["result"] = "success"
    elif res["outcome"] == "safe_abort" and not collided:
        out["result"] = "safe_abort"
    else:
        out["result"] = "failure"
    out["insert"] = res.get("insert")
    sim.close()
    return out
