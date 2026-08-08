"""The simulation object: builds the model, and runs the swing.

`GolfSwingSim` is composed from three mixins, one per stage of getting a golfer
onto the tee:

    IKSolver      golf/ik.py       inverse kinematics against the live model
    AddressSolver golf/posture.py  everything before the swing starts
    SwingPlanner  golf/planner.py  turning the script into a swing that works

They are separate files because they are separate concerns, but they are one
object because they all operate on the same `model`/`data`/`tracker`.
"""

from __future__ import annotations

from typing import Dict, Optional

import mujoco
import numpy as np

from .anthropometry import Anthropometry
from .equipment import Club
from .ik import IKSolver
from .model import build_model_xml
from .planner import SwingPlanner
from .posture import AddressSolver
from .swing import SwingController, swing_script
from .tracking import SwingTracker
from .util import DEG


class GolfSwingSim(IKSolver, AddressSolver, SwingPlanner):
    """A golfer at address, and a swing that hits the ball.

    Construction is a two-pass process, because a few things can only be
    measured once the model exists:

    pass 1  build -> solve the address pose -> measure the clubface heading and
            where the feet end up
    pass 2  rebuild with the face squared and the feet pinned there, re-solve
            address, settle under gravity, plan the swing, then run a few dry
            swings to tune the arc and place the ball
    """

    def __init__(self, anthro: Optional[Anthropometry] = None,
                 club: Optional[Club] = None, base: str = "feet",
                 timestep: float = 2e-4, calibrate: bool = True,
                 strength: float = 1.0, tempo: float = 1.0,
                 lead: float = 0.07, verbose: bool = True):
        self.anthro = anthro or Anthropometry()
        self.club = club or Club()
        self.base = base
        self.timestep = timestep
        self.verbose = verbose
        self.strength = strength

        # ---- build until the club sits square and level at address ---------
        # Each pass measures the address pose, works out how the head has to be
        # mounted and where the feet ended up, and rebuilds.  It converges in
        # two or three passes: moving the head moves the clubhead site, which
        # the address IK aims at, which moves the pose slightly.
        self._pins = None
        head_quat = None
        for attempt in range(4):
            self._load(build_model_xml(self.anthro, self.club, base, timestep,
                                       head_quat=head_quat,
                                       foot_pins=self._pins, strength=strength))
            self._solve_address(warn=False)
            if self._pins is None and base == "feet":
                self._pins = self._foot_pin_targets()
            if attempt and abs(self.face_heading()) < 0.25:
                break
            head_quat = self.head_alignment()
        if self.verbose:
            grip = self.address_grip_residual
            note = (f", trail hand {grip * 100:.1f} cm off the grip"
                    if grip > 5e-3 else "")
            print(f"  address: clubface {self.face_heading():+.2f} deg open, "
                  f"{self.face_loft():.1f} deg loft{note}")
        # The servo targets are the *solved* angles and the stored state is the
        # pose those targets settle into, so the swing starts from a genuine
        # equilibrium instead of twitching on the first step.
        self.address_angles: Dict[str, float] = self.tracker.joint_angles()
        self._settle(0.5)
        self._take_out_droop()
        self.qpos_address = self.data.qpos.copy()
        self.address_clubhead = self.tracker.positions()[
            self.tracker.index["clubhead"]].copy()

        # ---- plan the swing ------------------------------------------------
        script = swing_script(self.anthro, self.address_angles, tempo)
        self.plan_legs(script)
        self.plan_lead_arm(script)
        self.grip_residuals = self.plan_trail_arm(script)
        self.controller = SwingController(script, self.model, lead=lead * tempo)

        self.ball_pos = self.tracker.positions()[self.tracker.index["ball"]].copy()
        if calibrate:
            self.tune_swing(script)
            self.place_ball()
        self.report_grip_reach(self.grip_residuals)
        self.reset()

    # ---- construction ------------------------------------------------------
    def _load(self, xml: str) -> None:
        """Compile an MJCF string and cache the ids everything else needs."""
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.tracker = SwingTracker(self.model, self.data, self.anthro)

        self.ball_jadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        # The ball's freejoint is declared before the golfer's, so the root is
        # *not* at qpos[0:7] -- always look it up.
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = int(self.model.jnt_qposadr[rid]) if rid >= 0 else -1
        self.root_dadr = int(self.model.jnt_dofadr[rid]) if rid >= 0 else -1
        self.tee_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         "tee")
        self._ball_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "ball_g")
        self._head_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "club_head_g")

    def _set_pose(self, angles: Dict[str, float]) -> None:
        """Write joint angles (degrees) straight into qpos."""
        for n, v in angles.items():
            self.data.qpos[self.tracker.qadr[n]] = v * DEG

    def actuator_id(self, joint: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                 f"act_{joint}")

    # ---- ball --------------------------------------------------------------
    def _set_ball(self, pos: np.ndarray) -> None:
        a = self.ball_jadr
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.model.body_pos[self.tee_bid] = np.array([pos[0], pos[1], 0.0])

    def ball_contact(self) -> bool:
        """True while the clubhead is touching the ball."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._ball_gid in pair and self._head_gid in pair:
                return True
        return False

    # ---- running -----------------------------------------------------------
    def reset(self, place_ball: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos_address
        self.data.qvel[:] = 0.0
        if place_ball:
            self._set_ball(self.ball_pos)
        self.controller.apply(self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def step(self, t: Optional[float] = None) -> None:
        self.controller.apply(self.data, self.data.time if t is None else t)
        mujoco.mj_step(self.model, self.data)

    @property
    def duration(self) -> float:
        return self.controller.duration
