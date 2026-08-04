# RL_Golf

A simulated golf environment in MuJoCo, built so an RL agent can learn to swing
a golf club.

## Where it's at

`main.py` builds the physical side of the problem: a full-body humanoid golfer
holding a driver with both hands, at address to a teed ball, and swings it
through a scripted reference swing. The swing is a placeholder — learning a good
one is the actual project. What matters here is that the body, the club and the
measurement are right.

```bash
python main.py                  # 3D viewer, quarter-speed swing
python main.py --report         # headless, prints the kinematic report
python main.py --csv swing.csv  # log every tracked joint every 2 ms
python main.py --xml golf.xml   # dump the generated MJCF
```

## The model

* **36 actuated DOF**: hips (3), knee, ankle (3) per leg; lumbar and thoracic
  spine (3 each); neck (2); shoulder (3), elbow (2), wrist (2) per arm.
* **Scaled from height and mass** — segment lengths and masses use Winter (2009)
  proportions, so `--height 1.62 --mass 60` gives a consistently smaller golfer.
* **Human joint limits and torque limits**, driven by PD position servos. The
  lumbar spine really does only give ~13° of axial rotation each way; the
  thoracic spine supplies most of the shoulder turn.
* **Both hands on the club.** The lead hand grips it rigidly with a built-in
  grip angle; the trail hand is held on the shaft by an equality constraint. The
  arms are therefore a genuine closed kinematic loop, as in a real swing.
* **Legal driver, ball and tee** — 1.14 m / 315 g club, 42.7 mm / 45.9 g ball.
* Feet pinned to the ground at heel and toe (`--base feet`, default), or free
  for full balance physics (`--base free`), or pelvis welded (`--base pinned`).

The address pose, the clubface angle, the ball position and the depth of the
swing arc are all *solved*, not hard-coded — see the two-pass construction in
`GolfSwingSim.__init__`.

## Joint tracking

`SwingTracker` is the part the RL agent will use. Every major joint centre is a
site on the model, and the tracker exposes them:

| method | what you get |
| --- | --- |
| `positions()` | (N, 3) world positions of all 24 landmarks |
| `egocentric()` | same, relative to the golfer (origin at the pelvis, x = facing, z = up) |
| `parent_relative()` | each landmark minus its parent in the kinematic chain — the segment vectors |
| `pairwise_offsets()` | (N, N, 3) — every landmark relative to every other |
| `pairwise_distances()` | (N, N) distance matrix |
| `joint_angles()` / `joint_velocities()` | per-joint, by name |
| `metrics()` | hip turn, shoulder turn, X-factor, spine tilt, wrist cock, clubhead speed, attack angle, club path |
| `observation()` | flat vector of all of the above, ready for a policy |

## What the scripted swing is not

It reaches ~18 m/s (40 mph) of clubhead speed, about a third of a competent
driver, and comes in steep and out-to-in. That's a structural limit of an
open-loop PD script: with both feet planted the pelvis runs out of rotation
about 40° past square, so the body stops turning exactly when the clubhead needs
it, and nothing in the script exploits the ground reaction or the whip of the
release. Finding a swing that does is the RL problem.

## Next

* Gymnasium environment wrapping `GolfSwingSim` (observation from
  `SwingTracker.observation()`, action = the 36 servo targets)
* Reward on ball speed / launch conditions, with penalties for joint-limit and
  torque saturation
* Train with SB3 (already in the venv alongside MuJoCo and torch)
