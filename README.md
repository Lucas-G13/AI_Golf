# RL_Golf

A simulated golf environment in MuJoCo, built so an RL agent can learn to swing
a golf club.

## Where it's at

The `golf` package builds the physical side of the problem: a full-body
humanoid golfer holding a driver with both hands, at address to a teed ball,
and swings it through a scripted reference swing. The swing is a placeholder —
learning a good one is the actual project. What matters here is that the body,
the club and the measurement are right.

```bash
python main.py                  # 3D viewer, quarter-speed swing
python main.py --report         # headless, prints the kinematic report
python main.py --csv swing.csv  # log every tracked joint every 2 ms
python main.py --xml golf.xml   # dump the generated MJCF
```

**Viewer controls:** the swing plays once and then holds the finish so you can
orbit around it. Space (or Enter / Backspace / R) swings again, Esc quits, and
`--loop` restores repeat-forever if you want it running unattended.

## Layout

| file | what's in it |
| --- | --- |
| `main.py` | CLI only |
| `golf/anthropometry.py` | how big the golfer is (Winter 2009 proportions) |
| `golf/equipment.py` | the club, ball and tee |
| `golf/joints.py` | every joint, its limits and strength; the address pose |
| `golf/landmarks.py` | what gets measured, and the kinematic chain |
| `golf/model.py` | generates the MJCF |
| `golf/tracking.py` | `SwingTracker` — reading the body |
| `golf/swing.py` | the scripted keyframes and their interpolation |
| `golf/ik.py` | inverse kinematics against the live model |
| `golf/posture.py` | getting the golfer to address |
| `golf/planner.py` | turning the script into a swing that works |
| `golf/sim.py` | `GolfSwingSim` — ties it together |
| `golf/report.py` | running a swing, logging it, showing it |
| `golf/launch.py` | impact → launch conditions → carry and curve |
| `golf/env.py` | `GolfEnv` — the Gymnasium environment |
| `golf/top.py` | the top of the backswing: the handover both halves share |
| `train.py` | PPO training |
| `find_top.py` | searching for the top to hand over at |

`GolfSwingSim` is composed from the `ik`, `posture` and `planner` mixins: three
separate concerns, one object, because they all work on the same model.

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

## Training

```bash
python train.py                      # full swing, 11 workers
python train.py --start transition   # downswing only, ~4x faster
python train.py --steps 20000000 --resume runs/golf/final.zip
```

Or train the two halves separately, either side of a fixed handover:

```bash
python find_top.py                                     # choose the top
python train.py --stage downswing --out runs/downswing
python train.py --stage backswing --out runs/backswing
```

Then watch what it learned:

```bash
python main.py --policy runs/golf                      # 3D viewer
python main.py --policy runs/golf --report --episodes 50
python main.py --policy runs/golf --frames swing.png   # contact sheet
python main.py --policy reference --report             # the scripted swing
python main.py --downswing runs/downswing --report     # the downswing alone
python main.py --backswing runs/backswing --downswing runs/downswing --report
```

Two policies were trained, differing only in whether the reward priced the
*curve* or only where the ball finished (40 swings each, stochastic):

| | scripted | offline only | **+ spin axis** |
| --- | --- | --- | --- |
| clubhead speed | 18.2 m/s | 35.8 m/s | **38.9 m/s** |
| ball speed | 20.3 m/s | 41.0 m/s | **47.0 m/s** |
| smash factor | 1.11 | 1.14 | **1.21** |
| spin axis | +36° | +46° | **+28°** |
| carry | 0.4 m | 112 m | **149 m** |
| offline | −0.3 m | +9.9 m | **−4.8 m** |
| contact | — | 36/40 | **40/40** |

The scripted swing chops 29° down and drills the ball into the turf; both
agents learned to hit up on it. The difference between the two columns is
entirely the reward — see below.

The agent's action is a **residual on the scripted swing**: an action of zero
reproduces a swing that already contacts the ball, so there is a reward
gradient from the first episode. A policy over 36 raw joint targets would
flail and never make contact.

The reward comes from `golf/launch.py`, not from MuJoCo's contact. Rigid-body
contact gives a smash factor of 0.8 against a real 1.48, and cannot produce
gear effect at all — so "how far" would be wrong and "how straight" would be
unrepresentable. Instead the clubhead's state at closest approach goes through
an oblique-impact model and a ballistic flight model with drag and Magnus lift.
Clubhead/ball collision is switched off during training so the wrong impulse
doesn't perturb the swing.

Measured on an i7-1265U (2 P-cores + 8 E-cores, so scaling is limited):

| | |
|---|---|
| throughput, 11 workers | ~750 agent steps/s end to end |
| full swing episode | 120 steps @ 100 Hz control |
| downswing episode | 30 steps |
| 10M steps, full swing | ~3.7 hours |

PPO, not MuJoCo, is the bottleneck. Two settings matter a lot and are easy to
get wrong: `cone="pyramidal"` in the model (elliptic costs 2.4× per step for an
identical swing), and `log_std_init=-1.0` in the policy (SB3's default σ=1.0
drops the initial contact rate from ~29% to ~2%).

## What the reward measures is what you get

The first policy plateaued at ~115 m and would not move. The cause was a gap in
the reward, not a limit of the agent: `offline` measures where the ball
*finishes*, so the agent satisfied it by aiming left and slicing back. Net
offline near zero — and a spin axis stuck at 46° with a smash factor of 1.14
against a possible 1.48. That oblique strike is what capped distance.

Adding a term that prices the *curve itself* rather than the finishing position
broke the plateau within 450k steps: axis 46° → 30°, smash 1.14 → 1.28, carry
112 → 149 m, at essentially the same clubhead speed. Nothing about the physics
or the agent changed. Only the question being asked.

Two details in that term matter:

* **It must be capped.** Uncapped, a wild enough strike scores worse than never
  touching the ball, and the agent learns to avoid it.
* **`axis_ref` is 60°, not 30°.** A tighter reference saturates the penalty
  below the 46° the policy actually sat at, leaving no gradient exactly where
  it was needed.

The same lesson bit twice. An earlier weighting charged per-step torque costs
that accumulated to ~4.8 over an episode while the carry being achieved paid
0.07 — so the agent halved its clubhead speed to bunt the ball, and reward rose
the whole time. `RewardWeights` carries both stories in its docstring. There is
a ranking check in the repo history worth re-running whenever the weights
change: a complete miss must score below every strike, and the ranking must be
monotonic in both squareness and speed.

It bit a third time, on backspin. The flight model caps lift but holds
`DRAG_COEFF` constant, so a ball at 4600 rpm balloons in reality and costs
nothing here — and the downswing duly settled at 4640 rpm, 7-iron spin off a
driver, because nothing said otherwise. The penalty that fixed it is standing
in for physics `golf/launch.py` does not have, not for taste. Re-running the
ranking check caught the cap mis-sized on the first attempt: at `spin_cap`
1.25, a wild strike (40° axis, 6000 rpm, 120 m) scored +1.32 against +1.27 for
a 10 cm miss — 0.05 of margin between "hit it badly" and "don't hit it". The
fix was to tighten the *cap* rather than the rate, which bounds the worst case
without touching the gradient at the 4640 rpm the swing actually produced.

## Splitting the swing at the top

The two halves can be trained separately, which only means anything if they
agree on where one ends and the other begins. That agreement is `golf/top.py`:
one `TopState` in `runs/top.npz` that the downswing resets to and the backswing
is paid for reaching. Both stages re-plan the reference around it, so "the top"
means the same thing on both sides; change it and both policies are stale.

The top is searched, not written down — ten parameters (pelvis, lumbar and
thorax turn, trunk bend, lead arm) optimised by CEM against the same carry and
straightness the downswing is paid for. It chose **less coil and more arm**:
shoulder turn −100° → −86°, lead-arm abduction −34° → −40°, shoulder rotation
40° → 15°. The scripted swing from that top carries 53 m against 28 m from the
hand-written one.

The downswing is where the reward varies, and it went a long way:

| | full swing, one policy | **downswing from the searched top** |
| --- | --- | --- |
| clubhead speed | 38.9 m/s | **51.2 m/s** |
| ball speed | 47.0 m/s | **72.8 m/s** |
| smash factor | 1.21 | **1.42** |
| backspin | — | **2990 rpm** |
| spin axis | +28° | **−0.9°** |
| carry | 149 m | **244 m** |
| contact | 40/40 | 39/40 |

Not a like-for-like comparison: this policy is handed a good top for free.

The backspin penalty did its work through the mechanism it was supposed to.
The swing that spun 4640 rpm came in at −1.2° of attack; the one that spins
2990 comes in at **+3.4°** — it learned to hit *up* on a teed driver, which is
where the spin went. Club path tightened from +9.1° to +3.2° on the way, and
carry rose rather than fell, so the distance-for-spin trade the reward allowed
for (up to 42 m) never had to be paid.

### The handover is about momentum, not position

Composing the two halves is where the interesting failure lives. The first
trained backswing arrived at 2.3° and 3.1 cm — *better* clubhead accuracy than
the reference's 3.3 cm — and made the composed swing **worse than not training
a backswing at all**: 199 m and 17/30 against the reference backswing's 235 m
and 29/30.

Injecting each kind of error into the canonical top separates the cause:

| error injected at the top | contact | spin axis | carry |
| --- | --- | --- | --- |
| none | 29/30 | −0.5° | 246 m |
| pose, 2.3° RMS | 29/30 | −1.1° | 239 m |
| **velocity, 2.0 rad/s RMS** | **18/30** | **+23°** | **190 m** |

Pose error is nearly free; velocity error is ruinous. The handover reward had
it exactly backwards — `top_still` 2.0 at `vel_ref` 4.0 priced 2.1 rad/s at
about 1.0, against 10.0 available for pose and clubhead. The policy bought
0.2 cm of club accuracy and paid 2 rad/s of momentum for it. It was trained to
do that.

Two things follow, both now in `RewardWeights`:

* **Weight the handover by what it costs the shot.** Velocity is ~8× more
  expensive than pose, so it is weighted that way (`top_still` 8.0 against
  `top_pose` 3.0), not by which is easier to measure.
* **An arrival test that ignores velocity will lie to you.** `top_reached`
  originally checked 8° and 20 cm and reported 100% arrival while the composed
  swing was losing 35 m and half its strikes.

### Don't train the backswing

Reweighted and retrained for another 5M steps, the backswing moved the way it
was told — velocity error 2.11 → 1.69 rad/s, paid for out of pose — and was
still comprehensively beaten by not training it at all:

| backswing | angle | club | velocity | composed carry | contact |
| --- | --- | --- | --- | --- | --- |
| **scripted (zero action)** | **0.35°** | **3.3 cm** | **0.17 rad/s** | **244 m** | **28/30** |
| trained, deterministic | 4.56° | 11.4 cm | 0.57 rad/s | 202 m | 27/30 |
| trained, stochastic | 4.79° | 6.7 cm | 1.69 rad/s | 206 m | 23/30 |

The reason is structural, and worth knowing before repeating the experiment.
The optimum of this task *is* the zero action, which the reference already
supplies for free, so the best a policy can do is learn to stop interfering —
and after 5M steps it still emits a mean action of 0.29, about 6° of joint
target. Terminal-only reward over 82-step episodes is ~61k episodes in 5M
steps, nowhere near enough to drive a 36-dimensional policy onto exactly zero,
and `ent_coef` is actively paying it not to. The one thing a trained backswing
could add over the script is disturbance rejection, and the position servos
already do that themselves: 15° and 2 rad/s of perturbation at address still
arrives 0.79° and 12.6 cm from the top.

So the swing to use is the scripted backswing with the trained downswing:

```bash
python main.py --backswing reference --downswing runs/downswing --report
```

The split earns its keep entirely on the downswing side. Being able to *start*
from a good top is what made the downswing trainable; learning to *reach* one
was not worth the compute.

## Where it stopped

Full swing, scripted backswing into the trained downswing, 40 stochastic
swings: **51.2 m/s, smash 1.42, spin axis −2.5°, carry 242.6 m, 39/40**, off a
handover that arrives 0.4° and 3 cm from the canonical top. A real driver is
45 m/s, smash 1.48, ~2500 rpm, 220 m. The kinematics hold up too:
+3.4° attack, +3.2° club path, hips 39° open against shoulders square, and 55°
of wrist cock still held into impact. The remaining gap is not distance any
more, it is **dispersion**: offline scatters ±43 m even with the mean within
3 m of straight, and the reward prices the average shot with nothing paying for
consistency.

## Next

* Price dispersion, not just the mean shot — the swing is long and straight on
  average and wild individually
* Train the downswing against a realistic handover (larger `--vel-jitter`)
  rather than only the exact canonical top
* Raise `--residual` from 0.35 to ~0.6 so the policy can leave the reference's
  swing plane
* Revisit `--base free` (full balance) once the swing itself is good
