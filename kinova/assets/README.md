# Kinova Gen3 7-DoF — vendored from MuJoCo Menagerie

The MJCF template + meshes here are derived from the **MuJoCo Menagerie** Kinova Gen3 model.

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3
- Vendored at commit: `a09bd169ca2571361e09e465cd68b0c496115c3f`
- License: BSD-3-Clause (see `LICENSE`).

## Modifications relative to upstream

1. The single-file MJCF is templated as `gen3.xml.template` with Python-format placeholders for the 28 domain-randomization parameters consumed by `src/neural_controller/dynamics/kinova_gen3.py`:
   - `{m1}..{m7}` — link masses
   - `{b1}..{b7}` — per-joint viscous damping
   - `{mu1}..{mu7}` — per-joint Coulomb (dry) friction (`frictionloss`)
   - `{payload_mass}, {payload_cx}, {payload_cy}, {payload_cz}` — end-effector payload mass + COM offset in tool frame
   - `{payload_Ixx}, {payload_Iyy}, {payload_Izz}` — diagonal principal moments of the payload, with `iquat = identity` (principal frame == tool frame)
   - `{dt}` — MJX integrator timestep
   - `{gravity}` — gravity magnitude (m/s²)
2. The `<actuator>` block was rewritten from position-controlled `<position>` actuators (with internal kp/kv) to torque-controlled `<motor>` actuators with the same `forcerange` envelope (`±105` N·m large joints 1–4, `±52` N·m small joints 5–7). FDDP and the residual controller both produce torques directly.
3. A fixed `tool_frame` body was added as a child of `bracelet_link` at the original `pinch_site` location (`pos="0 0 -0.061525" quat="0 1 0 0"`). Its full inertial (mass + COM offset + diagonal inertia) is templated so we can sample arbitrary end-effector payloads at training time.
4. The `<keyframe>` block was removed (training/eval set initial state programmatically).
5. The `wrist` camera and the `pinch_site` outside the tool body were removed.
6. Joint defaults explicitly set `damping="0" frictionloss="0"` so MJX always allocates `dof_damping` and `dof_frictionloss` arrays of length 7 — required for `override_mjx_params` to write per-rollout DR samples into them.

## Body indexing in the compiled MJX model

| body_id | name | role | DR? |
|---|---|---|---|
| 0 | `world` | static | — |
| 1 | `base_link` | fixed mount | no (pre-arm; not actuated) |
| 2 | `shoulder_link` | joint_1 | mass `m1`, damping `b1`, friction `mu1` |
| 3 | `half_arm_1_link` | joint_2 | mass `m2`, damping `b2`, friction `mu2` |
| 4 | `half_arm_2_link` | joint_3 | mass `m3`, damping `b3`, friction `mu3` |
| 5 | `forearm_link` | joint_4 | mass `m4`, damping `b4`, friction `mu4` |
| 6 | `spherical_wrist_1_link` | joint_5 | mass `m5`, damping `b5`, friction `mu5` |
| 7 | `spherical_wrist_2_link` | joint_6 | mass `m6`, damping `b6`, friction `mu6` |
| 8 | `bracelet_link` | joint_7 | mass `m7`, damping `b7`, friction `mu7` |
| 9 | `tool_frame` | payload host (fixed to bracelet) | full rigid body (mass + COM + diaginertia) |

`override_mjx_params` writes:
- `body_mass[2..8]` ← `m1..m7`
- `body_mass[9], body_ipos[9], body_inertia[9]` ← payload params
- `dof_damping[0..6]` ← `b1..b7`
- `dof_frictionloss[0..6]` ← `mu1..mu7`

## Known simplifications

- DR scales **link mass scalars only**; the per-link inertia tensor stays at the Menagerie nominal. This matches the existing pendulum/double-pendulum behaviour — geometry is fixed, mass scales independently of the inertia tensor.
- Coulomb friction is set to zero in the **trajectory-optimization nominal** (FDDP is gradient-based; non-smooth `frictionloss` would destabilize it). Friction is applied only inside the differentiable training rollout the residual sees.
