import contextlib
import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def quat_rotate_inverse(quaternion, vector):
    """Rotate a world-frame vector into the base frame, as legged_gym's math helper does.

    Used for the base linear velocity: MuJoCo reports the free-joint velocity in
    the world frame, while the policy was trained on base_lin_vel, which Isaac Gym
    expresses in the base frame.
    """
    qw = quaternion[0]
    qvec = np.asarray(quaternion[1:4])
    return np.asarray(vector, dtype=np.float64) + 2.0 * np.cross(
        qvec, np.cross(qvec, vector) - qw * vector
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_observation(config, action, cmd, qj, dqj, quat, omega, base_lin_vel, sim_time):
    """Rebuild the policy observation exactly as the matching legged_gym env does.

    This repo trains two different observation layouts, so the config file has to
    say which one the policy was exported from:

      "humanoid" (default -- g1, h1, h1_2; see legged_gym/envs/g1/g1_env.py):
          [ang_vel, gravity, cmd, dof_pos, dof_vel, actions, sin_phase, cos_phase]
          -> 11 + 3 * num_actions values (47 for a 12-dof robot)

      "quadruped" (go2; carries no phase input but does observe base linear
      velocity -- see legged_gym/envs/base/legged_robot.py, which GO2RoughCfg
      inherits unmodified):
          [lin_vel, ang_vel, gravity, cmd, dof_pos, dof_vel, actions]
          -> 12 + 3 * num_actions values (48 for a 12-dof robot)

    The two are not interchangeable: the quadruped layout starts with three extra
    values, so feeding one to the other shifts every slot.
    """
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    qj = (qj - default_angles) * config["dof_pos_scale"]
    dqj = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega = omega * config["ang_vel_scale"]

    common = [omega, gravity_orientation, cmd * cmd_scale, qj, dqj, action]

    layout = config.get("observation_layout", "humanoid")
    if layout == "quadruped":
        observation_parts = [base_lin_vel * config["lin_vel_scale"]] + common
    else:
        period = 0.8
        phase = sim_time % period / period
        phase_observation = np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        )
        observation_parts = common + [phase_observation]

    obs = np.concatenate(observation_parts).astype(np.float32)
    if obs.size != config["num_obs"]:
        raise RuntimeError(
            f"num_obs={config['num_obs']} in the config file, but the "
            f"{layout!r} layout builds {obs.size} values -- fix num_obs"
        )
    clip_observations = config.get("clip_observations", 100.0)
    return np.clip(obs, -clip_observations, clip_observations)


def load_config(config_file):
    with open(
        f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r"
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in ("policy_path", "xml_path"):
        config[key] = config[key].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    return config


def run(config, headless=False, duration_override=None):
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    simulation_duration = (
        duration_override if duration_override is not None else config["simulation_duration"]
    )
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    action_scale = config["action_scale"]

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    clip_actions = config.get("clip_actions", 100.0)

    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # The deploy script indexes the state as [free joint, leg joints...], so the
    # model has to actually have that layout.
    if m.nq != 7 + num_actions or m.nv != 6 + num_actions or m.nu != num_actions:
        raise RuntimeError(
            f"MuJoCo model has nq={m.nq}, nv={m.nv}, nu={m.nu}; expected "
            f"nq={7 + num_actions}, nv={6 + num_actions}, nu={num_actions} for a "
            f"floating-base robot with {num_actions} actuated joints"
        )

    # Start from the pose the policy was trained to be reset to: base at the
    # configured height and every joint at its default angle. MuJoCo otherwise
    # starts hinges at 0 rad, which for go2 is outside the knee range, so the
    # first control steps would be a violent transient instead of the trained
    # initial state. Index order of the URDF (and of the trained policy) is
    # FL, FR, RL, RR x hip, thigh, calf -- matching default_angles.
    if "base_position" in config:
        d.qpos[0:3] = np.array(config["base_position"], dtype=np.float64)
    d.qpos[7 : 7 + num_actions] = default_angles
    mujoco.mj_forward(m, d)

    # load policy
    policy = torch.jit.load(policy_path)

    viewer_context = (
        contextlib.nullcontext(None)
        if headless
        else mujoco.viewer.launch_passive(m, d)
    )
    with viewer_context as viewer:
        # Close the viewer automatically after simulation_duration simulated seconds.
        next_status_time = 0.0
        while d.time < simulation_duration and (viewer is None or viewer.is_running()):
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()
                quat = d.qpos[3:7].copy()
                omega = d.qvel[3:6].copy()
                base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])

                obs = build_observation(
                    config,
                    action,
                    cmd,
                    qj,
                    dqj,
                    quat,
                    omega,
                    base_lin_vel,
                    counter * simulation_dt,
                )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -clip_actions, clip_actions)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            if headless:
                # No viewer to pace against: run as fast as the CPU allows, but
                # report the base state once per simulated second so a headless
                # run can still be checked.
                if d.time >= next_status_time:
                    next_status_time += 1.0
                    forward_velocity = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])[0]
                    print(
                        f"t={d.time:5.2f}s  base=[{d.qpos[0]:+.3f} {d.qpos[1]:+.3f} "
                        f"{d.qpos[2]:+.3f}]  vx={forward_velocity:+.3f} m/s",
                        flush=True,
                    )
                continue

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return d


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument(
        "--headless", action="store_true", help="run without opening a viewer window"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override simulation_duration, in simulated seconds",
    )
    args = parser.parse_args()
    run(load_config(args.config_file), headless=args.headless, duration_override=args.duration)
