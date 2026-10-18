"""Headless check of the keyboard command path.

Reproduces what `play.py` does with `commands.manual = True`, but writes `env.keys`
directly -- that dict is exactly what the viewer callback fills in real use
(base_task.py: `self.keys[evt.action] = (evt.value > 0)`), so this exercises the
same code path without needing a GUI.

Run from anywhere:

    PYTHONPATH=/home/ubuntu/B4_2025/unitree_rl_gym \
    LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/unitree-rl/lib \
    PATH=/home/ubuntu/miniconda3/envs/unitree-rl/bin:$PATH \
    python verify_manual.py --policy <policy_1.pt>
"""
import sys
import argparse

import isaacgym  # noqa: F401  (must be imported before torch and legged_gym)
import torch
# same import order as legged_gym/scripts/train.py: envs first, then utils
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def build_env(num_envs):
    # get_args() parses sys.argv with gymutil, so hand it a clean argument list.
    sys.argv = ['verify_manual.py', '--task=go2', '--headless',
                '--num_envs=%d' % num_envs]
    args = get_args()

    env_cfg, _ = task_registry.get_cfgs('go2')
    # same overrides play.py applies
    env_cfg.commands.manual = True
    env_cfg.commands.manual_env_id = 0
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e9
    env_cfg.viewer.ref_env = 0
    # keep the run deterministic-ish and do not sleep
    env_cfg.env.test = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False

    env, _ = task_registry.make_env(name='go2', args=args, env_cfg=env_cfg)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--num_envs', type=int, default=4)
    opts = ap.parse_args()

    env = build_env(opts.num_envs)
    policy = torch.jit.load(opts.policy, map_location=env.device)
    policy.eval()

    obs = env.get_observations()

    def run(keys, steps):
        """Hold `keys` pressed for `steps` policy steps. Returns forward travel [m]."""
        env.keys.clear()
        env.keys.update(keys)
        nonlocal obs
        forward = 0.0
        with torch.no_grad():
            for _ in range(steps):
                actions = policy(obs.detach())
                obs, _, _, _, _ = env.step(actions.detach())
                forward += float(env.base_lin_vel[0, 0]) * env.dt
        return forward

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print('%-46s %s   %s' % (name, 'PASS' if ok else 'FAIL', detail))

    # --- 0. warm up: no keys, robot stands -------------------------------
    run({}, 50)
    check('idle: commands[0] zero', bool(torch.all(env.commands[0, :3] == 0)),
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())

    # --- 1. W pressed ----------------------------------------------------
    fwd = run({'W': True}, 150)          # 150 * 0.02 s = 3 s
    cmd = env.commands[0, :3].tolist()
    vx = float(env.base_lin_vel[0, 0])
    check('W: commands[0,0] == max_cmd[0]', abs(cmd[0] - 0.8) < 1e-6,
          'commands[0,:3]=%s' % cmd)
    check('W: commands[1,1] stays zero', bool(torch.all(env.commands[1, :3] == 0)),
          'only env 0 is keyboard driven')
    check('W: body vx > 0.3 m/s', vx > 0.3, 'base_lin_vel[0,0]=%.3f m/s' % vx)
    check('W: robot travelled forward', fwd > 0.5, 'integral vx = %.3f m over 3 s' % fwd)

    # --- 2. release ------------------------------------------------------
    run({}, 100)
    cmd = env.commands[0, :3].tolist()
    check('release: commands back to zero', all(abs(c) < 1e-6 for c in cmd),
          'commands[0,:3]=%s' % cmd)

    # --- 3. strafe + yaw -------------------------------------------------
    run({'A': True}, 5)
    check('A: commands[0,1] == +max_cmd[1]', abs(float(env.commands[0, 1]) - 0.5) < 1e-6,
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())
    run({'D': True}, 5)
    check('D: commands[0,1] == -max_cmd[1]', abs(float(env.commands[0, 1]) + 0.5) < 1e-6,
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())
    run({'Q': True}, 5)
    check('Q: commands[0,2] == +max_cmd[2]', abs(float(env.commands[0, 2]) - 1.0) < 1e-6,
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())
    run({'E': True}, 5)
    check('E: commands[0,2] == -max_cmd[2]', abs(float(env.commands[0, 2]) + 1.0) < 1e-6,
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())
    run({'S': True}, 5)
    check('S: commands[0,0] == -max_cmd[0]', abs(float(env.commands[0, 0]) + 0.8) < 1e-6,
          'commands[0,:3]=%s' % env.commands[0, :3].tolist())

    print('\n%d/%d checks passed' % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
