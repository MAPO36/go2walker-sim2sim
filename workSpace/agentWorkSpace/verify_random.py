"""Task 7 step 2: with the keyboard OFF, does the policy still walk?

Uses play.py's own env overrides except `commands.manual = False`, so the random
command resampler runs and we can measure whether the robot tracks it.
Headless, so the result is numbers instead of eyeballing the viewer.
"""
import sys

import isaacgym  # noqa: F401  (before torch / legged_gym)
import torch
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    sys.argv = ['verify_random.py', '--task=go2', '--headless', '--num_envs=1']
    args = get_args()

    env_cfg, _ = task_registry.get_cfgs('go2')
    # play.py overrides, minus the keyboard
    env_cfg.commands.manual = False
    env_cfg.commands.heading_command = False
    env_cfg.env.test = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False

    env, _ = task_registry.make_env(name='go2', args=args, env_cfg=env_cfg)
    policy = torch.jit.load(
        '/home/ubuntu/B4_2025/unitree_rl_gym/logs/rough_go2/exported/policies/policy_1.pt',
        map_location=env.device)
    policy.eval()

    obs = env.get_observations()
    steps = 3000  # 60 s of simulated time
    n_reset = 0
    sum_abs_v = 0.0
    sum_abs_cmd = 0.0
    sum_track = 0.0
    n_moving = 0

    with torch.no_grad():
        for _ in range(steps):
            prev_len = int(env.episode_length_buf[0])
            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())
            if int(env.episode_length_buf[0]) < prev_len:
                n_reset += 1

            cmd = env.commands[0, :2]
            vel = env.base_lin_vel[0, :2]
            err = torch.sum(torch.square(cmd - vel))
            sum_track += float(torch.exp(-err / 0.25))
            sum_abs_v += float(torch.norm(vel))
            sum_abs_cmd += float(torch.norm(cmd))
            if float(torch.norm(cmd)) > 0.2:
                n_moving += 1

    print('simulated %.1f s (%d steps)' % (steps * env.dt, steps))
    print('command was non-zero on %d/%d steps' % (n_moving, steps))
    print('mean |cmd|  = %.3f m/s' % (sum_abs_cmd / steps))
    print('mean |vel|  = %.3f m/s' % (sum_abs_v / steps))
    print('mean tracking_lin_vel = %.3f  (training converged at ~0.85)' % (sum_track / steps))
    print('falls / resets = %d' % n_reset)
    ok = n_reset <= 2 and sum_track / steps > 0.6 and sum_abs_v / steps > 0.15
    print('\n%s' % ('PASS: policy keeps walking under random commands' if ok
                    else 'FAIL: policy is not walking properly'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
