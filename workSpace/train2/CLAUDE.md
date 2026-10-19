# Go2 零速度指令站定 — 任务说明（压缩版）

工程根目录：`/home/ubuntu/B4_2025/unitree_rl_gym`
详细流水账：本目录 `zero_cmd_worklog.txt`

## 0. 目标与硬约束

目标：`commands=0` 时站定。基线：Isaac Gym 20 s 漂移 0.21 m（1.0 cm/s）；
MuJoCo 0.52 m（2.6 cm/s）；`rew_tracking_lin_vel`=0.8337。

只改**训练奖励**。不得改：观测（47 维 / `num_obs` / `observation_layout`）、
`commands.ranges`（±1.0）、PD、`action_scale`、网络结构。
奖励改动必须重训 + 用 `play.py` 重新导出 `policy_1.pt`；旧 checkpoint 不能证明新奖励有效。

环境（每个新 shell）：
```bash
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
export PATH=/home/ubuntu/miniconda3/envs/unitree-rl/bin:$PATH
```
脚本里 `import isaacgym` 必须在 `import torch` 之前。

## 1. 根因

1. `tracking_lin_vel=exp(-v²/0.25)` 在 0 附近太平：v=0.026 m/s 只损失 0.27%，梯度趋 0。
2. 零线指令占比 ~3.1%（半径 0.2 圆 / 边长 2 正方形）；线速度+偏航都近 0 仅 **0.38%**。
3. **原地踏步主因**：`_reward_contact`（权重 0.18）按固定相位钟 `leg_phase`
   奖励 trot 触地模式，**与指令无关**；零指令下 trot 每步最多 +0.0144，
   `_reward_feet_swing_height`（-20）又把摆动腿塑造成 0.08 m 抬腿。
   所以狗在零指令下仍有明确正收益去原地踏步。
4. 结论：既要在零指令时**惩罚残余运动**，也要在零指令时**关闭步态奖励**。

## 2. 奖励设计（权重均乘 dt=0.02）

| 项 | 门控 | 惩罚 | 权重 |
|---|---|---|---|
| A `zero_cmd_lin_vel` | `norm(cmd[:,:2])<0.2` | `norm(base_lin_vel[:,:2])`（L1，不平方） | -2.0（-1~-5） |
| C `zero_cmd_dof_vel` | 线速度<0.2 **且** `|wz|<0.1` | `sum(dof_vel²)` | -5e-3（-5e-3~-2e-2） |
| G 步态门控 | 同 C | `contact`、`feet_swing_height` 乘以 `1-zero_mask` | 沿用原权重 |

- 偏航门控用 0.1：原地转向本来就要迈步，不能惩罚/关步态奖励。
- 一次只加一个变量，便于 A/B。备选：`stand_still`（dof_pos 偏离默认，同门控，-0.2 起）、
  `zero_cmd_action=-0.02`、方案 D（强制提高零指令采样比例）。

## 3. 实施步骤

0. 备份：策略 `logs/rough_go2/exported/policies/policy_before_zero_cmd.pt`；
   源码 `agentWorkSpace/backups/go2_env.py.before_zero_cmd`、`..._config...`。
1. `go2_env.py` 加 `_reward_<key>`（名字必须与配置 key 对应）。
2. `go2_config.py` 的 `rewards.scales` 加非零权重；有权重无函数 → 启动 AttributeError。
3. 冒烟：`python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=2`
   检查无 AttributeError、`Episode/rew_*` 出现且非零、无 NaN。
   **跑完删除该冒烟日志目录**（`play.py` 取排序最后一个 run，这是踩过的坑）。
4. **完整训练**：`--experiment_name=go2_2 --run_name=go2_2 --max_iterations=3000`。
   **！！到这一步停止并报告，训练与结果反馈由用户执行。
   用户明确表示可以进入下一阶段后，才继续。！！**
   4.1 用户反馈效果不佳 → 回第 2 节加下一项奖励 → 重跑第 4 步 → 等用户验收；
   循环直到用户明确表示通过。
5. 看 TensorBoard：`tracking_lin_vel>=0.7`、零指令项为小负数、
   `mean_episode_length` 接近 1002、`orientation/base_height` 不恶化。
   掉 tracking / episode 变短 → 回第 2 步减权重。
6. Isaac Gym 独立脚本测漂移：`num_envs=1`、无噪声/摩擦随机化/推力、`manual=True`、
   `heading_command=False`、确定性动作（策略均值）、20 s 内 reset 前记录、
   多 seed 取均值。目标 <0.05 m 优秀 / <0.10 m 可接受。
7. MuJoCo：`python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml --headless --duration 20`，
   比较 t=0 与 t=20 的 (x,y)。目标 <0.10 m 优秀 / <0.20 m 可接受。
8. 调权重：漂移大→A 加到 -5；行走变差→减半；仍踏步→C 加到 -1e-2~-2e-2，
   或加 `zero_cmd_action=-0.02`；被推倒变多→减小权重。
9. 回归：确认方向键前进/后退/平移/转向正常，g1/h1/h1_2 不受影响。
10. 更新 `zero_cmd_worklog.txt` 与本文件。

## 4. 坑

1. 门控阈值 0.2 必须与 `_resample_commands()` 的清零阈值一致，改就一起改。
2. 不要全局惩罚位置/速度，必须门控，否则毁掉正常行走。
3. `heading_command=True` 时 `commands[:,2]` 每步由朝向误差重算，常非 0；
   线速度门控安全，偏航门控必须放宽到 `|wz|<0.1`。
4. 权重先小后大；一次加太猛会让狗不敢动，且梯度会泛化到非零指令。
5. `only_positive_rewards` 裁剪：0.026 m/s 时 -2.0 每步仅 -0.00104，不触发。
6. 完全零指令样本只有 0.38%；指标长期不动 → 考虑方案 D。
7. 训练每 15 s 推一次机器人，被推瞬间惩罚变大是正常的；评测无推力。
8. 奖励与观测不要混改，否则无法定位变量。
9. 冒烟目录必须删，否则 `play.py` 加载错 run。
10. 导出的是 Actor；Critic 不部署。
11. 奖励函数内禁止 `.item()`/Python 循环，保持张量并行（`_reward_contact` 的 4 次循环是既有写法）。
12. `self.actions` 约 ±1，action 惩罚 -0.02 起步。
13. 1500 轮约 12 分钟；明显变慢先查是否写了下标循环。
14. 换种子/换 `num_envs`/换观测/换奖励都不可直接比数字；A/B 只改一个变量。

## 5. 验收

必达：`tracking_lin_vel>=0.7`；`mean_episode_length` 不明显低于 1002；
MuJoCo 方向键正常；g1/h1 不受影响。
目标：Isaac <0.05 m 优秀 / <0.10 m 可接受；MuJoCo <0.10 m 优秀 / <0.20 m 可接受。
不追求绝对 0：零指令下保持平衡本来就需要微小调整，目标是肉眼看不出来。

## 6. 已完成工作（2026-10-18）

- 备份完成：`policy_before_zero_cmd.pt`、`agentWorkSpace/backups/go2_env.py.before_zero_cmd`、
  `go2_config.py.before_zero_cmd`。
- `go2_env.py` 新增 `_reward_zero_cmd_lin_vel`、`_reward_zero_cmd_dof_vel`、`_zero_cmd_mask`；
  `_reward_contact`、`_reward_feet_swing_height` 加零指令门控。
- `go2_config.py` 新增 `zero_cmd_lin_vel=-2.0`、`zero_cmd_dof_vel=-5e-3`。
- 第 1 轮（方案 A，3000 轮，`logs/go2_2/Oct18_11-11-05_go2_2`）：
  tracking 0.8407、zero_cmd_lin_vel -0.0061、episode_length 1002.0、mean_reward 30.53；
  用户实测：仍原地踏步。
- 第 2 轮（方案 C）代码与冒烟已通过；`logs/go2_2` 下未见第二轮完整训练记录。
- 诊断（4096 env × 48 步随机动作）：零线指令 3.64%，线+偏航双门控 0.38%，
  门控命中样本充足；contact 门控在零指令环境上泄漏 0，验证正确。
- 第 3 轮（本次）：定位并修复踏步根因——零指令时关闭 `contact`/`feet_swing_height`；
  2 轮冒烟通过（无报错、无 NaN），冒烟目录已删除；**等待用户完整重训与反馈**。
