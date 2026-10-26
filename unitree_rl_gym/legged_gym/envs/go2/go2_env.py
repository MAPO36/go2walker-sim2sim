
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class GO2Robot(LeggedRobot):
    """ Go2 (12 DoF quadruped) environment.

        Differences vs. the base LeggedRobot / the g1+h1 subclasses:
          * observation is 47-dim: base_lin_vel is REMOVED (a real Go2 has no
            state estimator for body linear velocity) and sin/cos phase is appended
          * the base class has no feet_pos / feet_vel / phase, so _init_foot() and
            the phase computation are added here
    """

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # layout (47): ang_vel(3) gravity(3) commands(3) dof_pos(12) dof_vel(12) actions(12) sin cos
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        # Trot gait: the legs of one diagonal pair swing together.
        # self.feet_indices follows the URDF body order -> FL, FR, RL, RR,
        # so the phase vector is [FL, FR, RL, RR] = [phase, phase+0.5, phase+0.5, phase].
        # (g1/h1 only have 2 legs, hence a hard-coded 2-column leg_phase there.)
        self.leg_phase = torch.stack([self.phase,
                                      (self.phase + offset) % 1,
                                      (self.phase + offset) % 1,
                                      self.phase], dim=-1)

        return super()._post_physics_step_callback()


    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _zero_cmd_mask(self):
        """完全零指令的掩码：线速度和偏航都接近 0。

        原地转向本来就需要迈步，所以偏航也接近 0 才算“站定”。
        """
        return (
            (torch.norm(self.commands[:, :2], dim=1) < 0.2)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        ).float()

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        # 原地踏步的根源：contact 奖励按固定相位钟表鼓励 trot，和指令无关。
        # 零指令时把它关掉，狗才没有理由继续抬腿。
        return res * (1.0 - self._zero_cmd_mask())

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        # 同理：零指令时不再鼓励“抬到 0.08 m”的摆动腿形状。
        return torch.sum(pos_error, dim=(1)) * (1.0 - self._zero_cmd_mask())

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_hip_pos(self):
        # Go2 DoF order (go2.urdf): FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
        #                           RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        # -> the hip joints are 0, 3, 6, 9 (NOT 1, 2, 7, 8, which is the g1 hip roll/yaw pair)
        return torch.sum(torch.square(self.dof_pos[:,[0,3,6,9]]), dim=1)
        
    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0 or getattr(self.cfg.commands, "manual", False):
            return

        p = 0.3                     # 想多少比例就改这个数，0.3 = 30%
        pick = torch.rand(len(env_ids), device=self.device) < p
        zero_ids = env_ids[pick]
        if len(zero_ids) == 0:
            return

        self.commands[zero_ids] = 0.0
        # heading_command=True 时，wz 每步会按“目标朝向 - 当前朝向”重算。
        # 只清 0 不够，必须把目标朝向设成当前朝向，wz 才会真的是 0。
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[zero_ids], self.forward_vec[zero_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_ids, 3] = heading

    def _reward_zero_cmd_lin_vel(self):
        """0 线速度指令时，惩罚机身的水平速度。

        用一次范数而不是平方：平方项在 v→0 时梯度也→0，
        压不住最后那 1~3 cm/s 的残余漂移；一次范数在 0 附近梯度是常数。
        门控用 0.2，和 _resample_commands() 里的清零阈值一致。
        """
        is_zero_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.2).float()
        return torch.norm(self.base_lin_vel[:, :2], dim=1) * is_zero_cmd

    def _reward_zero_cmd_dof_pos(self):
        """0 指令（线速度和偏航都接近 0）时，惩罚关节偏离默认姿态的角度。

        替代原来的 zero_cmd_dof_vel（惩罚关节速度）：速度小不代表姿态稳，
        狗可以用很慢的关节运动把身体一点点挪走。这里直接罚姿态偏差。

        用平方：偏差很小时梯度也接近 0，但这是有意保留的——站定需要微小的
        关节调整来保持平衡，惩罚太陡会让狗僵住、被推倒。权重从 -1.0 起步。
        偏航门控用 0.1：原地转向本来就需要迈步，不能惩罚。
        """
        is_zero_cmd = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.2)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        ).float()
        
        err = torch.clamp(self.dof_pos - self.default_dof_pos, -0.5, 0.5)
        return torch.sum(torch.square(err), dim=1) * is_zero_cmd
