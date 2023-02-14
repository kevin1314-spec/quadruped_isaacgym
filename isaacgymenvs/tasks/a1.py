import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *
from .base.vec_task import VecTask

from typing import Tuple, Dict


class A1(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 27
        self.cfg["env"]["numActions"] = 12

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.height_points = self.init_height_points()
        self.measured_heights = None
        self.height_meas_scale = 5.0
        FL_phase = torch.tensor([1,1,1,0,0])
        FR_phase = torch.tensor([0,0,1,1,1])
        self.contact_phase = torch.stack((FL_phase,FR_phase,FR_phase,FL_phase),dim=1)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        #commands
        x_vel = torch.full((self.num_envs,),1.,device='cuda:0')
        y_vel = torch.full((self.num_envs,),0.,device='cuda:0')
        ang_vel = torch.full((self.num_envs,),0.,device='cuda:0')
        self.commands = torch.stack((x_vel,y_vel,ang_vel),dim=1)

        r=torch.full((self.num_envs,),0.8,device='cuda:0')
        p=torch.full((self.num_envs,),0.6,device='cuda:0')
        y=torch.full((self.num_envs,),0.6,device='cuda:0')
        self.euler_max = torch.stack((r,p,y),dim=1)
        self.ones = torch.full((self.num_envs,3),1.,device='cuda:0')
        self.zeros = torch.full((self.num_envs,3),0.,device='cuda:0')
        self.one = torch.full((self.num_envs,),1.,device='cuda:0')
        self.zero = torch.full((self.num_envs,),0.,device='cuda:0')
        self.horizontal_scale = 0.05 
        self.vertical_scale = 0.005 
        self.height_samples = torch.zeros((1000, 1000), dtype=torch.int32,device='cuda:0')
        
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contact_x = torch.full((self.num_envs,4),-0.006368238836675744,device=self.device)
        self.feet_gap = torch.zeros_like(self.last_contact_x)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        #self._create_trimesh()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """terrain_width = 70.
        terrain_length = 45. 
        horizontal_scale = 0.05 
        vertical_scale = 0.005 
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)
        
        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
 
        heightfield[0:1400, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.08, max_height=0.08, step=0.05, downsampled_scale=0.15).height_field_raw"""
        
        num_terrain = 8
        terrain_width = 5.
        terrain_length = 60.
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005   # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((120*num_terrain, num_cols), dtype=np.int16)
        
        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        def slope_sub_terrain(): return SubTerrain(width=20, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        
        heightfield[0:20, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[20:120, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[120:140, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[140:240, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[240:260, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[260:360, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[360:380, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[380:480, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[480:500, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[500:600, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[600:620, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[620:720, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[720:740, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[740:840, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        heightfield[840:860, :] = sloped_terrain(slope_sub_terrain(), slope=0.001).height_field_raw
        heightfield[860:960, :] = stairs_terrain(new_sub_terrain(), step_width=0.25, step_height=0.1).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = 10.5
        tm_params.dynamic_friction = 10.5
        tm_params.restitution = 0.0
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/a1_description/urdf/a1.urdf"

        #asset_path = os.path.join(asset_root, asset_file)
        #asset_root = os.path.dirname(asset_path)
        #asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_env_origins = np.zeros((8, 8, 3))
        self.terrain_levels = torch.randint(0, 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, 1, (self.num_envs,), device=self.device)
        self.terrain_origins = torch.from_numpy(self.terrain_env_origins).to(self.device).to(torch.float)
        spacing = 15.5

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        #extremity_name = "foot"
        #feet_names = [s for s in body_names if extremity_name in s]
        #self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "calf" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0
        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, 8)
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
            pos = self.env_origins[i].clone()
            #pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "a1", i, 1, 0)
            
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

        #for i in range(len(feet_names)):
        #   self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        x_1 = torch.mul(self.actions[:,0],0.12)
        y_1 = torch.add(torch.mul(self.actions[:,1],0.02),0.0838)
        z_1 = torch.add(torch.mul(self.actions[:,2],0.15),0.25)
        x_2 = torch.mul(self.actions[:,3],0.12)
        y_2 = torch.add(torch.mul(self.actions[:,4],0.02),0.0838)
        z_2 = torch.add(torch.mul(self.actions[:,5],0.15),0.25)
        x_3 = torch.mul(self.actions[:,6],0.12)
        y_3 = torch.add(torch.mul(self.actions[:,7],0.02),0.0838)
        z_3 = torch.add(torch.mul(self.actions[:,8],0.15),0.25)
        x_4 = torch.mul(self.actions[:,9],0.12)
        y_4 = torch.add(torch.mul(self.actions[:,10],0.02),0.0838)
        z_4 = torch.add(torch.mul(self.actions[:,11],0.15),0.25)
        
        FL_angle = ik(x_1,y_1,z_1,self.num_envs)
        FR_angle = ik(x_2,y_2,z_2,self.num_envs)
        RL_angle = ik(x_3, y_3, z_3,self.num_envs)
        RR_angle = ik(x_4, y_4, z_4,self.num_envs)
        
        self.targets = torch.cat((FL_angle,FR_angle,RL_angle,RR_angle),dim=1)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.targets))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        self.compute_observations()
        self.compute_reward()
        
        
        
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

    def compute_reward(self):
        # velocity tracking reward
        lin_x_vel_error = torch.square(torch.sub(self.commands[:,0] , self.base_lin_vel[:,0]))
        ang_vel_error = torch.square(torch.sub(self.commands[:, 2] , self.base_ang_vel[:, 2]))
        rew_forward = torch.mul((self.base_lin_vel[:,0]>0.25),(torch.abs(self.euler[:,2])<0.2))
        rew_lin_x_vel = torch.exp(-lin_x_vel_error/0.25)
        rew_ang = torch.mul(torch.exp(-ang_vel_error/0.25) , 0.005)
        
        
        """
        # torque penalty
        rew_torque = torch.mul(torch.sum(torch.square(self.torques), dim=1) , -0.0001)

        # joint acceleration and velocity
        rew_joint_acc = torch.mul(torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1), -0.00025)
        rew_joint_vel = torch.mul(torch.sum(torch.square(self.dof_vel), dim=1), -0.0001)"""
        # y-axis distance reward
        deviation_error = torch.abs(self.root_states[:,1]-8)
        rew_deviation = torch.mul(deviation_error, -1.)
        
        # hip motion penalty
        rew_hip = torch.mul(torch.sum(torch.abs(self.actions[:, [1, 4, 7, 10]]), dim=1),-0.1)
        
        # air time reward
        first_contact = torch.mul((self.feet_air_time > 0.) , self.contact)
        self.feet_air_time += self.dt
        rew_airTime = torch.mul(torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1), 0.15)
        self.feet_air_time *= ~self.contact
        
        # contact phase
        contact = ((self.contact).clone().to(self.device)).float()
        rew_contact = torch.mul(torch.add(torch.abs(contact[:,0]-contact[:,3]),torch.abs(contact[:,1]-contact[:,2])) ,-0.3)
        rew_dif_1 = torch.mul(torch.add(torch.abs(contact[:,0]-contact[:,1]),torch.abs(contact[:,2]-contact[:,3])) ,0.05)
        rew_dif_2 = torch.mul(torch.add(torch.abs(contact[:,0]-contact[:,2]),torch.abs(contact[:,1]-contact[:,3])) ,0.05)
        rew_dof_pos = torch.mul(torch.add(torch.norm(self.actions[:,0:3]-self.actions[:,9:12],dim=1),torch.norm(self.actions[:,3:6]-self.actions[:,6:9],dim=1)) ,-0.05)
        
        FL_xyz = fk(self.dof_pos[:,0],self.dof_pos[:,1],self.dof_pos[:,2])
        FR_xyz = fk(self.dof_pos[:,3],self.dof_pos[:,4],self.dof_pos[:,5])
        RL_xyz = fk(self.dof_pos[:,6],self.dof_pos[:,7],self.dof_pos[:,8])
        RR_xyz = fk(self.dof_pos[:,9],self.dof_pos[:,10],self.dof_pos[:,11])
        
        #trot_plane:rew_forward:1.0  rew_airTime:0.15  rew_contact:-0.3  rew_dif_1:0.05   rew_dof_pos:-0.05 rew_base:-0.1
        #move straight:rew_forward:1.0  rew_airTime:0.15  rew_contact:-0.3  rew_dif_1:0.05   rew_dof_pos:-0.05 rew_deviation:-1.
        
        # base balance
        rew_base = torch.mul(torch.norm(self.euler[:,:2]),-0.1)
        
        """#feet_contact_forces_judge
        true = torch.ones((self.num_envs,4),device='cuda:0')
        false = torch.zeros((self.num_envs,4),device='cuda:0')
        contact_penalty = torch.sum(torch.where(self.feet_contact_forces>80 , true, false).int(),dim=1)
        contact_penalty = torch.mul(contact_penalty,-0.1)"""

        total_reward = rew_forward + rew_airTime + rew_contact + rew_dif_1 + rew_dif_2 + rew_dof_pos + rew_base + rew_lin_x_vel
      
        # euler_judge
        euler_reset = torch.norm(torch.where(torch.abs(self.euler) > self.euler_max, self.ones, self.zeros).float(),dim=1).int()
        a=torch.mul(euler_reset,-2)
        total_reward =torch.add(total_reward,a)

        # height_judge
        height_reset = torch.where(self.root_states[:,2]<0.28, self.one, self.zero).int()
        #height_success = torch.where(self.root_states[:,2]>2.0, self.one, self.zero).int()
        c=torch.mul(height_reset,-2)
        #d=torch.mul(height_success,50)
        total_reward = torch.add(total_reward,c)
        #total_reward = torch.add(total_reward,d)
        
        # reset agents
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        #stuck_reset = (self.base_lin_vel[:,0]<0.01)&(self.progress_buf>40)
        #reset = reset | stuck_reset
        #reset = reset | height_success
        reset = reset | euler_reset
        reset = reset | height_reset
        time_out = self.progress_buf > self.max_episode_length  
        reset = reset | time_out

        total_reward = torch.clip(total_reward, 0., None)
        
        self.rew_buf[:] = total_reward
        self.reset_buf[:] = reset
        
        
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.contact = torch.stack((self.contact_forces[:,3,2],self.contact_forces[:,6,2],self.contact_forces[:,9,2],self.contact_forces[:,12,2]),dim=1) > 1.
        
        self.contact_z_forces = self.contact_forces[:,:,2]
        self.feet_contact_forces = torch.stack((self.contact_z_forces[:,3],self.contact_z_forces[:,6],self.contact_z_forces[:,9],self.contact_z_forces[:,12]),dim=1)
        self.base_quat = self.root_states[:, 3:7]
        self.euler = torch.abs(quat_to_euler(self.base_quat,self.num_envs))
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        obs = torch.cat((self.root_states[:,:3],
                       self.base_lin_vel[:,:2],
                       self.base_ang_vel,
                       self.dof_pos,
                       self.contact,
                       self.euler
                       ), dim=-1)
        self.obs_buf[:] = obs

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-2.1, -1.8, -1.5, -1.2, 1.5, 1.8, 2.1, 2.4], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points = (points/self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.vertical_scale


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def ik(x,y,z,num_envs):
    # type: (Tensor, Tensor, Tensor, int)
    h=0.0838
    d=torch.sqrt(torch.add(torch.pow(y,2),torch.pow(z,2)))
    a=torch.full((num_envs,),0.0839,device='cuda:0')
    d=torch.where(d>0.0838,d,a)
    l=torch.sqrt(torch.sub(torch.pow(d,2),h*h))
    gamma=torch.sub(torch.atan(torch.div(y,z)),torch.atan(torch.div(h,l)))
    s=torch.sqrt(torch.add(torch.pow(l,2),torch.pow(x,2)))
    n=torch.div((torch.sub(torch.pow(s,2),0.08)),0.4)
    b=torch.full((num_envs,),0.1999999,device='cuda:0')
    c=torch.full((num_envs,),-0.199999,device='cuda:0')
    n=torch.where(n>=0.2,b,n)
    n=torch.where(n<=-0.2,c,n)
    beta=torch.mul(torch.acos(torch.div(n,0.2)),-1)
    alpha=torch.sub(torch.acos(torch.div(torch.add(0.2,n),s)),torch.atan(torch.div(x,l)))
    angle=torch.stack((gamma,alpha,beta),dim=1)

    return angle

@torch.jit.script
def fk(gamma,alpha,beta):
    beta=torch.mul(beta,-1)
    h=0.0838
    theta=torch.sub(torch.add(1.57079637,beta),alpha)
    x=-torch.mul(0.2,torch.add(torch.sin(alpha),torch.cos(theta)))
    y=h*torch.cos(gamma)+0.2*torch.mul(torch.add(torch.cos(alpha),torch.sin(theta)),torch.sin(gamma))
    z=h*torch.sin(gamma)+0.2*torch.mul(torch.add(torch.cos(alpha),torch.sin(theta)),torch.cos(gamma))
    xyz = torch.stack((x,y,z),dim=1)
    return xyz
     

@torch.jit.script
def quat_to_euler(a,num_envs):
    # type: (Tensor,int)
     shape = torch.rand(num_envs,3).shape
     a.reshape(-1,4)
     x,y,z,w = a[:,0],a[:,1],a[:,2],a[:,3]
     r = torch.atan2(torch.mul(torch.add(torch.mul(w,x),torch.mul(y,z)),2),torch.sub(1,2*(torch.pow(x,2)+torch.pow(y,2))))
     p = torch.asin(2*(torch.mul(w,y)-torch.mul(z,x)))
     y = torch.atan2(2*(torch.mul(w,z)+torch.mul(x,y)),1-2*(torch.pow(z,2)+torch.pow(y,2)))
     euler = torch.stack([r,p,y], dim=-1).view(shape)
     return euler
     
@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


