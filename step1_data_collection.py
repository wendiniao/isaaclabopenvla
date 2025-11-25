# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Step 1: Data Collection Script - Collect training data using inverse kinematics

This script demonstrates grabbing a cube using inverse kinematics and collecting
training data (images + actions) for fine-tuning OpenVLA.

Usage:
./isaaclab.sh -p scripts/tutorials/05_controllers/step1_data_collection.py
"""

import argparse
import logging
import os
import json
import numpy as np
import torch
from PIL import Image
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for OpenVLA fine-tuning.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--data_dir", type=str, default="./openvla_training_data", help="Directory to save training data.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Suppress warnings
import carb
carb.settings.get_settings().set("/log/level", 40)
carb.settings.get_settings().set("/log/omni.graph.core.plugin/level", 40)
carb.settings.get_settings().set("/log/omni.usd/level", 40)
carb.settings.get_settings().set("/log/omni.usdImaging/level", 40)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


class DataCollector:
    """Data collector for OpenVLA training."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.action_dir = os.path.join(data_dir, "actions")
        
        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.action_dir, exist_ok=True)
        
        self.data_count = 0
        self.collected_data = []
        
    def save_data_point(self, rgb_image, action, instruction, robot_state):
        """Save a single data point (image + action + metadata)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"{timestamp}_{self.data_count:04d}"
        
        # Save image
        if isinstance(rgb_image, torch.Tensor):
            rgb_np = rgb_image.cpu().numpy()
        else:
            rgb_np = rgb_image
            
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
            
        image_path = os.path.join(self.image_dir, f"{data_id}.png")
        Image.fromarray(rgb_np).save(image_path)
        
        # Save action and metadata
        action_data = {
            "data_id": data_id,
            "instruction": instruction,
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "robot_state": robot_state.tolist() if isinstance(robot_state, np.ndarray) else robot_state,
            "image_path": image_path,
            "timestamp": timestamp
        }
        
        action_path = os.path.join(self.action_dir, f"{data_id}.json")
        with open(action_path, 'w') as f:
            json.dump(action_data, f, indent=2)
            
        self.collected_data.append(action_data)
        self.data_count += 1
        
        print(f"💾 Saved data point {self.data_count}: {data_id}")
        
    def save_dataset_summary(self):
        """Save summary of collected dataset."""
        summary_path = os.path.join(self.data_dir, "dataset_summary.json")
        summary = {
            "total_samples": self.data_count,
            "collection_date": datetime.now().isoformat(),
            "robot_type": args_cli.robot,
            "task_description": "cube grasping task",
            "data_format": "RGB images + 7-DOF end-effector poses",
            "samples": self.collected_data
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📋 Dataset summary saved to {summary_path}")
        print(f"🎯 Total collected samples: {self.data_count}")


@configclass
class CubeGraspingSceneCfg(InteractiveSceneCfg):
    """Configuration for cube grasping scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tables/Willow.usd", 
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,  # Table is static
            disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, -0.75)),
    )
    
    # Cube to grasp
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )

    # robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1, 0, 0, 0)
    )

    # Camera for data collection
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0,0.0,0.0,0.0),
            convention="ros"
        ),
    )


def run_data_collection(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run data collection using inverse kinematics."""
    
    # Extract scene entities
    robot = scene["robot"]
    camera = scene["camera"]
    cube = scene["cube"]
    
    # Initialize data collector
    data_collector = DataCollector(args_cli.data_dir)
    
    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="svd"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, 
        num_envs=scene.num_envs, 
        device=sim.device
    )

    # Robot configuration
    robot_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=["panda_joint.*"], 
        body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(scene)
    
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Simulation parameters
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Grasping sequence states
    grasp_state = "approach"  # "approach", "descend", "grasp", "lift", "done"
    data_collection_steps = []
    
    # Define grasping trajectory waypoints
    cube_pos = torch.tensor([0.5, 0.0, 0.05], device=sim.device)
    
    waypoints = [
        # Approach: above cube
        torch.cat([cube_pos + torch.tensor([0.0, 0.0, 0.15], device=sim.device), 
                   torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)]),
        # Pre-grasp: closer to cube
        torch.cat([cube_pos + torch.tensor([0.0, 0.0, 0.10], device=sim.device), 
                   torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)]),
        # Grasp: at cube level
        torch.cat([cube_pos + torch.tensor([0.0, 0.0, 0.025], device=sim.device), 
                   torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)]),
        # Lift: lift cube
        torch.cat([cube_pos + torch.tensor([0.0, 0.0, 0.20], device=sim.device), 
                   torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)]),
    ]
    
    current_waypoint = 0
    waypoint_steps = 100  # Steps to spend at each waypoint
    gripper_closed = False
    
    # Task instruction
    task_instruction = "pick up the red cube"
    
    print(f"🤖 Starting data collection for task: '{task_instruction}'")
    print(f"📸 Will collect ~20 data points during grasping sequence")
    print(f"💾 Data will be saved to: {args_cli.data_dir}")
    
    # Simulation loop
    while simulation_app.is_running() and data_collector.data_count < 20:
        
        # Get current robot state
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        
        # Compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Set target pose based on current waypoint
        if current_waypoint < len(waypoints):
            target_pose = waypoints[current_waypoint].unsqueeze(0)
            
            # Set IK command
            ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
            ik_commands[0] = target_pose
            diff_ik_controller.set_command(ik_commands)
            
            # Check if we should move to next waypoint
            current_ee_pos = ee_pose_w[0, 0:3]
            target_pos = target_pose[0, 0:3]
            distance_to_target = torch.norm(current_ee_pos - target_pos)
            
            if distance_to_target < 0.02 or count % waypoint_steps == waypoint_steps - 1:
                current_waypoint += 1
                print(f"🎯 Moving to waypoint {current_waypoint}")
                
                # Close gripper when reaching grasp waypoint
                if current_waypoint == 3 and not gripper_closed:
                    print("🤏 Closing gripper...")
                    # Close gripper
                    current_joint_pos = robot.data.joint_pos.clone()
                    finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
                    for joint_name in finger_joint_names:
                        if joint_name in robot.joint_names:
                            joint_idx = robot.joint_names.index(joint_name)
                            current_joint_pos[:, joint_idx] = 0.01
                    
                    finger_joint_ids = [robot.joint_names.index(name) for name in finger_joint_names 
                                      if name in robot.joint_names]
                    if finger_joint_ids:
                        robot.set_joint_position_target(current_joint_pos[:, finger_joint_ids], 
                                                      joint_ids=finger_joint_ids)
                    gripper_closed = True
        
        # Compute joint positions
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Collect data every few steps
        if count % 25 == 0 and camera.data is not None and hasattr(camera.data, 'output'):
            rgb_image = camera.data.output.get("rgb")
            if rgb_image is not None and len(rgb_image) > 0:
                # Get current end-effector pose as action
                current_ee_pose = ee_pose_w[0].cpu().numpy()  # [x, y, z, qx, qy, qz, qw]
                
                # Get current joint positions as robot state
                current_joint_state = robot.data.joint_pos[0, :7].cpu().numpy()
                
                # Save data point
                data_collector.save_data_point(
                    rgb_image=rgb_image[0],
                    action=current_ee_pose,
                    instruction=task_instruction,
                    robot_state=current_joint_state
                )
        
        # Apply joint commands
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Print progress
        if count % 100 == 0:
            ee_pos = ee_pose_w[0, 0:3].cpu().numpy()
            print(f"Step {count}: EE Position: {ee_pos}, Data collected: {data_collector.data_count}/20")
        
        # End collection after reaching final waypoint
        if current_waypoint >= len(waypoints) and data_collector.data_count >= 15:
            break
    
    # Save dataset summary
    data_collector.save_dataset_summary()
    print("✅ Data collection complete!")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Design scene
    scene_cfg = CubeGraspingSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Starting data collection with inverse kinematics")
    
    # Run data collection
    run_data_collection(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()