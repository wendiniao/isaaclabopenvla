#!/usr/bin/env python3
"""
Step 3: Fine-tuned OpenVLA Model Control
Uses your fine-tuned model to grab the cube in Isaac Lab
"""

import argparse
import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="franka_panda")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--model_dir", type=str, default="./openvla_finetuned/final_model", help="Path to fine-tuned model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set("/log/level", 40)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


class FineTunedOpenVLAController:
    """Controller using your fine-tuned OpenVLA model"""
    
    def __init__(self, model_dir, base_model="openvla/openvla-7b", device="cuda"):
        self.device = device
        self.model_dir = model_dir
        self.base_model = base_model
        
        print(f"🎯 Loading FINE-TUNED OpenVLA model...")
        print(f"📁 Model directory: {model_dir}")
        print(f"🔧 Base model: {base_model}")
        
        # Verify model directory exists
        if not os.path.exists(model_dir):
            raise ValueError(f"Fine-tuned model directory not found: {model_dir}")
        
        self.load_model_and_processor()
        
        print("✅ Fine-tuned model loaded successfully!")
        print("🧠 FINE-TUNED MODEL CONTROL")
        print("   Using your custom-trained model for cube grasping!")
        print()
    
    def load_model_and_processor(self):
        """Load the fine-tuned model with LoRA adapter"""
        
        # Load processor from fine-tuned directory
        print("📥 Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, 
            trust_remote_code=True
        )
        
        # Configure 8-bit quantization (same as training)
        print("📥 Loading base model with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        
        # Load base model
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Prepare for loading LoRA
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Load LoRA adapter
        print("📥 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        self.model.eval()
        
        print("✅ Fine-tuned model with LoRA loaded!")
    
    def predict_action_from_image(self, rgb_image, instruction="pick up the red cube"):
        """Predict action using fine-tuned model"""
        
        # Prepare image
        if isinstance(rgb_image, torch.Tensor):
            rgb_np = rgb_image.cpu().numpy()
        else:
            rgb_np = rgb_image
        
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        image = Image.fromarray(rgb_np).resize((224, 224))
        
        # Use the SAME format as training data
        prompt = f"Task: {instruction} Action:"
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Explicitly move inputs to CUDA as the warning suggests
        try:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        except Exception as e:
            print(f"Warning: Could not move inputs to CUDA: {e}")
        
        # Generate prediction
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=1.0
                )
                
                # Decode output
                output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Parse action from the response
                action = self._parse_action_from_text(output_text)
                
                return action, output_text
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
                return None, f"Error: {e}"
    
    def _parse_action_from_text(self, text):
        """Parse action from fine-tuned model output"""
        
        # The fine-tuned model should output: move_to_pose(x, y, z, qx, qy, qz, qw)
        pattern = r'move_to_pose\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)'
        match = re.search(pattern, text)
        
        if match:
            try:
                values = [float(match.group(i)) for i in range(1, 8)]
                action = np.array(values, dtype=np.float32)
                
                if self._is_valid_action(action):
                    print(f"   📍 Fine-tuned model output: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
                    return action
                else:
                    print(f"   ⚠️ Invalid action from model: {action}")
            except ValueError as e:
                print(f"   ⚠️ Error parsing action: {e}")
        
        # Fallback: try to extract any coordinate pattern
        coord_pattern = r'([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)'
        coord_match = re.search(coord_pattern, text)
        
        if coord_match:
            try:
                x, y, z = float(coord_match.group(1)), float(coord_match.group(2)), float(coord_match.group(3))
                # Use default downward-facing orientation
                action = np.array([x, y, z, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                
                if self._is_valid_action(action):
                    print(f"   📍 Extracted coordinates: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    return action
            except ValueError:
                pass
        
        print(f"   ❌ Could not parse action from: {text[:100]}...")
        return None
    
    def _is_valid_action(self, action):
        """Validate action is in workspace"""
        if action is None or len(action) < 7:
            return False
            
        x, y, z = action[0:3]
        
        # Workspace bounds
        if x < 0.1 or x > 0.9:
            return False
        if y < -0.4 or y > 0.6:
            return False
        if z < 0.02 or z > 0.8:
            return False
        
        # Check quaternion is reasonable
        quat = action[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 0.1 or quat_norm > 2.0:
            return False
        
        return True


@configclass
class CubeGraspingSceneCfg(InteractiveSceneCfg):
    """Configuration for cube grasping scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.4)),
    )
    
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            scale=(1.2, 0.8, 0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # Red cube to grasp
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                disable_gravity=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.2, 0.05), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    
    # Robot
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
        rot=(1.0, 0.0, 0.0, 0.0)
    )
    
    # External camera for better view
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.43, 1.8, 2.2),
            rot=(0.0078, 0.0030, 0.3583, 0.9336),
            convention="opengl"
        ),
    )


def run_finetuned_model_control(sim, scene, model_controller):
    """Control loop using your FINE-TUNED model"""
    
    robot = scene["robot"]
    camera = scene["camera"]
    cube = scene["cube"]
    
    # IK controller setup
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="dls", 
        ik_params={"k_val": 1.0}
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, 
        num_envs=scene.num_envs, 
        device=sim.device
    )
    
    # Robot entity configuration
    robot_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=["panda_joint[1-7]"], 
        body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(scene)
    
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Let scene settle
    print("⏳ Settling scene...")
    for _ in range(100):
        sim.step()
        scene.update(sim_dt)
    
    # Get initial positions
    cube_pos = cube.data.root_state_w[0, 0:3].cpu().numpy()
    print(f"📍 Cube position: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f}]")
    
    robot_ee_pos = robot.data.body_state_w[0, robot_entity_cfg.body_ids[0], 0:3].cpu().numpy()
    print(f"🤖 Robot EE position: [{robot_ee_pos[0]:.2f}, {robot_ee_pos[1]:.2f}, {robot_ee_pos[2]:.2f}]")
    
    print("🎯 Starting FINE-TUNED MODEL control...")
    print("   Your custom model should perform much better than the original!")
    print()
    
    # Control variables
    current_target = None
    gripper_state = "open"
    prediction_interval = 25  # Ask model every 25 steps
    last_prediction_step = -prediction_interval
    
    # Statistics
    total_predictions = 0
    valid_predictions = 0
    failed_predictions = 0
    
    try:
        while simulation_app.is_running() and count < 2000:
            
            # Get robot state
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            ee_pos = ee_pose_w[0, 0:3].cpu().numpy()
            
            # GET FINE-TUNED MODEL PREDICTION
            if count - last_prediction_step >= prediction_interval:
                
                rgb_image = camera.data.output.get("rgb") if camera.data else None
                
                if rgb_image is not None and len(rgb_image) > 0:
                    
                    # Ask YOUR fine-tuned model what to do
                    predicted_action, model_output = model_controller.predict_action_from_image(
                        rgb_image[0],
                        instruction="pick up the red cube"  # Same instruction as training
                    )
                    
                    total_predictions += 1
                    
                    if predicted_action is not None:
                        # FINE-TUNED MODEL SUCCESS!
                        current_target = torch.tensor(
                            predicted_action, 
                            device=sim.device, 
                            dtype=torch.float32
                        ).unsqueeze(0)
                        
                        valid_predictions += 1
                        
                        print(f"🎯 Step {count}: Fine-tuned model → [{predicted_action[0]:.2f}, {predicted_action[1]:.2f}, {predicted_action[2]:.2f}]")
                        print(f"   Current EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
                        print(f"   Model says: {model_output.split('Action:')[-1][:60]}...")
                        
                        # Grasp logic: close gripper when getting close to cube level
                        if predicted_action[2] < 0.12 and gripper_state == "open":
                            print(f"   🤏 Fine-tuned model triggers grasp!")
                            gripper_state = "closing"
                        
                    else:
                        # MODEL FAILED
                        failed_predictions += 1
                        print(f"⚠️  Step {count}: Fine-tuned model prediction failed")
                        print(f"   Output: {model_output[:80]}...")
                        
                        # Use reasonable fallback
                        if current_target is None:
                            fallback_target = torch.tensor(
                                [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15, 0.0, 1.0, 0.0, 0.0],
                                device=sim.device,
                                dtype=torch.float32
                            ).unsqueeze(0)
                            current_target = fallback_target
                            print(f"   🔄 Using fallback target")
                    
                    last_prediction_step = count
            
            # Execute current target
            if current_target is not None:
                ik_cmd = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
                ik_cmd[0] = current_target
                diff_ik_controller.set_command(ik_cmd)
                
                joint_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                
                # Prepare full joint command
                full_joint_cmd = robot.data.joint_pos.clone()
                full_joint_cmd[:, robot_entity_cfg.joint_ids] = joint_des
                
                # Handle gripper
                if gripper_state == "closing":
                    gripper_state = "closed"
                    print("   ✊ Gripper closed by fine-tuned model!")
                
                gripper_pos = 0.005 if gripper_state == "closed" else 0.04
                
                # Set gripper joints
                for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
                    if joint_name in robot.joint_names:
                        joint_idx = robot.joint_names.index(joint_name)
                        full_joint_cmd[:, joint_idx] = gripper_pos
                
                robot.set_joint_position_target(full_joint_cmd)
            
            # Step simulation
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(sim_dt)
            
            # Check for success
            if gripper_state == "closed":
                cube_z = cube.data.root_state_w[0, 2].item()
                if cube_z > 0.20:
                    print(f"\n🎉 SUCCESS! Fine-tuned model lifted cube to {cube_z:.2f}m")
                    break
            
            # Progress update
            if count % 200 == 0:
                cube_z = cube.data.root_state_w[0, 2].item()
                print(f"📊 Step {count}: Cube z={cube_z:.3f}, Gripper={gripper_state}, Valid predictions={valid_predictions}/{total_predictions}")
                    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        traceback.print_exc()
    
    # Final results
    cube_final_z = cube.data.root_state_w[0, 2].item()
    success = cube_final_z > 0.15
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Final cube height: {cube_final_z:.3f}m")
    print(f"Total steps: {count}")
    print(f"Total predictions: {total_predictions}")
    print(f"Valid predictions: {valid_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    
    if total_predictions > 0:
        success_rate = valid_predictions / total_predictions * 100
        print(f"Fine-tuned model success rate: {success_rate:.1f}%")
    
    print(f"Task success: {'✅ YES' if success else '❌ NO'}")
    
    if success:
        print("\n🎯 CONGRATULATIONS! Your fine-tuned model successfully completed the task!")
        print("   The fine-tuning process worked!")
    else:
        print("\n💡 The fine-tuned model needs more training or different data.")


def main():
    """Main function"""
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        print("="*60)
        print("FINE-TUNED OpenVLA MODEL CONTROL")
        print("="*60)
        print("Using your custom fine-tuned model for cube grasping")
        print()
        
        # Initialize fine-tuned model controller
        model_controller = FineTunedOpenVLAController(
            model_dir=args_cli.model_dir,
            device=args_cli.device
        )
        
        # Setup simulation
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        
        if not args_cli.headless:
            sim.set_camera_view([1.5, 1.5, 1.2], [0.4, 0.0, 0.2])
        
        # Create scene
        scene_cfg = CubeGraspingSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        
        # Run fine-tuned model control
        run_finetuned_model_control(sim, scene, model_controller)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()