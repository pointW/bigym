"""Simple test to visualize RBY1 with grippers using BiGym."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer

# Import BiGym components
from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.action_modes import JointPositionActionMode


def main():
    """Visualize RBY1 with grippers using BiGym environment."""
    
    print("Creating RBY1 with Robotiq grippers using BiGym ReachTarget environment...")
    
    try:
        # Create ReachTargetRBY1 environment
        env = ReachTargetRBY1(
            action_mode=JointPositionActionMode(absolute=True, floating_base=True),
            render_mode=None,  # We'll use our own viewer
        )
        
        print("✅ Created RBY1 environment")
        print("   BiGym automatically attaches Robotiq grippers to RBY1")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Access the underlying MuJoCo model and data
        mojo = env._mojo
        model = mojo.physics.model._model
        data = mojo.physics.data._data
        
        print(f"\nModel statistics:")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  DOF: {model.nq}")
        print(f"  Actuators: {model.nu}")
        
        # Count Robotiq/gripper components
        gripper_bodies = 0
        gripper_geoms = 0
        gripper_joints = 0
        
        print("\nSearching for gripper components...")
        
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and ('gripper' in name.lower() or 'robotiq' in name.lower() or
                        'driver' in name.lower() or 'follower' in name.lower() or
                        'finger' in name.lower() or 'coupler' in name.lower()):
                gripper_bodies += 1
                if gripper_bodies <= 5:  # Show first few
                    print(f"  Gripper body: {name}")
        
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ('gripper' in name.lower() or 'robotiq' in name.lower() or
                        'finger' in name.lower() or 'pad' in name.lower()):
                gripper_geoms += 1
        
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and ('gripper' in name.lower() or 'finger' in name.lower() or
                        'driver' in name.lower() or 'follower' in name.lower()):
                gripper_joints += 1
        
        print(f"\nGripper components found:")
        print(f"  Bodies: {gripper_bodies}")
        print(f"  Geoms: {gripper_geoms}")
        print(f"  Joints: {gripper_joints}")
        
        if gripper_bodies > 0:
            print("\n✅ Robotiq grippers are attached!")
        else:
            print("\n⚠️ No gripper components found")
            print("   BiGym may be using a simplified gripper model")
        
        # Check for end effectors
        left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_l")
        right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_r")
        
        if left_ee_id >= 0 and right_ee_id >= 0:
            left_pos = data.site_xpos[left_ee_id]
            right_pos = data.site_xpos[right_ee_id]
            print(f"\nEnd effector positions:")
            print(f"  Left:  [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
            print(f"  Right: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
        
        # Check for base control (RBY1 specific)
        base_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
        if base_target_id >= 0:
            print(f"\nBase control: Mocap body 'base_target' found")
            print("  RBY1 base is controlled via mocap body (no wheel actuators)")
        else:
            print(f"\nBase control: No mocap body found")
            print("  Base may be controlled through actuators")
        
        print(f"\nEnvironment info:")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Observation space keys: {list(obs.keys())}")
        
        print("\nControls:")
        print("  Mouse: Rotate view")
        print("  Scroll: Zoom")
        print("  ESC/Q: Exit")
        print("  0-4: Toggle geometry groups")
        print("\nLaunching viewer...")
        print("RBY1 ReachTarget environment includes targets and RBY1 with grippers")
        
        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Camera setup - adjusted for RBY1's size
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 4.0
            viewer.cam.lookat[:] = [0, 0, 1.0]
            
            # Hide collision geometry
            viewer.opt.geomgroup[3] = 0
            
            # Animation
            time = 0
            action_dim = env.action_space.shape[0]
            
            while viewer.is_running():
                time += 0.01
                phase = np.sin(time * 2)
                
                # Create a simple action respecting bounds
                action = np.zeros(action_dim)
                
                # Get action space bounds
                low_bounds = env.action_space.low
                high_bounds = env.action_space.high
                
                # RBY1 with mocap base control:
                # [base_x, base_y, base_rz, torso_joints(6), right_arm(7), left_arm(7), grippers(2)]
                # Total: 3 + 6 + 7 + 7 + 2 = 25
                
                # Move base slightly (controlled via mocap)
                if action_dim >= 3:
                    action[0] = 0.1 * phase  # Base X (range: -0.2, 0.2)
                    action[1] = 0.05 * phase  # Base Y (range: -0.2, 0.2)
                    action[2] = 0.2 * phase  # Base rotation (range: -0.5, 0.5)
                
                # Add some movement to torso and arms
                if action_dim >= 23:  # Has full body (3 base + 6 torso + 14 arms)
                    # Torso (indices 3-8)
                    action[4] = 0.5 * phase  # Torso pitch (range: -1.04, 1.52)
                    
                    # Right arm (indices 9-15)
                    action[9] = 0.5 * phase  # Right shoulder (range: -2.35, 2.35)
                    action[10] = -0.5 * phase  # Right arm_1 (range: -3.14, 0.05) - use negative
                    
                    # Left arm (indices 16-22)
                    action[16] = -0.5 * phase  # Left shoulder (range: -2.35, 2.35)
                    action[17] = 0.5 * phase  # Left arm_1 (range: -0.05, 3.14) - use positive
                    
                    # Grippers (last 2, range: 0, 1)
                    if action_dim >= 25:
                        action[-2] = 0.5 * (1 + phase)  # Left gripper (0 to 1)
                        action[-1] = 0.5 * (1 + phase)  # Right gripper (0 to 1)
                
                # Clip actions to ensure they're within bounds
                action = np.clip(action, low_bounds, high_bounds)
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                
                # Reset if episode ends
                if terminated or truncated:
                    obs, _ = env.reset()
                
                # Sync viewer
                viewer.sync()
        
        print("\nViewer closed.")
        env.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()