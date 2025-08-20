"""Simple test to visualize H1 with grippers using BiGym."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer

# Import BiGym components
from bigym.envs.reach_target import ReachTarget
from bigym.action_modes import JointPositionActionMode


def main():
    """Visualize H1 with grippers using BiGym environment."""
    
    print("Creating H1 with Robotiq grippers using BiGym ReachTarget environment...")
    
    try:
        # Create ReachTarget environment - it uses H1 by default with grippers
        env = ReachTarget(
            action_mode=JointPositionActionMode(absolute=True, floating_base=False),
            render_mode=None,  # We'll use our own viewer
        )
        
        print("✅ Created H1 environment")
        print("   BiGym automatically attaches Robotiq grippers to H1")
        
        # Reset environment
        obs, info = env.reset()
        
        # Access the underlying MuJoCo model and data
        # BiGym uses mojo wrapper, we need to access the physics
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
        left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_ee_site")
        right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")
        
        if left_ee_id < 0:
            left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_end_effector")
        if right_ee_id < 0:
            right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_end_effector")
        
        if left_ee_id >= 0 and right_ee_id >= 0:
            left_pos = data.site_xpos[left_ee_id]
            right_pos = data.site_xpos[right_ee_id]
            print(f"\nEnd effector positions:")
            print(f"  Left:  [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
            print(f"  Right: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
        
        print("\nEnvironment info:")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Observation space keys: {list(obs.keys())}")
        
        print("\nControls:")
        print("  Mouse: Rotate view")
        print("  Scroll: Zoom")
        print("  ESC/Q: Exit")
        print("  0-4: Toggle geometry groups")
        print("\nLaunching viewer...")
        print("ReachTarget environment includes targets and H1 with grippers")
        
        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Camera setup
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0, 0, 0.8]
            
            # Hide collision geometry
            viewer.opt.geomgroup[3] = 0
            
            # Animation
            time = 0
            action_dim = env.action_space.shape[0]
            
            while viewer.is_running():
                time += 0.01
                phase = np.sin(time * 2)
                
                # Create a simple action
                action = np.zeros(action_dim)
                
                # Add some movement to the arms
                # The exact mapping depends on the action mode
                # For JointPositionActionMode with H1:
                # Typically: [left_arm_joints..., right_arm_joints..., grippers]
                
                if action_dim >= 10:  # Has arms
                    # Simple movement
                    action[0] = 0.2 * phase  # Some joint
                    action[5] = 0.2 * phase  # Another joint
                    
                    # Try to control grippers (usually last 2)
                    if action_dim >= 12:
                        action[-2] = max(0, phase)  # Left gripper
                        action[-1] = max(0, phase)  # Right gripper
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Reset if episode ends
                if terminated or truncated:
                    obs, info = env.reset()
                
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