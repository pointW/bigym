"""Visual test for RBY1 IK solver with MuJoCo viewer."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import os
import time
from bigym.ik.rby1_ik import RBY1IK


def main():
    """Main function to test RBY1 IK visually."""
    print("Loading RBY1 model...")
    
    # Load model
    model_path = Path(__file__).parent.parent / "bigym" / "envs" / "xmls" / "rby1" / "model_act.xml"
    
    # Change to model directory for relative paths
    original_dir = os.getcwd()
    os.chdir(model_path.parent)
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path.name))
        data = mujoco.MjData(model)
        
        # Initialize IK solver
        print("Initializing IK solver...")
        ik_solver = RBY1IK(model, data)
        
        # Set initial configuration with proper base height
        data.qpos[:] = 0
        # Set base height to proper ground level for RBY1 (wheels should touch ground)
        # RBY1 base is typically around 0.3-0.4m high
        data.qpos[2] = 0.35  # Adjust Z to proper height for RBY1
        data.qpos[3] = 1.0  # Identity quaternion for base orientation
        mujoco.mj_forward(model, data)
        
        # Store base configuration (will remain fixed)
        base_pos = np.array([0.0, 0.0, data.qpos[2]])  # Keep the proper Z height
        base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Get initial end effector positions
        left_ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_l")
        right_ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_r")
        
        initial_left_pos = data.site_xpos[left_ee_site_id].copy()
        initial_right_pos = data.site_xpos[right_ee_site_id].copy()
        
        print(f"Initial left EE position: {initial_left_pos}")
        print(f"Initial right EE position: {initial_right_pos}")
        print(f"Base position (fixed): X={base_pos[0]:.3f}, Y={base_pos[1]:.3f}, Z={base_pos[2]:.3f}")
        
        print(f"\nModel info:")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  Actuators: {model.nu}")
        print(f"  Sites: {model.nsite}")
        print(f"  IK-controlled joints: {len(ik_solver.ik_controlled_indices)} (torso + arms)")
        
        # Check what geom groups are available
        print(f"\nVisualization tip: Press these keys in the viewer:")
        print(f"  'C' - Toggle contact points")
        print(f"  'T' - Toggle transparency")
        print(f"  '0' - Toggle geom group 0 (usually visual geoms)")
        print(f"  '1' - Toggle geom group 1 (usually collision geoms)")
        print(f"  '2' - Toggle geom group 2")
        
        # Define target sequence - only arm movements, base stays fixed
        targets = [
            {
                'left': initial_left_pos + np.array([0.1, 0.05, 0.05]),
                'right': initial_right_pos + np.array([0.1, -0.05, 0.05]),
                'description': 'Both arms slightly forward and up'
            },
            {
                'left': initial_left_pos + np.array([0.0, 0.15, 0.1]),
                'right': initial_right_pos + np.array([0.0, -0.15, 0.1]),
                'description': 'Arms wide and up'
            },
            {
                'left': initial_left_pos + np.array([0.1, 0.1, -0.05]),
                'right': initial_right_pos + np.array([0.1, -0.1, -0.05]),
                'description': 'Arms forward and slightly down'
            },
            {
                'left': initial_left_pos + np.array([0.15, 0.0, 0.0]),
                'right': initial_right_pos + np.array([0.15, 0.0, 0.0]),
                'description': 'Both arms straight forward'
            },
            {
                'left': initial_left_pos + np.array([0.05, 0.0, 0.15]),
                'right': initial_right_pos + np.array([0.05, 0.0, 0.15]),
                'description': 'Both arms up high'
            },
        ]
        
        # Launch viewer
        print("\nLaunching MuJoCo viewer...")
        print("The viewer will auto-cycle through targets every 3 seconds")
        print("Base remains fixed, only torso and arms move")
        print("Close the window to exit")
        print("-" * 50)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            target_idx = 0
            solving = False
            solve_time = 0
            auto_cycle = True  # Auto-cycle through targets
            auto_cycle_time = time.time()
            
            # Set camera to a good viewing angle
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0, 0, 0.5]
            
            # Disable collision geometry visualization by default
            # Group 0 = visual geoms, Group 1 = collision geoms
            viewer.opt.geomgroup[1] = 0  # Hide collision geoms (often shown as cylinders)
            
            # Enable visualization options
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # Disable contact points
            
            # Main loop
            while viewer.is_running():
                current_time = time.time()
                
                # Auto-cycle through targets every 3 seconds
                if auto_cycle and current_time - auto_cycle_time > 3.0:
                    target_idx = (target_idx + 1) % len(targets)
                    solving = True
                    solve_time = current_time
                    auto_cycle_time = current_time
                    print(f"\nTarget {target_idx + 1}: {targets[target_idx]['description']}")
                
                # Solve IK for current target
                if solving and current_time - solve_time < 2.0:  # Solve for 2 seconds
                    target = targets[target_idx]
                    
                    # Solve IK with fixed base
                    solution, success, info = ik_solver.solve(
                        base_pos=base_pos,
                        base_quat=base_quat,
                        left_target_pos=target['left'],
                        right_target_pos=target['right'],
                        current_qpos=data.qpos.copy(),
                        max_iterations=50,
                        tolerance=0.01
                    )
                    
                    # Smoothly interpolate to solution
                    alpha = min(1.0, (current_time - solve_time) * 2.0)  # Ramp up over 0.5 seconds
                    data.qpos[:] = (1 - alpha) * data.qpos + alpha * solution
                    
                    if success and alpha >= 1.0:
                        print(f"  ✓ IK solved successfully in {info['iterations']} iterations")
                        if 'errors' in info:
                            for key, value in info['errors'].items():
                                print(f"    {key}: {value:.4f}")
                        
                        # Verify base didn't move
                        actual_base_pos = data.qpos[0:3]
                        base_drift = np.linalg.norm(actual_base_pos - base_pos)
                        if base_drift > 0.001:
                            print(f"  ⚠️ Base drift detected: {base_drift:.6f}m")
                    elif not success and alpha >= 1.0:
                        print(f"  ✗ IK failed to converge after {info['iterations']} iterations")
                        if 'errors' in info:
                            for key, value in info['errors'].items():
                                print(f"    {key}: {value:.4f}")
                
                # Forward dynamics
                mujoco.mj_forward(model, data)
                
                # Update viewer (visualization happens automatically through MuJoCo)
                viewer.sync()
        
        print("\nViewer closed.")
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()