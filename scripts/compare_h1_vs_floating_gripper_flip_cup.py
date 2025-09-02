#!/usr/bin/env python3
"""Compare H1 joint demo replay with floating gripper cartesian demo replay."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.manipulation import FlipCup
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo import Demo

from bigym.floating_gripper_action_mode import FloatingGripperActionMode
from bigym.robots.configs.floating_grippers import FloatingGrippers
from bigym.const import HandSide


def get_gripper_info(env, side_str="left"):
    """Get gripper position and control info.
    
    Both H1 and FG use the same ROBOTIQ gripper. We retrieve exactly:
    1. base_mount body position/orientation (from ROBOTIQ model)
    2. left_pad body position/orientation (fingertip)
    3. right_pad body position/orientation (fingertip)
    4. Gripper control value (0=open, 255=closed)
    
    For reference, the wrist site (H1) or mocap body (FG) is where the gripper attaches.
    """
    physics = env.robot._mojo.physics
    model = physics.model
    data = physics.data
    
    info = {}
    
    # Convert side string to HandSide enum
    from bigym.const import HandSide
    from scipy.spatial.transform import Rotation
    side = HandSide.LEFT if side_str == "left" else HandSide.RIGHT
    
    # First get the attachment point for reference (wrist site for H1, mocap for FG)
    is_floating_gripper = hasattr(env.robot, '__class__') and 'FloatingGrippers' in str(env.robot.__class__)
    
    if is_floating_gripper:
        # Floating gripper: Get mocap body position (this is what we control)
        mocap_name = f"{side_str}_gripper_mocap"  # Note: has "_mocap" suffix
        try:
            mocap_id = model.body(mocap_name).id
            info['attachment_pos'] = data.xpos[mocap_id].copy()
            info['attachment_quat'] = data.xquat[mocap_id].copy()
        except:
            info['attachment_pos'] = np.zeros(3)
            info['attachment_quat'] = np.array([1, 0, 0, 0])
    else:
        # H1: Get wrist site position (where gripper is attached)
        wrist_site = env.robot._wrist_sites[side]
        site_bind = physics.bind(wrist_site.mjcf)
        info['attachment_pos'] = site_bind.xpos.copy()
        rot = Rotation.from_matrix(site_bind.xmat.reshape(3, 3))
        info['attachment_quat'] = rot.as_quat()[[3, 0, 1, 2]]  # Convert [x,y,z,w] to [w,x,y,z]
    
    # Now get the ROBOTIQ gripper bodies (same for both H1 and FG)
    # The gripper namespace differs: "h1/robotiq_2f85_{side}" for H1, just "{side}" for FG
    
    # Try different naming patterns
    if is_floating_gripper:
        # FG uses robotiq_2f85_ prefix
        base_mount_name = f"robotiq_2f85_{side_str}/base_mount"
        left_pad_name = f"robotiq_2f85_{side_str}/left_pad"
        right_pad_name = f"robotiq_2f85_{side_str}/right_pad"
    else:
        # H1 uses h1/ namespace
        base_mount_name = f"h1/robotiq_2f85_{side_str}/base_mount"
        left_pad_name = f"h1/robotiq_2f85_{side_str}/left_pad"
        right_pad_name = f"h1/robotiq_2f85_{side_str}/right_pad"
    
    # 1. Get base_mount body (ROBOTIQ gripper base)
    try:
        base_mount_id = model.body(base_mount_name).id
        info['base_mount_pos'] = data.xpos[base_mount_id].copy()
        info['base_mount_quat'] = data.xquat[base_mount_id].copy()
    except:
        # Fallback to attachment point
        info['base_mount_pos'] = info['attachment_pos'].copy()
        info['base_mount_quat'] = info['attachment_quat'].copy()
    
    # 2. Get left_pad body (left fingertip of this gripper)
    try:
        left_pad_id = model.body(left_pad_name).id
        info['left_pad_pos'] = data.xpos[left_pad_id].copy()
        info['left_pad_quat'] = data.xquat[left_pad_id].copy()
    except:
        info['left_pad_pos'] = np.zeros(3)
        info['left_pad_quat'] = np.array([1, 0, 0, 0])
    
    # 3. Get right_pad body (right fingertip of this gripper)  
    try:
        right_pad_id = model.body(right_pad_name).id
        info['right_pad_pos'] = data.xpos[right_pad_id].copy()
        info['right_pad_quat'] = data.xquat[right_pad_id].copy()
    except:
        info['right_pad_pos'] = np.zeros(3)
        info['right_pad_quat'] = np.array([1, 0, 0, 0])
    
    # 4. Calculate opening width (distance between pads)
    info['opening_width'] = np.linalg.norm(info['left_pad_pos'] - info['right_pad_pos'])
    
    # 5. Get gripper control value (from actuator)
    gripper = env.robot._grippers[side]
    if hasattr(gripper, '_actuators') and len(gripper._actuators) > 0:
        ctrl_val = physics.bind(gripper._actuators[0]).ctrl
        # Handle scalar or array
        if hasattr(ctrl_val, '__len__'):
            info['gripper_ctrl'] = ctrl_val[0]
        else:
            info['gripper_ctrl'] = ctrl_val
    else:
        info['gripper_ctrl'] = 0.0
    
    return info
    


def compare_demos():
    """Compare H1 joint demo with floating gripper cartesian demo."""
    
    # Use a specific seed for testing
    test_seed = 1718813881
    control_frequency = 20
    
    print("="*80)
    print("COMPARING H1 JOINT vs FLOATING GRIPPER CARTESIAN")
    print(f"Demo seed: {test_seed}")
    print("="*80)
    
    # ============ PART 1: H1 JOINT DEMO ============
    print("\n" + "="*60)
    print("PART 1: H1 JOINT DEMO REPLAY")
    print("="*60)
    
    # Create H1 environment (from examples/replay_demo.py)
    h1_env = FlipCup(
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]  # 4 DOF
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        render_mode=None,  # Headless for testing
    )
    
    # Get joint demos
    metadata = Metadata.from_env(h1_env)
    demo_store = DemoStore()
    joint_demos = demo_store.get_demos(metadata, amount=60, frequency=control_frequency)
    
    # Find demo with our test seed
    h1_demo = None
    for demo in joint_demos:
        if demo.seed == test_seed:
            h1_demo = demo
            break
    
    if h1_demo is None:
        print(f"ERROR: Could not find joint demo with seed {test_seed}")
        return
    
    print(f"Found H1 joint demo with {len(h1_demo.timesteps)} timesteps")
    
    # Reset and step through H1 demo
    h1_env.reset(seed=test_seed)
    
    h1_gripper_data = []
    for step_idx, timestep in enumerate(h1_demo.timesteps):
        # Get action from timestep
        if hasattr(timestep, 'action'):
            action = timestep.action
        elif hasattr(timestep, 'info') and 'action' in timestep.info:
            action = timestep.info['action']
        else:
            # For DemoStep, the action is stored differently
            action = timestep.info.get('demo_action')
            if action is None:
                action = timestep.executed_action
            if action is None:
                # Skip if no action found
                continue
        
        # Step environment
        obs, reward, terminated, truncated, info = h1_env.step(action)
        
        # Get gripper info
        left_info = get_gripper_info(h1_env, "left")
        right_info = get_gripper_info(h1_env, "right")
        
        h1_gripper_data.append({
            'step': step_idx,
            'left': left_info,
            'right': right_info,
            'success': info.get('task_success', False)
        })
        
        if info.get('task_success', False):
            print(f"H1 achieved success at step {step_idx}")
            break
        
        if terminated or truncated:
            break
    
    h1_env.close()
    
    # ============ PART 2: FLOATING GRIPPER CARTESIAN DEMO ============
    print("\n" + "="*60)
    print("PART 2: FLOATING GRIPPER CARTESIAN DEMO REPLAY")
    print("="*60)
    
    # Load cartesian demo (from scripts/floating_gripper_demo_replay.py)
    demo_dir = "rby1_cartesian_demos_flipcup"
    demo_path = Path(demo_dir)
    demo_files = sorted(demo_path.glob("rby1_cartesian_demo_*.safetensors"))
    
    # Find demo with our test seed
    fg_demo = None
    for demo_file in demo_files:
        demo = Demo.from_safetensors(demo_file)
        if demo.seed == test_seed:
            fg_demo = demo
            break
    
    if fg_demo is None:
        print(f"ERROR: Could not find cartesian demo with seed {test_seed}")
        print("Available demo seeds:")
        for demo_file in demo_files[:5]:
            demo = Demo.from_safetensors(demo_file)
            print(f"  {demo.seed}")
        return
    
    print(f"Found floating gripper cartesian demo with {len(fg_demo.timesteps)} timesteps")
    
    # Create floating gripper environment
    fg_env = FlipCup(
        action_mode=FloatingGripperActionMode(control_frequency=20),
        control_frequency=20,
        render_mode=None,  # Headless
        robot_cls=FloatingGrippers
    )
    
    # Reset and step through floating gripper demo
    fg_env.reset(seed=test_seed)
    
    fg_gripper_data = []
    fg_success_step = -1
    for step_idx in range(len(fg_demo.timesteps)):
        timestep = fg_demo.timesteps[step_idx]
        
        # Get action from demo
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            action = timestep.action
        
        if action is None:
            print(f"Step {step_idx}: No action found!")
            continue
        
        # Step environment
        obs, reward, terminated, truncated, info = fg_env.step(action)
        
        # Get gripper info
        left_info = get_gripper_info(fg_env, "left")
        right_info = get_gripper_info(fg_env, "right")
        
        fg_gripper_data.append({
            'step': step_idx,
            'left': left_info,
            'right': right_info,
            'success': info.get('task_success', False)
        })
        
        if info.get('task_success', False):
            print(f"Floating gripper achieved success at step {step_idx}")
            fg_success_step = step_idx
            break
        
        # Add debug info near expected success step
        if step_idx == 105:
            print(f"Debug at step 105 (H1 success step):")
            print(f"  Reward: {reward}")
            print(f"  Info keys: {info.keys()}")
            if 'task_success' in info:
                print(f"  Task success: {info['task_success']}")
            if 'success_rate' in info:
                print(f"  Success rate: {info['success_rate']}")
        
        if terminated or truncated:
            break
    
    fg_env.close()
    
    # ============ PART 3: COMPARISON ============
    print("\n" + "="*60)
    print("PART 3: COMPARISON RESULTS")
    print("="*60)
    
    # Compare the data
    min_steps = min(len(h1_gripper_data), len(fg_gripper_data))
    print(f"\nComparing {min_steps} steps")
    
    max_base_error = 0.0
    max_tip_error = 0.0
    
    # Sample some steps for detailed comparison
    sample_steps = [0, min_steps//4, min_steps//2, 3*min_steps//4, min_steps-1]
    
    for step in sample_steps:
        if step >= min_steps:
            continue
            
        h1_data = h1_gripper_data[step]
        fg_data = fg_gripper_data[step]
        
        print(f"\n--- Step {step} ---")
        
        # Compare ATTACHMENT POINTS (wrist site for H1, mocap for FG)
        left_attach_error = np.linalg.norm(h1_data['left']['attachment_pos'] - fg_data['left']['attachment_pos'])
        print(f"Left attachment point (wrist/mocap):")
        print(f"  H1 pos:  {h1_data['left']['attachment_pos']}")
        print(f"  FG pos:  {fg_data['left']['attachment_pos']}")
        print(f"  Error:   {left_attach_error*1000:.3f}mm")
        
        right_attach_error = np.linalg.norm(h1_data['right']['attachment_pos'] - fg_data['right']['attachment_pos'])
        print(f"Right attachment point (wrist/mocap):")
        print(f"  H1 pos:  {h1_data['right']['attachment_pos']}")
        print(f"  FG pos:  {fg_data['right']['attachment_pos']}")
        print(f"  Error:   {right_attach_error*1000:.3f}mm")
        
        # Compare BASE_MOUNT bodies
        left_base_error = np.linalg.norm(h1_data['left']['base_mount_pos'] - fg_data['left']['base_mount_pos'])
        print(f"\nLeft base_mount body:")
        print(f"  H1 pos:  {h1_data['left']['base_mount_pos']}")
        print(f"  FG pos:  {fg_data['left']['base_mount_pos']}")
        print(f"  Error:   {left_base_error*1000:.3f}mm")
        
        right_base_error = np.linalg.norm(h1_data['right']['base_mount_pos'] - fg_data['right']['base_mount_pos'])
        print(f"Right base_mount body:")
        print(f"  H1 pos:  {h1_data['right']['base_mount_pos']}")
        print(f"  FG pos:  {fg_data['right']['base_mount_pos']}")
        print(f"  Error:   {right_base_error*1000:.3f}mm")
        
        # Compare PAD bodies (fingertips)
        left_left_pad_error = np.linalg.norm(h1_data['left']['left_pad_pos'] - fg_data['left']['left_pad_pos'])
        left_right_pad_error = np.linalg.norm(h1_data['left']['right_pad_pos'] - fg_data['left']['right_pad_pos'])
        print(f"\nLeft gripper pads:")
        print(f"  Left pad error:  {left_left_pad_error*1000:.3f}mm")
        print(f"  Right pad error: {left_right_pad_error*1000:.3f}mm")
        
        right_left_pad_error = np.linalg.norm(h1_data['right']['left_pad_pos'] - fg_data['right']['left_pad_pos'])
        right_right_pad_error = np.linalg.norm(h1_data['right']['right_pad_pos'] - fg_data['right']['right_pad_pos'])
        print(f"Right gripper pads:")
        print(f"  Left pad error:  {right_left_pad_error*1000:.3f}mm")
        print(f"  Right pad error: {right_right_pad_error*1000:.3f}mm")
        
        # Compare opening widths
        print(f"\nGripper opening widths:")
        print(f"  H1 left:  {h1_data['left']['opening_width']*1000:.1f}mm")
        print(f"  FG left:  {fg_data['left']['opening_width']*1000:.1f}mm")
        print(f"  H1 right: {h1_data['right']['opening_width']*1000:.1f}mm")
        print(f"  FG right: {fg_data['right']['opening_width']*1000:.1f}mm")
        
        # Compare gripper control values
        print(f"\nGripper control values (actuator):")
        h1_left_ctrl = h1_data['left']['gripper_ctrl']
        fg_left_ctrl = fg_data['left']['gripper_ctrl']
        h1_right_ctrl = h1_data['right']['gripper_ctrl']
        fg_right_ctrl = fg_data['right']['gripper_ctrl']
        
        # For display, show if they're open or closed
        def classify_ctrl(val):
            if val < 128:
                return f'{val:.0f} (open)'
            else:
                return f'{val:.0f} (closed)'
        
        print(f"  H1 left:  {classify_ctrl(h1_left_ctrl)}")
        print(f"  FG left:  {classify_ctrl(fg_left_ctrl)}")
        print(f"  H1 right: {classify_ctrl(h1_right_ctrl)}")
        print(f"  FG right: {classify_ctrl(fg_right_ctrl)}")
        
        max_base_error = max(max_base_error, left_base_error, right_base_error)
        max_tip_error = max(max_tip_error, 
                           left_left_pad_error, left_right_pad_error,
                           right_left_pad_error, right_right_pad_error)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Max gripper base error: {max_base_error*1000:.3f}mm")
    print(f"Max fingertip error: {max_tip_error*1000:.3f}mm")
    
    # Check success
    h1_success = any(d['success'] for d in h1_gripper_data)
    fg_success = any(d['success'] for d in fg_gripper_data)
    
    print(f"\nH1 task success: {'✅ Yes' if h1_success else '❌ No'}")
    print(f"Floating gripper task success: {'✅ Yes' if fg_success else '❌ No'}")
    
    if max_base_error < 0.001:  # Less than 1mm
        print("\n✨ EXCELLENT: Gripper bases match within 1mm!")
    elif max_base_error < 0.01:  # Less than 10mm
        print("\n✅ GOOD: Gripper bases match within 10mm")
    else:
        print(f"\n⚠️ WARNING: Large gripper base mismatch ({max_base_error*1000:.1f}mm)")


if __name__ == "__main__":
    compare_demos()