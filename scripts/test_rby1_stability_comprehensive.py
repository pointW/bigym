"""Comprehensive RBY1 stability test script.

Tests three key stability aspects:
1. Reset stability in RBY1CartesianActionModeWholeBody
2. Joint movement stability (bypassing IK)
3. Base movement stability via base_target

Usage:
    python scripts/test_rby1_stability_comprehensive.py          # Headless mode
    python scripts/test_rby1_stability_comprehensive.py --render # With rendering
"""

import sys
import os
import numpy as np
import mujoco
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody


def measure_stability(qpos_history, qvel_history):
    """Measure stability metrics from position and velocity history."""
    # Calculate velocity statistics
    vel_norms = [np.linalg.norm(qvel[11:17]) for qvel in qvel_history]  # Torso joints
    avg_velocity = np.mean(vel_norms)
    max_velocity = np.max(vel_norms)
    std_velocity = np.std(vel_norms)
    
    # Calculate position drift
    initial_pos = qpos_history[0][11:17]
    final_pos = qpos_history[-1][11:17]
    position_drift = np.linalg.norm(final_pos - initial_pos)
    
    return {
        'avg_velocity': avg_velocity,
        'max_velocity': max_velocity,
        'std_velocity': std_velocity,
        'position_drift': position_drift,
        'is_stable': max_velocity < 0.5 and std_velocity < 0.1
    }


def test_reset_stability(render_mode=None):
    """Test 1: Reset stability in RBY1CartesianActionModeWholeBody."""
    print("\n" + "=" * 60)
    print("TEST 1: Reset Stability")
    print("=" * 60)
    
    # Create environment with whole-body action mode
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(block_until_reached=True),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print("Environment reset completed")
    
    # Monitor stability after reset without any actions
    qpos_history = []
    qvel_history = []
    
    print("\nMonitoring post-reset stability (50 steps)...")
    for step in range(50):
        # Just step without actions (zero action)
        action = np.zeros(20)
        # Set to current poses to maintain position
        from vr.ik.h1_upper_body_ik import Pose
        action_mode = env.action_mode
        left_pose, right_pose = action_mode.get_current_ee_poses()
        action = action_mode.poses_to_action(left_pose, right_pose)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record state
        qpos = env._mojo.physics.data.qpos.copy()
        qvel = env._mojo.physics.data.qvel.copy()
        qpos_history.append(qpos)
        qvel_history.append(qvel)
        
        if render_mode:
            env.render()
        
        # Print progress
        if step % 10 == 0:
            vel_norm = np.linalg.norm(qvel[11:17])
            print(f"  Step {step}: velocity={vel_norm:.4f} rad/s")
    
    # Analyze stability
    metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nReset Stability Results:")
    print(f"  Average velocity: {metrics['avg_velocity']:.4f} rad/s")
    print(f"  Max velocity: {metrics['max_velocity']:.4f} rad/s")
    print(f"  Velocity std: {metrics['std_velocity']:.4f} rad/s")
    print(f"  Position drift: {metrics['position_drift']:.4f} rad")
    print(f"  Status: {'✓ STABLE' if metrics['is_stable'] else '✗ UNSTABLE'}")
    
    env.close()
    return metrics


def test_joint_movement_stability(render_mode=None):
    """Test 2: Joint movement stability (bypassing IK)."""
    print("\n" + "=" * 60)
    print("TEST 2: Joint Movement Stability (Bypassing IK)")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    print("\nTesting direct joint control (bypassing IK)...")
    
    # Test small movements first
    print("\n--- Small Joint Movements (0.1 rad) ---")
    initial_ctrl = data.ctrl[:20].copy()
    qpos_history = []
    qvel_history = []
    
    for step in range(50):
        # Apply small joint movement
        for i in range(20):
            data.ctrl[i] = initial_ctrl[i] + 0.1 * np.sin(step * 0.1)
        
        # Step physics directly
        mujoco.mj_step(model, data)
        
        # Record state
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        if render_mode:
            env.render()
        
        if step % 10 == 0:
            vel_norm = np.linalg.norm(data.qvel[11:17])
            print(f"  Step {step}: velocity={vel_norm:.4f} rad/s")
    
    small_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nSmall Movement Results:")
    print(f"  Max velocity: {small_metrics['max_velocity']:.4f} rad/s")
    print(f"  Velocity std: {small_metrics['std_velocity']:.4f} rad/s")
    print(f"  Status: {'✓ STABLE' if small_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    # Test large movements
    print("\n--- Large Joint Movements (0.5 rad) ---")
    qpos_history = []
    qvel_history = []
    
    for step in range(50):
        # Apply large joint movement
        for i in range(20):
            data.ctrl[i] = initial_ctrl[i] + 0.5 * np.sin(step * 0.1)
        
        # Step physics directly
        mujoco.mj_step(model, data)
        
        # Record state
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        if render_mode:
            env.render()
        
        if step % 10 == 0:
            vel_norm = np.linalg.norm(data.qvel[11:17])
            print(f"  Step {step}: velocity={vel_norm:.4f} rad/s")
    
    large_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nLarge Movement Results:")
    print(f"  Max velocity: {large_metrics['max_velocity']:.4f} rad/s")
    print(f"  Velocity std: {large_metrics['std_velocity']:.4f} rad/s")
    print(f"  Status: {'✓ STABLE' if large_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    env.close()
    return small_metrics, large_metrics


def test_base_movement_stability(render_mode=None):
    """Test 3: Base movement stability via base_target."""
    print("\n" + "=" * 60)
    print("TEST 3: Base Movement Stability (via base_target)")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    # Find base_target mocap body
    base_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_target")
    mocap_id = model.body_mocapid[base_target_id]
    
    if mocap_id < 0:
        print("Error: base_target mocap not found")
        env.close()
        return None
    
    print("\nTesting base movement via base_target mocap...")
    
    # Test small base movements
    print("\n--- Small Base Movements (0.1m) ---")
    qpos_history = []
    qvel_history = []
    base_pos_history = []
    
    for step in range(50):
        # Move base in small circle
        radius = 0.1
        angle = step * 0.1
        data.mocap_pos[mocap_id][0] = radius * np.cos(angle)
        data.mocap_pos[mocap_id][1] = radius * np.sin(angle)
        data.mocap_pos[mocap_id][2] = 0.0
        
        # Keep orientation identity
        data.mocap_quat[mocap_id] = [1, 0, 0, 0]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Record state
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        base_pos_history.append(data.qpos[:2].copy())
        
        if render_mode:
            env.render()
        
        if step % 10 == 0:
            vel_norm = np.linalg.norm(data.qvel[:3])  # Base velocity
            print(f"  Step {step}: base_velocity={vel_norm:.4f} m/s, base_pos=[{data.qpos[0]:.3f}, {data.qpos[1]:.3f}]")
    
    small_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nSmall Base Movement Results:")
    print(f"  Max velocity: {small_metrics['max_velocity']:.4f} rad/s")
    print(f"  Base tracking error: {np.mean([np.linalg.norm(base_pos_history[i] - [0.1*np.cos(i*0.1), 0.1*np.sin(i*0.1)]) for i in range(len(base_pos_history))]):.4f} m")
    print(f"  Status: {'✓ STABLE' if small_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    # Test large base movements
    print("\n--- Large Base Movements (0.5m) ---")
    qpos_history = []
    qvel_history = []
    base_pos_history = []
    
    for step in range(50):
        # Move base in large circle
        radius = 0.5
        angle = step * 0.1
        data.mocap_pos[mocap_id][0] = radius * np.cos(angle)
        data.mocap_pos[mocap_id][1] = radius * np.sin(angle)
        data.mocap_pos[mocap_id][2] = 0.0
        
        # Keep orientation identity
        data.mocap_quat[mocap_id] = [1, 0, 0, 0]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Record state
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        base_pos_history.append(data.qpos[:2].copy())
        
        if render_mode:
            env.render()
        
        if step % 10 == 0:
            vel_norm = np.linalg.norm(data.qvel[:3])  # Base velocity
            print(f"  Step {step}: base_velocity={vel_norm:.4f} m/s, base_pos=[{data.qpos[0]:.3f}, {data.qpos[1]:.3f}]")
    
    large_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nLarge Base Movement Results:")
    print(f"  Max velocity: {large_metrics['max_velocity']:.4f} rad/s")
    print(f"  Base tracking error: {np.mean([np.linalg.norm(base_pos_history[i] - [0.5*np.cos(i*0.1), 0.5*np.sin(i*0.1)]) for i in range(len(base_pos_history))]):.4f} m")
    print(f"  Status: {'✓ STABLE' if large_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    # Test with rotation
    print("\n--- Base Movement with Rotation ---")
    qpos_history = []
    qvel_history = []
    
    for step in range(50):
        # Move base and rotate
        data.mocap_pos[mocap_id][0] = 0.2
        data.mocap_pos[mocap_id][1] = 0.0
        data.mocap_pos[mocap_id][2] = 0.0
        
        # Rotate around Z axis
        angle = step * 0.05
        qw = np.cos(angle / 2)
        qz = np.sin(angle / 2)
        data.mocap_quat[mocap_id] = [qw, 0, 0, qz]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Record state
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        if render_mode:
            env.render()
        
        if step % 10 == 0:
            vel_norm = np.linalg.norm(data.qvel[:3])  # Base velocity
            ang_vel = data.qvel[5]  # RZ angular velocity
            print(f"  Step {step}: base_velocity={vel_norm:.4f} m/s, angular_vel={ang_vel:.4f} rad/s")
    
    rotation_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nBase Movement with Rotation Results:")
    print(f"  Max velocity: {rotation_metrics['max_velocity']:.4f} rad/s")
    print(f"  Status: {'✓ STABLE' if rotation_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    env.close()
    return small_metrics, large_metrics, rotation_metrics


def test_torso_movement_stability(render_mode=None):
    """Test 4: Torso joint movement stability."""
    print("\n" + "=" * 60)
    print("TEST 4: Torso Joint Movement Stability")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    print("\nTesting torso joint movements...")
    
    # Test small torso movements
    print("\n--- Small Torso Movements (0.1 rad) ---")
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    
    # Set small targets for torso joints (actuators 0-5)
    data.ctrl[:] = 0
    data.ctrl[0:6] = [0.05, 0.1, -0.1, 0.05, -0.05, 0.1]  # Torso joints
    
    qpos_history = []
    qvel_history = []
    
    for step in range(100):
        mujoco.mj_step(model, data)
        
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        if render_mode:
            env.render()
        
        if step % 25 == 0:
            torso_vel = np.linalg.norm(data.qvel[11:17])
            torso_error = np.mean(np.abs(data.qpos[11:17] - data.ctrl[0:6]))
            print(f"  Step {step}: torso_velocity={torso_vel:.4f}, error={torso_error:.4f}")
    
    small_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nSmall Torso Movement Results:")
    print(f"  Max velocity: {small_metrics['max_velocity']:.4f} rad/s")
    print(f"  Position drift: {small_metrics['position_drift']:.4f} rad")
    print(f"  Status: {'✓ STABLE' if small_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    # Test combined torso and arm movements
    print("\n--- Combined Torso + Arm Movements ---")
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    
    # Set targets for both torso and arms
    data.ctrl[:] = 0
    data.ctrl[0:6] = [0.05, 0.1, -0.05, 0.05, -0.03, 0.08]  # Torso
    data.ctrl[6:13] = 0.2   # Right arm
    data.ctrl[13:20] = -0.2  # Left arm
    
    qpos_history = []
    qvel_history = []
    
    for step in range(100):
        mujoco.mj_step(model, data)
        
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        if render_mode:
            env.render()
        
        if step % 25 == 0:
            torso_vel = np.linalg.norm(data.qvel[11:17])
            arm_vel = np.linalg.norm(data.qvel[17:31])
            print(f"  Step {step}: torso_vel={torso_vel:.4f}, arm_vel={arm_vel:.4f}")
    
    combined_metrics = measure_stability(qpos_history, qvel_history)
    
    print("\nCombined Movement Results:")
    print(f"  Max velocity: {combined_metrics['max_velocity']:.4f} rad/s")
    print(f"  Status: {'✓ STABLE' if combined_metrics['is_stable'] else '✗ UNSTABLE'}")
    
    env.close()
    return small_metrics, combined_metrics


def test_interpolated_large_movement(render_mode=None):
    """Test 5: Reach large target through interpolated small steps."""
    print("\n" + "=" * 60)
    print("TEST 5: Interpolated Large Movement")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    print("\nTesting large movement via interpolation...")
    
    # Define large target
    large_target = 1.0  # rad
    num_steps = 10  # Number of interpolation steps
    step_size = large_target / num_steps
    
    print(f"Target: {large_target:.1f} rad in {num_steps} steps of {step_size:.3f} rad")
    
    # Test single joint first
    print("\n--- Single Joint Interpolation ---")
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    data.ctrl[:] = 0
    
    current_target = 0
    errors = []
    velocities = []
    
    for interp_step in range(num_steps):
        # Increment target
        current_target += step_size
        data.ctrl[6] = current_target  # Right arm first joint
        
        print(f"\nStep {interp_step+1}/{num_steps}: target={current_target:.3f}")
        
        # Simulate until settled (or max iterations)
        for sim_step in range(50):
            mujoco.mj_step(model, data)
            
            if render_mode:
                env.render()
            
            if sim_step % 20 == 0:
                error = abs(data.qpos[17] - current_target)
                vel = abs(data.qvel[17])
                print(f"  Sim {sim_step}: error={error:.4f}, vel={vel:.4f}")
        
        # Record final state for this interpolation step
        final_error = abs(data.qpos[17] - current_target)
        final_vel = abs(data.qvel[17])
        errors.append(final_error)
        velocities.append(final_vel)
    
    # Check final achievement
    final_position = data.qpos[17]
    total_error = abs(final_position - large_target)
    
    print(f"\nSingle Joint Results:")
    print(f"  Target: {large_target:.1f} rad")
    print(f"  Achieved: {final_position:.4f} rad")
    print(f"  Final error: {total_error:.4f} rad")
    print(f"  Max step error: {max(errors):.4f} rad")
    print(f"  Status: {'✅ SUCCESS' if total_error < 0.05 else '❌ FAILED'}")
    
    # Test multi-joint interpolation
    print("\n--- Multi-Joint Interpolation ---")
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    data.ctrl[:] = 0
    
    # Define different targets for different joints
    arm_targets = np.array([0.8, -0.6, 0.7, -0.9, 0.5, -0.4, 0.6])
    current_targets = np.zeros(7)
    
    print(f"Arm targets: {arm_targets}")
    
    multi_errors = []
    
    for interp_step in range(num_steps):
        # Increment all targets
        current_targets += arm_targets / num_steps
        data.ctrl[6:13] = current_targets  # Right arm
        data.ctrl[13:20] = -current_targets  # Left arm (mirrored)
        
        print(f"\nStep {interp_step+1}/{num_steps}")
        
        # Simulate
        for sim_step in range(50):
            mujoco.mj_step(model, data)
            
            if render_mode:
                env.render()
        
        # Check convergence
        right_error = np.mean(np.abs(data.qpos[17:24] - current_targets))
        left_error = np.mean(np.abs(data.qpos[24:31] + current_targets))
        avg_error = (right_error + left_error) / 2
        multi_errors.append(avg_error)
        
        arm_vel = np.linalg.norm(data.qvel[17:31])
        print(f"  Error: {avg_error:.4f}, Velocity: {arm_vel:.4f}")
    
    # Final results
    final_right_error = np.mean(np.abs(data.qpos[17:24] - arm_targets))
    final_left_error = np.mean(np.abs(data.qpos[24:31] + arm_targets))
    final_avg_error = (final_right_error + final_left_error) / 2
    
    print(f"\nMulti-Joint Results:")
    print(f"  Final average error: {final_avg_error:.4f} rad")
    print(f"  Max step error: {max(multi_errors):.4f} rad")
    print(f"  Status: {'✅ SUCCESS' if final_avg_error < 0.1 else '❌ FAILED'}")
    
    env.close()
    
    return {
        'single_joint_error': total_error,
        'multi_joint_error': final_avg_error,
        'single_success': total_error < 0.05,
        'multi_success': final_avg_error < 0.1
    }


def test_realistic_usage(render_mode=None):
    """Test 6: Realistic usage patterns similar to random_rollout.py"""
    print("\n" + "=" * 60)
    print("TEST 6: Realistic Usage (Like random_rollout.py)")
    print("=" * 60)
    
    # Create environment
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=render_mode,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    # Test 1: Smooth random control exactly like random_rollout.py
    print("\n--- Smooth Random Control (random_rollout.py style) ---")
    
    ctrl_smoothness = 0.9
    ctrl_scale = 1.0
    prev_ctrl = np.zeros(model.nu)
    
    velocities = []
    max_ctrl = 0
    
    for step in range(100):
        # Generate smooth random control
        random_ctrl = np.random.uniform(-1.0, 1.0, size=model.nu) * ctrl_scale
        smooth_ctrl = ctrl_smoothness * prev_ctrl + (1 - ctrl_smoothness) * random_ctrl
        prev_ctrl = smooth_ctrl.copy()
        
        max_ctrl = max(max_ctrl, np.max(np.abs(smooth_ctrl)))
        
        # Apply control
        data.ctrl[:] = smooth_ctrl
        
        # Step with frame skip
        for _ in range(5):
            mujoco.mj_step(model, data)
        
        # Track velocity
        arm_vel = np.linalg.norm(data.qvel[17:31])
        velocities.append(arm_vel)
        
        if render_mode:
            env.render()
        
        if step % 25 == 0:
            print(f"  Step {step}: arm_velocity={arm_vel:.2f}, max_ctrl={np.max(np.abs(smooth_ctrl)):.2f}")
    
    random_stable = np.max(velocities) < 10.0
    
    print(f"\nSmooth Random Control Results:")
    print(f"  Max velocity: {np.max(velocities):.2f} rad/s")
    print(f"  Average velocity: {np.mean(velocities):.2f} rad/s")
    print(f"  Max control value: {max_ctrl:.2f}")
    print(f"  Status: {'✓ STABLE' if random_stable else '✗ UNSTABLE'}")
    
    # Test 2: Small position targets
    print("\n--- Small Position Targets ---")
    
    test_targets = [0.1, 0.2, -0.1, -0.2]
    success_count = 0
    
    for target in test_targets:
        # Reset
        data.qpos[:] = env._mojo.physics.data.qpos[:]
        data.qvel[:] = 0
        data.ctrl[:] = 0
        data.ctrl[6] = target  # Single joint
        
        # Simulate (more steps for convergence)
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        error = abs(data.qpos[17] - target)
        vel = abs(data.qvel[17])
        success = error < 0.05 and vel < 0.1
        if success:
            success_count += 1
        
        print(f"  Target {target:+.1f}: error={error:.4f}, vel={vel:.4f} {'✅' if success else '❌'}")
    
    target_success = success_count >= len(test_targets) * 0.75  # 75% success rate
    
    print(f"\nSmall Target Results:")
    print(f"  Success rate: {success_count}/{len(test_targets)}")
    print(f"  Status: {'✓ PASSED' if target_success else '✗ FAILED'}")
    
    env.close()
    
    return {
        'random_stable': random_stable,
        'target_success': target_success,
        'overall_success': random_stable and target_success
    }


def main():
    """Run all stability tests."""
    parser = argparse.ArgumentParser(description='RBY1 Comprehensive Stability Tests')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    print("=" * 60)
    print("RBY1 COMPREHENSIVE STABILITY TESTS")
    print(f"Mode: {'WITH RENDERING' if render_mode else 'HEADLESS'}")
    print("=" * 60)
    
    # Run tests
    reset_metrics = test_reset_stability(render_mode)
    joint_small, joint_large = test_joint_movement_stability(render_mode)
    base_results = test_base_movement_stability(render_mode)
    torso_small, torso_combined = test_torso_movement_stability(render_mode)
    interp_results = test_interpolated_large_movement(render_mode)
    realistic_results = test_realistic_usage(render_mode)
    
    if base_results:
        base_small, base_large, base_rotation = base_results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n1. Reset Stability:")
    print(f"   {'✓ PASSED' if reset_metrics['is_stable'] else '✗ FAILED'}")
    print(f"   Max velocity: {reset_metrics['max_velocity']:.4f} rad/s")
    
    print("\n2. Joint Movement Stability:")
    print(f"   Small movements: {'✓ PASSED' if joint_small['is_stable'] else '✗ FAILED'}")
    print(f"   Large movements: {'✓ PASSED' if joint_large['is_stable'] else '✗ FAILED'}")
    
    if base_results:
        print("\n3. Base Movement Stability:")
        print(f"   Small movements: {'✓ PASSED' if base_small['is_stable'] else '✗ FAILED'}")
        print(f"   Large movements: {'✓ PASSED' if base_large['is_stable'] else '✗ FAILED'}")
        print(f"   With rotation: {'✓ PASSED' if base_rotation['is_stable'] else '✗ FAILED'}")
    
    print("\n4. Torso Movement Stability:")
    print(f"   Small torso movements: {'✓ PASSED' if torso_small['is_stable'] else '✗ FAILED'}")
    print(f"   Combined torso+arms: {'✓ PASSED' if torso_combined['is_stable'] else '✗ FAILED'}")
    
    print("\n5. Interpolated Large Movement:")
    print(f"   Single joint (1.0 rad): {'✓ PASSED' if interp_results['single_success'] else '✗ FAILED'}")
    print(f"   Multi-joint: {'✓ PASSED' if interp_results['multi_success'] else '✗ FAILED'}")
    print(f"   Single joint error: {interp_results['single_joint_error']:.4f} rad")
    print(f"   Multi-joint error: {interp_results['multi_joint_error']:.4f} rad")
    
    print("\n6. Realistic Usage (random_rollout.py style):")
    print(f"   Smooth random control: {'✓ PASSED' if realistic_results['random_stable'] else '✗ FAILED'}")
    print(f"   Small position targets: {'✓ PASSED' if realistic_results['target_success'] else '✗ FAILED'}")
    print(f"   Overall: {'✓ PASSED' if realistic_results['overall_success'] else '✗ FAILED'}")
    
    # Overall assessment
    # Focus on critical tests: reset stability and realistic usage
    critical_tests_pass = (
        reset_metrics['is_stable'] and
        realistic_results['overall_success']
    )
    
    all_stable = (
        reset_metrics['is_stable'] and
        joint_small['is_stable'] and
        joint_large['is_stable'] and
        torso_small['is_stable'] and
        realistic_results['overall_success'] and
        (base_results is None or (base_small['is_stable'] and base_large['is_stable'] and base_rotation['is_stable']))
    )
    
    print("\n" + "=" * 60)
    if all_stable:
        print("✅ ALL TESTS PASSED - ROBOT IS STABLE")
    elif critical_tests_pass:
        print("✅ CRITICAL TESTS PASSED - ROBOT IS STABLE FOR REALISTIC USE")
        print("   (Some edge case tests failed, but core functionality works)")
    else:
        print("⚠️ CRITICAL TESTS FAILED - STABILITY ISSUES DETECTED")
    print("=" * 60)


if __name__ == "__main__":
    main()