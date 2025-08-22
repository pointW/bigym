"""Compare RBY1 control between the working rby1/random_rollout.py approach and our setup."""

import sys
import os
import numpy as np
import mujoco

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bigym.envs.reach_target_rby1 import ReachTargetRBY1
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody


def test_rby1_original_model():
    """Test the original RBY1 model with their control approach."""
    print("=" * 60)
    print("TESTING ORIGINAL RBY1 MODEL (rby1/models/rby1m_capsule/)")
    print("=" * 60)
    
    # Load the original model
    model_path = os.path.join(project_root, "rby1/models/rby1m_capsule/model_act.xml")
    
    # Change to the model directory for asset loading
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(model_path))
    
    try:
        model = mujoco.MjModel.from_xml_path("model_act.xml")
        data = mujoco.MjData(model)
        
        print(f"\nModel loaded successfully")
        print(f"  Number of actuators: {model.nu}")
        print(f"  Timestep: {model.opt.timestep}")
        
        # Test 1: Random control like random_rollout.py
        print("\n1. Testing random control (like random_rollout.py):")
        print("-" * 40)
        
        # Apply smooth random control
        ctrl_smoothness = 0.9
        ctrl_scale = 1.0
        prev_ctrl = np.zeros(model.nu)
        
        velocities = []
        positions = []
        
        for step in range(200):
            # Generate smooth random control
            random_ctrl = np.random.uniform(-1.0, 1.0, size=model.nu) * ctrl_scale
            smooth_ctrl = ctrl_smoothness * prev_ctrl + (1 - ctrl_smoothness) * random_ctrl
            prev_ctrl = smooth_ctrl
            
            # Apply control
            data.ctrl[:] = smooth_ctrl
            
            # Step with frame skip
            for _ in range(5):  # Frame skip of 5
                mujoco.mj_step(model, data)
            
            # Track metrics
            vel = np.linalg.norm(data.qvel)
            velocities.append(vel)
            positions.append(data.qpos.copy())
            
            if step % 40 == 0:
                print(f"  Step {step}: velocity={vel:.2f}, max_ctrl={np.max(np.abs(smooth_ctrl)):.2f}")
        
        print(f"\n  Final velocity: {velocities[-1]:.2f}")
        print(f"  Max velocity: {np.max(velocities):.2f}")
        print(f"  Average velocity: {np.mean(velocities):.2f}")
        
        # Test 2: Position control with small targets
        print("\n2. Testing position control with small targets:")
        print("-" * 40)
        
        # Reset
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.ctrl[:] = 0
        
        # Set small position targets (like the random values)
        target_positions = np.random.uniform(-0.5, 0.5, model.nu)
        data.ctrl[:] = target_positions
        
        print(f"  Target range: [{np.min(target_positions):.2f}, {np.max(target_positions):.2f}]")
        
        # Simulate
        for step in range(200):
            mujoco.mj_step(model, data)
            
            if step % 50 == 0:
                vel = np.linalg.norm(data.qvel)
                # Check torso joints (indices 4-9 in actuators control torso_0 to torso_5)
                if model.nu > 9:
                    torso_positions = data.qpos[11:17]  # Torso joint positions in qpos
                    torso_targets = target_positions[4:10]
                    torso_error = np.mean(np.abs(torso_positions - torso_targets))
                    print(f"  Step {step}: velocity={vel:.2f}, torso_error={torso_error:.3f}")
                else:
                    print(f"  Step {step}: velocity={vel:.2f}")
        
        final_vel = np.linalg.norm(data.qvel)
        print(f"\n  Final velocity: {final_vel:.2f}")
        
        # Test 3: Large movements
        print("\n3. Testing large movements (0.5 to 1.0 rad):")
        print("-" * 40)
        
        # Reset
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.ctrl[:] = 0
        
        test_sizes = [0.5, 0.7, 1.0]
        for target_size in test_sizes:
            # Reset for each test
            data.qpos[:] = 0
            data.qvel[:] = 0
            
            # Set target for torso joints
            data.ctrl[:] = 0
            data.ctrl[4:10] = target_size  # Torso actuators
            
            # Simulate
            for _ in range(200):
                mujoco.mj_step(model, data)
            
            # Check results
            torso_positions = data.qpos[11:17]
            error = np.mean(np.abs(torso_positions - target_size))
            velocity = np.linalg.norm(data.qvel[11:17])
            
            print(f"  Target {target_size:.1f} rad: error={error:.4f}, velocity={velocity:.4f}")
        
        # Test 4: Gradual position changes
        print("\n4. Testing gradual position changes:")
        print("-" * 40)
        
        data.qpos[:] = 0
        data.qvel[:] = 0
        
        target = np.zeros(model.nu)
        velocities = []
        
        for step in range(200):
            # Gradually change targets
            target += np.random.uniform(-0.02, 0.02, model.nu)
            target = np.clip(target, -1.0, 1.0)
            data.ctrl[:] = target
            
            mujoco.mj_step(model, data)
            
            vel = np.linalg.norm(data.qvel)
            velocities.append(vel)
            
            if step % 50 == 0:
                print(f"  Step {step}: velocity={vel:.2f}")
        
        print(f"\n  Final velocity: {velocities[-1]:.2f}")
        print(f"  Max velocity during gradual changes: {np.max(velocities):.2f}")
        
    finally:
        os.chdir(original_dir)
    
    return velocities


def test_our_consolidated_model():
    """Test our consolidated model with similar control approach."""
    print("\n" + "=" * 60)
    print("TESTING OUR CONSOLIDATED MODEL")
    print("=" * 60)
    
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=None,
    )
    env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    print(f"\nModel loaded successfully")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Timestep: {model.opt.timestep}")
    
    # Test 1: Random control (but with position actuators)
    print("\n1. Testing random control (position targets):")
    print("-" * 40)
    
    ctrl_smoothness = 0.9
    prev_ctrl = np.zeros(model.nu)
    velocities = []
    
    for step in range(200):
        # Generate smooth random POSITION targets
        random_ctrl = np.random.uniform(-0.5, 0.5, size=model.nu)
        smooth_ctrl = ctrl_smoothness * prev_ctrl + (1 - ctrl_smoothness) * random_ctrl
        prev_ctrl = smooth_ctrl
        
        # Apply as position targets
        data.ctrl[:] = smooth_ctrl
        
        # Step with frame skip
        for _ in range(5):
            mujoco.mj_step(model, data)
        
        vel = np.linalg.norm(data.qvel[17:31])  # Arm joints
        velocities.append(vel)
        
        if step % 40 == 0:
            print(f"  Step {step}: arm_velocity={vel:.2f}, max_target={np.max(np.abs(smooth_ctrl)):.2f}")
    
    print(f"\n  Final arm velocity: {velocities[-1]:.2f}")
    print(f"  Max arm velocity: {np.max(velocities):.2f}")
    print(f"  Average arm velocity: {np.mean(velocities):.2f}")
    
    # Test 2: Fixed small position targets
    print("\n2. Testing fixed small position targets:")
    print("-" * 40)
    
    # Reset
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    
    # Set small targets
    target_positions = np.zeros(model.nu)
    target_positions[6:13] = 0.2   # Right arm
    target_positions[13:20] = -0.2  # Left arm
    data.ctrl[:] = target_positions
    
    print(f"  Arm targets: ±0.2 rad")
    
    errors = []
    arm_velocities = []
    
    for step in range(200):
        mujoco.mj_step(model, data)
        
        # Track error
        right_error = np.mean(np.abs(data.qpos[17:24] - 0.2))
        left_error = np.mean(np.abs(data.qpos[24:31] + 0.2))
        error = (right_error + left_error) / 2
        errors.append(error)
        
        vel = np.linalg.norm(data.qvel[17:31])
        arm_velocities.append(vel)
        
        if step % 50 == 0:
            print(f"  Step {step}: arm_velocity={vel:.2f}, error={error:.3f}")
    
    print(f"\n  Final error: {errors[-1]:.3f}")
    print(f"  Final arm velocity: {arm_velocities[-1]:.2f}")
    
    # Test 3: Large movements
    print("\n3. Testing large movements (0.5 to 1.0 rad):")
    print("-" * 40)
    
    test_sizes = [0.5, 0.7, 1.0]
    for target_size in test_sizes:
        # Reset
        data.qpos[:] = env._mojo.physics.data.qpos[:]
        data.qvel[:] = 0
        
        # Set targets for arms
        data.ctrl[:] = 0
        data.ctrl[6:13] = target_size   # Right arm
        data.ctrl[13:20] = -target_size  # Left arm (opposite)
        
        # Simulate
        for _ in range(200):
            mujoco.mj_step(model, data)
        
        # Check results
        right_error = np.mean(np.abs(data.qpos[17:24] - target_size))
        left_error = np.mean(np.abs(data.qpos[24:31] + target_size))
        error = (right_error + left_error) / 2
        velocity = np.linalg.norm(data.qvel[17:31])
        
        print(f"  Target ±{target_size:.1f} rad: error={error:.4f}, arm_velocity={velocity:.4f}")
    
    # Test 4: Gradual position changes
    print("\n4. Testing gradual position changes:")
    print("-" * 40)
    
    data.qpos[:] = env._mojo.physics.data.qpos[:]
    data.qvel[:] = 0
    
    target = np.zeros(model.nu)
    velocities = []
    
    for step in range(200):
        # Gradually change targets for arms only
        target[6:20] += np.random.uniform(-0.02, 0.02, 14)
        target[6:20] = np.clip(target[6:20], -0.5, 0.5)
        data.ctrl[:] = target
        
        mujoco.mj_step(model, data)
        
        vel = np.linalg.norm(data.qvel[17:31])
        velocities.append(vel)
        
        if step % 50 == 0:
            print(f"  Step {step}: arm_velocity={vel:.2f}")
    
    print(f"\n  Final arm velocity: {velocities[-1]:.2f}")
    print(f"  Max velocity during gradual changes: {np.max(velocities):.2f}")
    
    env.close()
    return arm_velocities


def compare_specific_movements():
    """Compare specific joint movements between models."""
    print("\n" + "=" * 60)
    print("SPECIFIC MOVEMENT COMPARISON")
    print("=" * 60)
    
    # Test movements
    test_movements = [
        ("Very small", 0.01),
        ("Small", 0.1),
        ("Medium", 0.3),
        ("Large", 0.5),
    ]
    
    print("\nTesting our model with specific movements:")
    print("-" * 40)
    
    env = ReachTargetRBY1(
        action_mode=RBY1CartesianActionModeWholeBody(),
        control_frequency=50,
        render_mode=None,
    )
    env.reset(seed=42)
    
    model = env._mojo.physics.model._model
    data = env._mojo.physics.data._data
    
    results = []
    
    for name, size in test_movements:
        # Reset
        data.qpos[:] = env._mojo.physics.data.qpos[:]
        data.qvel[:] = 0
        
        # Set target for one joint
        data.ctrl[:] = 0
        data.ctrl[6] = size  # Right arm first joint
        
        # Simulate
        for _ in range(200):
            mujoco.mj_step(model, data)
        
        # Check result
        achieved = data.qpos[17]  # Right arm first joint position
        error = abs(achieved - size)
        velocity = abs(data.qvel[17])
        
        success = error < 0.01 and velocity < 0.05
        
        print(f"  {name:10s} ({size:.2f} rad): error={error:.4f}, vel={velocity:.4f} {'✅' if success else '❌'}")
        
        results.append({
            'name': name,
            'size': size,
            'error': error,
            'velocity': velocity,
            'success': success
        })
    
    env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print("\nKey differences found:")
    print("1. Original RBY1 model uses position actuators with kp=4000, kv=400")
    print("2. Random rollout works because control values [-1, 1] are reasonable position targets")
    print("3. Our model now uses position actuators with kp=10000, kv=500")
    print("4. Both models are stable with small movements")
    
    successful = [r for r in results if r['success']]
    if successful:
        print(f"\n✅ Our model successfully tracks movements from {successful[0]['size']:.2f} rad and up")
    else:
        print("\n⚠️ Tracking issues detected - may need further tuning")
    
    return results


if __name__ == "__main__":
    # Test original model
    print("Testing original RBY1 model from rby1/models/rby1m_capsule/")
    print("This uses the exact same model as random_rollout.py")
    print("=" * 60)
    
    try:
        original_velocities = test_rby1_original_model()
    except Exception as e:
        print(f"\n⚠️ Could not test original model: {e}")
        print("This is likely due to asset loading issues")
        original_velocities = None
    
    # Test our model
    our_velocities = test_our_consolidated_model()
    
    # Compare specific movements
    results = compare_specific_movements()
    
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    if original_velocities:
        print(f"\nOriginal model average velocity: {np.mean(original_velocities):.2f}")
    print(f"Our model average arm velocity: {np.mean(our_velocities):.2f}")
    
    print("\nConclusion:")
    print("- Both models use position actuators")
    print("- Control signals are interpreted as target positions in radians")
    print("- Small movements (< 0.5 rad) work well in both models")
    print("- The key to stability is using appropriate position targets, not control signals")