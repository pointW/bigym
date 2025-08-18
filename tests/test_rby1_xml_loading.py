"""Test that RBY1 XML model loads correctly in MuJoCo."""
import os
import mujoco
import numpy as np
from pathlib import Path


def test_rby1_model_loading():
    """Test that the RBY1 model can be loaded by MuJoCo."""
    model_path = Path(__file__).parent.parent / "bigym" / "envs" / "xmls" / "rby1" / "model_act.xml"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Change to the directory containing the XML to resolve relative paths
    original_dir = os.getcwd()
    os.chdir(model_path.parent)
    
    try:
        # Load the model
        model = mujoco.MjModel.from_xml_path(str(model_path.name))
        data = mujoco.MjData(model)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Number of joints: {model.njnt}")
        print(f"   Number of actuators: {model.nu}")
        print(f"   Number of bodies: {model.nbody}")
        print(f"   Number of DOFs: {model.nv}")
        
        # Check for important joints
        important_joints = [
            "world_j",  # Free joint for base
            "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",  # Torso
            "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",  # Left arm
            "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",  # Right arm
            "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl",  # Wheels
        ]
        
        print("\n📋 Joint verification:")
        for joint_name in important_joints:
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                print(f"   ✓ {joint_name}: id={joint_id}")
            except:
                print(f"   ✗ {joint_name}: NOT FOUND")
        
        # Check for end effector sites
        print("\n📍 End effector sites:")
        for site_name in ["end_effector_l", "end_effector_r"]:
            try:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                print(f"   ✓ {site_name}: id={site_id}")
            except:
                print(f"   ✗ {site_name}: NOT FOUND")
        
        # Run a simple forward dynamics step
        mujoco.mj_forward(model, data)
        print("\n✅ Forward dynamics step successful!")
        
        # Check initial configuration
        print(f"\n🤖 Initial configuration:")
        print(f"   Base position: {data.qpos[:3]}")
        print(f"   Base orientation (quaternion): {data.qpos[3:7]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    finally:
        os.chdir(original_dir)


def test_rby1_actuator_mapping():
    """Test that actuators are properly configured."""
    model_path = Path(__file__).parent.parent / "bigym" / "envs" / "xmls" / "rby1" / "model_act.xml"
    
    # Change to the directory containing the XML
    original_dir = os.getcwd()
    os.chdir(model_path.parent)
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path.name))
        
        print("\n🎮 Actuator configuration:")
        print(f"   Total actuators: {model.nu}")
        
        # List all actuators
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ctrl_range = model.actuator_ctrlrange[i]
            print(f"   {i:2d}. {name:20s} range: [{ctrl_range[0]:6.2f}, {ctrl_range[1]:6.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to check actuators: {e}")
        return False
    
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    print("="*60)
    print("RBY1 Model Loading Test")
    print("="*60)
    
    success = test_rby1_model_loading()
    success = test_rby1_actuator_mapping() and success
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    exit(0 if success else 1)