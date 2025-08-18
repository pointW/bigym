"""Unit tests for RBY1 IK solver."""
import pytest
import numpy as np
import mujoco
from pathlib import Path
import os


# Check if mink is available
try:
    import mink
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False


@pytest.mark.skipif(not MINK_AVAILABLE, reason="Mink library not available")
class TestRBY1IKSolver:
    """Test RBY1 IK solver functionality."""
    
    @pytest.fixture
    def rby1_model_and_data(self):
        """Load RBY1 model and create data."""
        model_path = Path(__file__).parent.parent / "bigym" / "envs" / "xmls" / "rby1" / "model_act.xml"
        
        # Change to model directory for relative paths
        original_dir = os.getcwd()
        os.chdir(model_path.parent)
        
        try:
            model = mujoco.MjModel.from_xml_path(str(model_path.name))
            data = mujoco.MjData(model)
            yield model, data
        finally:
            os.chdir(original_dir)
    
    def test_ik_solver_initialization(self, rby1_model_and_data):
        """Test that IK solver initializes correctly."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        
        # Create solver
        solver = RBY1IK(model, data)
        
        # Check that solver is initialized
        assert solver.model is not None
        assert solver.data is not None
        
        # Check joint indices are set up
        assert hasattr(solver, 'torso_qpos_indices')
        assert hasattr(solver, 'left_arm_qpos_indices')
        assert hasattr(solver, 'right_arm_qpos_indices')
        
        # Check joint indices
        assert len(solver.torso_qpos_indices) == 6
        assert len(solver.left_arm_qpos_indices) == 7
        assert len(solver.right_arm_qpos_indices) == 7
    
    def test_forward_kinematics(self, rby1_model_and_data):
        """Test forward kinematics computation."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        solver = RBY1IK(model, data)
        
        # Set a known configuration
        qpos = np.zeros(model.nq)
        qpos[3] = 1.0  # Set quaternion w to 1 (identity rotation)
        
        # Get end effector positions
        left_pos = solver._get_site_position("end_effector_l", qpos)
        right_pos = solver._get_site_position("end_effector_r", qpos)
        
        # Check that positions are reasonable
        assert left_pos.shape == (3,)
        assert right_pos.shape == (3,)
        
        # Left and right should be symmetric (approximately)
        # when all joints are at zero
        assert np.abs(left_pos[1] + right_pos[1]) < 0.1  # Y should be opposite
        assert np.abs(left_pos[0] - right_pos[0]) < 0.1  # X should be similar
        assert np.abs(left_pos[2] - right_pos[2]) < 0.1  # Z should be similar
    
    def test_ik_solve_single_arm(self, rby1_model_and_data):
        """Test IK solving for single arm reaching."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        solver = RBY1IK(model, data)
        
        # Set initial configuration with proper base height
        initial_qpos = np.zeros(model.nq)
        initial_qpos[2] = 0.35  # Set proper base height
        initial_qpos[3] = 1.0  # Identity quaternion
        
        # Fixed base position and orientation
        base_pos = np.array([0.0, 0.0, 0.35])  # Proper base height
        base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Get current left end effector position with proper base height
        left_ee_pos = solver._get_site_position("end_effector_l", initial_qpos)
        
        # Create a target slightly offset from current position (smaller movement)
        target_pos = left_ee_pos + np.array([0.05, 0.0, 0.05])
        
        # Solve IK with fixed base
        solution, success, info = solver.solve(
            base_pos=base_pos,
            base_quat=base_quat,
            left_target_pos=target_pos,
            current_qpos=initial_qpos,
            max_iterations=100,
            tolerance=0.001  # Use 1mm tolerance for solve
        )
        
        # Check solution
        assert solution.shape == initial_qpos.shape
        assert "errors" in info
        
        # Verify the solution reaches the target (within tolerance)
        final_pos = solver._get_site_position("end_effector_l", solution)
        error = np.linalg.norm(final_pos - target_pos)
        assert error < 0.001  # 1mm tolerance - achievable with base movement allowed
    
    def test_ik_solve_dual_arm(self, rby1_model_and_data):
        """Test IK solving for dual arm reaching."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        solver = RBY1IK(model, data)
        
        # Set initial configuration with proper base height
        initial_qpos = np.zeros(model.nq)
        initial_qpos[2] = 0.35  # Set proper base height
        initial_qpos[3] = 1.0  # Identity quaternion
        
        # Fixed base position and orientation
        base_pos = np.array([0.0, 0.0, 0.35])
        base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Get current end effector positions with proper base height
        left_ee_pos = solver._get_site_position("end_effector_l", initial_qpos)
        right_ee_pos = solver._get_site_position("end_effector_r", initial_qpos)
        
        # Create targets (smaller movements for better convergence)
        left_target = left_ee_pos + np.array([0.03, 0.02, 0.0])
        right_target = right_ee_pos + np.array([0.03, -0.02, 0.0])
        
        # Solve IK for both arms with fixed base
        solution, success, info = solver.solve(
            base_pos=base_pos,
            base_quat=base_quat,
            left_target_pos=left_target,
            right_target_pos=right_target,
            current_qpos=initial_qpos,
            max_iterations=100,
            tolerance=0.02
        )
        
        # Check solution
        assert solution.shape == initial_qpos.shape
        assert "left_position_error" in info["errors"]
        assert "right_position_error" in info["errors"]
        
        # Verify both arms reach their targets
        final_left = solver._get_site_position("end_effector_l", solution)
        final_right = solver._get_site_position("end_effector_r", solution)
        
        left_error = np.linalg.norm(final_left - left_target)
        right_error = np.linalg.norm(final_right - right_target)
        
        assert left_error < 0.001  # 1mm tolerance for dual-arm IK
        assert right_error < 0.001
    
    def test_base_movement_minimal(self, rby1_model_and_data):
        """Test that base movement is minimal during IK solving."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        solver = RBY1IK(model, data)
        
        # Set initial configuration with proper base height
        initial_qpos = np.zeros(model.nq)
        initial_qpos[2] = 0.35  # Set proper base height
        initial_qpos[3] = 1.0  # Identity quaternion
        
        # Set specific base position and orientation
        base_pos = np.array([0.5, 0.3, 0.35])  # X, Y, Z position
        base_quat = solver._euler_to_quat(0, 0, np.pi / 4)  # 45 degrees rotation
        
        # Update initial qpos with base position for getting correct EE position
        initial_qpos[0:3] = base_pos
        initial_qpos[3:7] = base_quat
        
        # Get end effector position
        left_ee_pos = solver._get_site_position("end_effector_l", initial_qpos)
        target_pos = left_ee_pos + np.array([0.05, 0.05, 0.0])
        
        # Solve IK with specific base configuration
        solution, success, info = solver.solve(
            base_pos=base_pos,
            base_quat=base_quat,
            left_target_pos=target_pos,
            current_qpos=initial_qpos,
            max_iterations=100,
            tolerance=0.01
        )
        
        # Check that base movement is minimal (within 10cm)
        base_movement = np.linalg.norm(solution[0:3] - base_pos)
        assert base_movement < 0.1  # Base should move less than 10cm
        # Orientation should remain relatively unchanged
        quat_diff = np.abs(solution[3:7] - base_quat).max()
        assert quat_diff < 0.1  # Small quaternion change
    
    def test_quaternion_conversion(self, rby1_model_and_data):
        """Test quaternion utility functions."""
        from bigym.ik.rby1_ik import RBY1IK
        
        model, data = rby1_model_and_data
        solver = RBY1IK(model, data)
        
        # Test quaternion to rotation matrix
        quat = np.array([1, 0, 0, 0])  # Identity
        R = solver._quat_to_rotmat(quat)
        assert np.allclose(R, np.eye(3))
        
        # Test Euler to quaternion
        quat = solver._euler_to_quat(0, 0, 0)
        assert np.allclose(quat, [1, 0, 0, 0])
        
        # Test 90 degree rotation around Z
        quat = solver._euler_to_quat(0, 0, np.pi/2)
        # Should be approximately [0.707, 0, 0, 0.707]
        assert np.abs(quat[0] - np.sqrt(2)/2) < 0.01
        assert np.abs(quat[3] - np.sqrt(2)/2) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])