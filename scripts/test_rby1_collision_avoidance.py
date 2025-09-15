#!/usr/bin/env python3
"""Test RBY1 self-collision avoidance effectiveness during demo rollout.

This script rolls out an RBY1 demo and compares collision counts with and without
the IK solver's collision avoidance constraints to evaluate their effectiveness.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import mujoco
from collections import defaultdict

# from bigym.envs.test_env import TestEnv
from bigym.envs.pick_and_place import TakeCups
from bigym.envs.manipulation import FlipCup
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from demonstrations.demo import Demo
from bigym.robots.configs.rby1 import RBY1
from bigym.const import HandSide


def get_collision_pairs_from_ik(env):
    """Get the exact geom pairs that the IK solver tries to avoid collisions for.
    
    This function uses the provided environment to get the actual collision pairs
    including environment collision avoidance.
    
    Args:
        env: The environment to get geom pairs from
        
    Returns:
        List of tuples containing (group1, group2) geom sets
    """
    from bigym.ik.rby1_whole_body_ik import RBY1WholeBodyIK
    
    # Get the IK solver from the provided environment
    physics = env.robot._mojo.physics
    model = physics.model._model
    data = physics.data._data
    
    ik_solver = RBY1WholeBodyIK(model, data)
    
    # Get environment geoms
    environment_geoms = ik_solver._get_environment_geoms()
    
    # Define geom groups exactly as in rby1_whole_body_ik.py
    base_group = {"base_col_0", "base_col_1"}

    torso_0_group = {"torso_0_col_0", "torso_0_col_1"}
    torso_1_group = {"torso_1_col_0", "torso_1_col_1", "torso_1_col_2", "torso_1_col_3", "torso_1_col_4", "torso_1_col_5", "torso_1_col_6", "torso_1_col_7", "torso_1_col_8", "torso_1_col_9", "torso_1_col_10"}
    torso_2_group = {"torso_2_col_0", "torso_2_col_1", "torso_2_col_2", "torso_2_col_3", "torso_2_col_4", "torso_2_col_5", "torso_2_col_6", "torso_2_col_7", "torso_2_col_8", "torso_2_col_9", "torso_2_col_10"}
    torso_4_group = {"torso_4_col_0", "torso_4_col_1", "torso_4_col_2", "torso_4_col_3"}
    torso_5_group = {"torso_5_col_0", "torso_5_col_1", "torso_5_col_2", "torso_5_col_3", "torso_5_col_4"}

    right_arm_0_group = {"right_arm_0_col_0", "right_arm_0_col_1", "right_arm_0_col_2"}
    right_arm_1_group = {"right_arm_1_col_0"}
    right_arm_2_group = {"right_arm_2_col_0", "right_arm_2_col_1", "right_arm_2_col_2", "right_arm_2_col_3", "right_arm_2_col_4", "right_arm_2_col_5", "right_arm_2_col_6", "right_arm_2_col_7"}
    right_arm_3_group = {"right_arm_3_col_0", "right_arm_3_col_1", "right_arm_3_col_2", "right_arm_3_col_3"}
    right_arm_4_group = {"right_arm_4_col_0", "right_arm_4_col_1", "right_arm_4_col_2", "right_arm_4_col_3", "right_arm_4_col_4"}
    right_arm_5_group = {"right_arm_5_col_0", "right_arm_5_col_1", "right_arm_5_col_2"}
    right_arm_6_group = {"right_arm_6_col_0"}
    right_arm_7_group = {"right_arm_7_col_0"}

    left_arm_0_group = {"left_arm_0_col_0", "left_arm_0_col_1", "left_arm_0_col_2"}
    left_arm_1_group = {"left_arm_1_col_0"}
    left_arm_2_group = {"left_arm_2_col_0", "left_arm_2_col_1", "left_arm_2_col_2", "left_arm_2_col_3", "left_arm_2_col_4", "left_arm_2_col_5", "left_arm_2_col_6", "left_arm_2_col_7"}
    left_arm_3_group = {"left_arm_3_col_0", "left_arm_3_col_1", "left_arm_3_col_2", "left_arm_3_col_3"}
    left_arm_4_group = {"left_arm_4_col_0", "left_arm_4_col_1", "left_arm_4_col_2", "left_arm_4_col_3", "left_arm_4_col_4"}
    left_arm_5_group = {"left_arm_5_col_0", "left_arm_5_col_1", "left_arm_5_col_2"}
    left_arm_6_group = {"left_arm_6_col_0"}
    left_arm_7_group = {"left_arm_7_col_0"}

    # Add namespace prefix if needed
    if ik_solver.has_namespace:
        base_group = {"rby1/" + name for name in base_group}
        torso_0_group = {"rby1/" + name for name in torso_0_group}
        torso_1_group = {"rby1/" + name for name in torso_1_group}
        torso_2_group = {"rby1/" + name for name in torso_2_group}
        torso_4_group = {"rby1/" + name for name in torso_4_group}
        torso_5_group = {"rby1/" + name for name in torso_5_group}
        right_arm_0_group = {"rby1/" + name for name in right_arm_0_group}
        right_arm_1_group = {"rby1/" + name for name in right_arm_1_group}
        right_arm_2_group = {"rby1/" + name for name in right_arm_2_group}
        right_arm_3_group = {"rby1/" + name for name in right_arm_3_group}
        right_arm_4_group = {"rby1/" + name for name in right_arm_4_group}
        right_arm_5_group = {"rby1/" + name for name in right_arm_5_group}
        right_arm_6_group = {"rby1/" + name for name in right_arm_6_group}
        right_arm_7_group = {"rby1/" + name for name in right_arm_7_group}
        left_arm_0_group = {"rby1/" + name for name in left_arm_0_group}
        left_arm_1_group = {"rby1/" + name for name in left_arm_1_group}
        left_arm_2_group = {"rby1/" + name for name in left_arm_2_group}
        left_arm_3_group = {"rby1/" + name for name in left_arm_3_group}
        left_arm_4_group = {"rby1/" + name for name in left_arm_4_group}
        left_arm_5_group = {"rby1/" + name for name in left_arm_5_group}
        left_arm_6_group = {"rby1/" + name for name in left_arm_6_group}
        left_arm_7_group = {"rby1/" + name for name in left_arm_7_group}

    base_torso_group = base_group | torso_0_group | torso_1_group | torso_2_group | torso_4_group | torso_5_group
    left_arm_group = left_arm_0_group | left_arm_1_group | left_arm_2_group | left_arm_3_group | left_arm_4_group | left_arm_5_group | left_arm_6_group | left_arm_7_group
    right_arm_group = right_arm_0_group | right_arm_1_group | right_arm_2_group | right_arm_3_group | right_arm_4_group | right_arm_5_group | right_arm_6_group | right_arm_7_group

    # Environment collision group - all robot collision geoms
    robot_collision_group = base_torso_group | left_arm_group | right_arm_group

    geom_pairs = [
        (base_torso_group, left_arm_group),
        (base_torso_group, right_arm_group),
        (left_arm_group, right_arm_group),
    ]
    
    # Add environment collision avoidance if environment geoms exist
    if environment_geoms:
        geom_pairs.append((robot_collision_group, environment_geoms))
    
    return geom_pairs


def check_target_collisions(physics, model, data, ik_geom_pairs):
    """Check for collisions and near-misses between the specific groups that IK is trying to avoid.
    
    Args:
        physics: MuJoCo physics object
        model: MuJoCo model
        data: MuJoCo data
        ik_geom_pairs: List of geom pairs that IK solver tries to avoid collisions for
        
    Returns:
        dict: Collision statistics for target collision pairs
    """
    num_contacts = data.ncon
    collision_stats = {
        'total_contacts': num_contacts,
        'total_self_collisions': 0,  # All self-collisions
        'target_collisions': 0,  # Collisions between groups that IK should avoid
        'target_collision_pairs': defaultdict(int),
        'target_collision_details': [],
        'all_self_collision_pairs': defaultdict(int),  # All self-collision pairs
        'all_self_collision_details': [],  # All self-collision details
        'near_misses': 0,  # Near-miss collisions (close but not touching)
        'near_miss_pairs': defaultdict(int),
        'near_miss_details': []
    }
    
    # Create mapping from geom names to their groups
    geom_to_group = {}
    for group1, group2 in ik_geom_pairs:
        for geom in group1:
            # Check if this is an environment geom
            if 'rby1/' not in geom and 'base_col' not in geom and 'torso_' not in geom and 'arm_' not in geom:
                geom_to_group[geom] = 'environment'
            else:
                # Robot geom - determine group
                if 'left_arm' in geom:
                    geom_to_group[geom] = 'left_arm'
                    geom_to_group[f"rby1/{geom}"] = 'left_arm'
                elif 'right_arm' in geom:
                    geom_to_group[geom] = 'right_arm'
                    geom_to_group[f"rby1/{geom}"] = 'right_arm'
                else:
                    geom_to_group[geom] = 'base_torso'
                    geom_to_group[f"rby1/{geom}"] = 'base_torso'
        
        for geom in group2:
            # Check if this is an environment geom
            if 'rby1/' not in geom and 'base_col' not in geom and 'torso_' not in geom and 'arm_' not in geom:
                geom_to_group[geom] = 'environment'
            else:
                # Robot geom - determine group
                if 'left_arm' in geom:
                    geom_to_group[geom] = 'left_arm'
                    geom_to_group[f"rby1/{geom}"] = 'left_arm'
                elif 'right_arm' in geom:
                    geom_to_group[geom] = 'right_arm'
                    geom_to_group[f"rby1/{geom}"] = 'right_arm'
                else:
                    geom_to_group[geom] = 'base_torso'
                    geom_to_group[f"rby1/{geom}"] = 'base_torso'
    
    for i in range(num_contacts):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        if geom1_id >= 0 and geom2_id >= 0:
            # Get geom names
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if geom1_name and geom2_name:
                # Get body names
                body1_id = model.geom_bodyid[geom1_id]
                body2_id = model.geom_bodyid[geom2_id]
                body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                
                # Check if this is a robot self-collision
                if body1_name and body2_name and 'rby1' in body1_name.lower() and 'rby1' in body2_name.lower():
                    # Count all self-collisions
                    collision_stats['total_self_collisions'] += 1
                    
                    # Store all self-collision details
                    body_pair_key = tuple(sorted([body1_name, body2_name]))
                    collision_stats['all_self_collision_pairs'][body_pair_key] += 1
                    collision_stats['all_self_collision_details'].append({
                        'geom1': geom1_name,
                        'geom2': geom2_name,
                        'body1': body1_name,
                        'body2': body2_name
                    })
                
                # Check if this is a collision between groups that IK should avoid
                # This includes both robot self-collisions and robot-environment collisions
                group1 = geom_to_group.get(geom1_name)
                group2 = geom_to_group.get(geom2_name)
                
                if group1 and group2 and group1 != group2:
                    # This is a target collision that IK should avoid
                    collision_stats['target_collisions'] += 1
                    
                    # Create collision pair key
                    pair_key = tuple(sorted([group1, group2]))
                    collision_stats['target_collision_pairs'][pair_key] += 1
                    
                    # Store detailed collision info
                    collision_stats['target_collision_details'].append({
                        'geom1': geom1_name,
                        'geom2': geom2_name,
                        'body1': body1_name,
                        'body2': body2_name,
                        'group1': group1,
                        'group2': group2
                    })
    
    # Check for near-miss collisions (geoms that are close but not touching)
    # This helps us see when the IK solver is actively preventing collisions
    near_miss_threshold = 0.05  # 5cm threshold for near-misses
    
    for group1, group2 in ik_geom_pairs:
        for geom1_name in group1:
            for geom2_name in group2:
                # Skip if these geoms are already in collision
                already_colliding = False
                for detail in collision_stats['target_collision_details']:
                    if ((detail['geom1'] == geom1_name and detail['geom2'] == geom2_name) or
                        (detail['geom1'] == geom2_name and detail['geom2'] == geom1_name)):
                        already_colliding = True
                        break
                
                if already_colliding:
                    continue
                
                # Get geom IDs
                geom1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
                geom2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)
                
                if geom1_id >= 0 and geom2_id >= 0:
                    # Get geom positions
                    geom1_pos = data.geom_xpos[geom1_id]
                    geom2_pos = data.geom_xpos[geom2_id]
                    
                    # Calculate distance
                    distance = np.linalg.norm(geom1_pos - geom2_pos)
                    
                    if distance < near_miss_threshold:
                        # This is a near-miss - IK is preventing collision
                        collision_stats['near_misses'] += 1
                        
                        # Get group names
                        group1_name = geom_to_group.get(geom1_name, 'unknown')
                        group2_name = geom_to_group.get(geom2_name, 'unknown')
                        
                        if group1_name != group2_name:
                            pair_key = tuple(sorted([group1_name, group2_name]))
                            collision_stats['near_miss_pairs'][pair_key] += 1
                            
                            collision_stats['near_miss_details'].append({
                                'geom1': geom1_name,
                                'geom2': geom2_name,
                                'distance': distance,
                                'group1': group1_name,
                                'group2': group2_name
                            })
    
    return collision_stats


def test_collision_avoidance_effectiveness(demo_idx=0, max_steps=None):
    """Test the effectiveness of IK collision avoidance during demo rollout.
    
    Args:
        demo_idx: Which demo to test (default: 0)
        max_steps: Maximum number of steps to test (default: all)
    """
    
    # Load RBY1 demo
    demo_dir = Path("rby1_cartesian_demos_flipcup")
    demo_files = sorted(demo_dir.glob("rby1_cartesian_demo_*.safetensors"))
    
    if not demo_files or demo_idx >= len(demo_files):
        print(f"Demo {demo_idx} not found!")
        return None
    
    demo = Demo.from_safetensors(demo_files[demo_idx])
    print(f"Loaded demo {demo_idx} with seed {demo.seed}, {len(demo.timesteps)} timesteps")
    
    # Create RBY1 environment with IK collision avoidance (default)
    action_mode_with_ik = RBY1CartesianActionModeWholeBody(
        direct_mode=False,
        block_until_reached=False,
        control_frequency=20
    )
    
    env_with_ik = FlipCup(
        action_mode=action_mode_with_ik,
        control_frequency=20,
        render_mode="human",
        robot_cls=RBY1
    )
    
    # Reset with demo seed
    env_with_ik.reset(seed=demo.seed)
    
    # Get geom pairs that IK solver tries to avoid collisions for
    ik_geom_pairs = get_collision_pairs_from_ik(env_with_ik)
    print(f"\nIK Solver collision avoidance targets {len(ik_geom_pairs)} geom group pairs:")
    for i, (group1, group2) in enumerate(ik_geom_pairs):
        print(f"  Pair {i+1}: {len(group1)} geoms <-> {len(group2)} geoms")
        print(f"    Group 1: {sorted(list(group1))[:3]}...")  # Show first 3
        print(f"    Group 2: {sorted(list(group2))[:3]}...")  # Show first 3
        
        # Check if this is an environment collision pair
        if any('rby1/' not in geom and 'base_col' not in geom and 'torso_' not in geom and 'arm_' not in geom for geom in group1 | group2):
            print(f"    *** ENVIRONMENT COLLISION AVOIDANCE ***")
    
    # Storage for collision statistics
    steps = []
    target_collision_stats = []
    
    print(f"\nStarting collision avoidance effectiveness test...")
    print(f"Target collision pairs to avoid:")
    for i, (group1, group2) in enumerate(ik_geom_pairs):
        print(f"  {i+1}. {len(group1)} geoms <-> {len(group2)} geoms")
    
    num_steps = len(demo.timesteps) if max_steps is None else min(max_steps, len(demo.timesteps))
    
    for step_idx in range(num_steps):
        timestep = demo.timesteps[step_idx]
        
        # Get action
        action = timestep.info.get('demo_action')
        if action is None:
            action = timestep.executed_action
        if action is None:
            action = timestep.action
        
        if action is None:
            print(f"Step {step_idx}: No action found!")
            continue
        
        # Step environment with IK collision avoidance
        try:
            _, _, terminated, truncated, info = env_with_ik.step(action)
            env_with_ik.render()
            
            # Get physics objects
            physics = env_with_ik.robot._mojo.physics
            model = physics.model._model
            data = physics.data._data
            
            # Check for target collisions (between groups that IK should avoid)
            stats = check_target_collisions(physics, model, data, ik_geom_pairs)
            
            # Store statistics
            steps.append(step_idx)
            target_collision_stats.append(stats)
            
            # Print progress every 10 steps
            if step_idx % 10 == 0:
                print(f"Step {step_idx}: {stats['total_self_collisions']} total self-collisions, {stats['target_collisions']} target collisions")
                if stats['target_collisions'] > 0:
                    print(f"  Target collision pairs: {dict(stats['target_collision_pairs'])}")
            
            if info.get('task_success', False):
                print(f"✅ SUCCESS at step {step_idx}!")
                break
            
            if terminated or truncated:
                print(f"Episode ended at step {step_idx}")
                break
                
        except Exception as e:
            print(f"  ERROR at step {step_idx}: {e}")
            break
    
    env_with_ik.close()
    
    # Analyze results
    print("\n" + "="*80)
    print("COLLISION AVOIDANCE EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    if len(steps) > 0:
        # Calculate summary statistics
        total_self_collisions = sum(stats['total_self_collisions'] for stats in target_collision_stats)
        total_target_collisions = sum(stats['target_collisions'] for stats in target_collision_stats)
        steps_with_self_collisions = sum(1 for stats in target_collision_stats if stats['total_self_collisions'] > 0)
        steps_with_target_collisions = sum(1 for stats in target_collision_stats if stats['target_collisions'] > 0)
        
        print(f"\nOverall Statistics:")
        print(f"  Total steps analyzed: {len(steps)}")
        print(f"  Steps with self-collisions: {steps_with_self_collisions} ({steps_with_self_collisions/len(steps)*100:.1f}%)")
        print(f"  Steps with target collisions: {steps_with_target_collisions} ({steps_with_target_collisions/len(steps)*100:.1f}%)")
        print(f"  Total self-collisions: {total_self_collisions}")
        print(f"  Total target collisions: {total_target_collisions}")
        
        if total_target_collisions == 0:
            print(f"  ✅ SUCCESS: No collisions between target groups detected!")
            print(f"  IK collision avoidance is working effectively.")
        else:
            print(f"  ❌ ISSUE: {total_target_collisions} collisions between target groups detected.")
            print(f"  IK collision avoidance needs improvement.")
        
        # Analyze collision patterns
        print(f"\nAll Self-Collision Pattern Analysis:")
        
        # Aggregate all self-collision pairs across all steps
        all_self_collision_pairs = defaultdict(int)
        all_self_collision_details = []
        
        for stats in target_collision_stats:
            for pair, count in stats['all_self_collision_pairs'].items():
                all_self_collision_pairs[pair] += count
            all_self_collision_details.extend(stats['all_self_collision_details'])
        
        if all_self_collision_pairs:
            print(f"  Most frequent self-collision pairs:")
            for pair, count in sorted(all_self_collision_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {pair[0]} <-> {pair[1]}: {count} collisions")
        
        print(f"\nTarget Collision Pattern Analysis:")
        
        # Aggregate collision pairs across all steps
        all_target_collision_pairs = defaultdict(int)
        all_collision_details = []
        
        for stats in target_collision_stats:
            for pair, count in stats['target_collision_pairs'].items():
                all_target_collision_pairs[pair] += count
            all_collision_details.extend(stats['target_collision_details'])
        
        if all_target_collision_pairs:
            print(f"  Target collision pairs (groups that should not collide):")
            for pair, count in sorted(all_target_collision_pairs.items(), key=lambda x: x[1], reverse=True):
                print(f"    {pair[0]} <-> {pair[1]}: {count} collisions")
        
        # Show detailed collision examples
        if all_collision_details:
            print(f"\nDetailed Collision Examples (first 5):")
            for i, detail in enumerate(all_collision_details[:5]):
                print(f"  {i+1}. {detail['group1']} <-> {detail['group2']}")
                print(f"     Bodies: {detail['body1']} <-> {detail['body2']}")
                print(f"     Geoms: {detail['geom1']} <-> {detail['geom2']}")
        
        # Step-by-step analysis for first few steps with collisions
        print(f"\nStep-by-step Analysis (first 10 steps with any collisions):")
        collision_steps = [i for i, stats in enumerate(target_collision_stats) if stats['total_self_collisions'] > 0][:10]
        
        for step_idx in collision_steps:
            stats = target_collision_stats[step_idx]
            print(f"  Step {step_idx}: {stats['total_self_collisions']} total self-collisions, {stats['target_collisions']} target collisions")
            if stats['target_collision_pairs']:
                print(f"    Target pairs: {dict(stats['target_collision_pairs'])}")
    
    return {
        'steps': steps,
        'target_collision_stats': target_collision_stats,
        'ik_geom_pairs': ik_geom_pairs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RBY1 collision avoidance effectiveness")
    parser.add_argument("--demo", type=int, default=0, help="Demo index to test")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps to test")
    args = parser.parse_args()
    
    test_collision_avoidance_effectiveness(
        demo_idx=args.demo,
        max_steps=args.max_steps
    )
