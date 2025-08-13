#!/usr/bin/env python3
"""Wrapper script to run demo player with workarounds for macOS GLFW issues."""

import os
import sys

# Set environment variables to mitigate GLFW conflicts
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['MUJOCO_GL'] = 'glfw'

# Add project to Python path
sys.path.insert(0, '/Users/dian/Documents/projects/bigym')

try:
    # Import and run the demo player
    from tools.demo_player.main import DemoPlayer
    app = DemoPlayer()
except Exception as e:
    print(f"Demo player failed with error: {e}")
    print("Try using the simpler replay example instead:")
    print("PYTHONPATH=/Users/dian/Documents/projects/bigym python examples/replay_demo.py")