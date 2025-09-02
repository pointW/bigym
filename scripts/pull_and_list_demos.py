#!/usr/bin/env python3
"""Pull all available demonstrations and list them."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from demonstrations.demo_store import DemoStore
from bigym.const import DEMO_RELEASES, DEMO_VERSION

def main():
    """Pull and list all available demos."""
    
    print("BigGym Demonstration Manager")
    print("="*80)
    
    # Show download URL
    url = f"{DEMO_RELEASES}/v{DEMO_VERSION}/demonstrations.zip"
    print(f"Demo repository URL: {url}")
    print(f"Demo version: {DEMO_VERSION}")
    
    # Initialize demo store
    demo_store = DemoStore()
    
    # Check if demos are already cached
    if demo_store.cached:
        print("\n✅ Demos are already cached locally")
    else:
        print("\n📥 Demos not found locally. Downloading...")
        try:
            demo_store.pull_demos()
            print("✅ Demos downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download demos: {e}")
            return
    
    # List downloaded demos
    print("\n" + "="*80)
    print("AVAILABLE DEMONSTRATIONS:")
    print("="*80)
    
    cache_path = Path.home() / ".bigym" / "demonstrations"
    
    if cache_path.exists():
        # Group demos by environment
        env_demos = {}
        
        # Walk through the cache directory
        for root, dirs, files in os.walk(cache_path):
            root_path = Path(root)
            # Skip the root directory itself
            if root_path == cache_path:
                continue
            
            # Get relative path from cache
            rel_path = root_path.relative_to(cache_path)
            parts = rel_path.parts
            
            if len(parts) >= 1:
                # First part is usually the environment name or robot name
                env_or_robot = parts[0]
                
                # Count demo files in this directory
                demo_files = [f for f in files if f.endswith('.safetensors') or f.endswith('.pkl')]
                
                if demo_files:
                    if env_or_robot not in env_demos:
                        env_demos[env_or_robot] = {}
                    
                    # Create a description of this demo set
                    desc = "/".join(parts[1:]) if len(parts) > 1 else "default"
                    env_demos[env_or_robot][desc] = len(demo_files)
        
        # Display organized by environment
        for env_name in sorted(env_demos.keys()):
            print(f"\n📁 {env_name}:")
            for config, count in env_demos[env_name].items():
                print(f"    {config}: {count} demos")
        
        # Also show the directory structure
        print("\n" + "="*80)
        print("DIRECTORY STRUCTURE:")
        print("="*80)
        
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            """Show directory tree."""
            if current_depth >= max_depth:
                return
            
            items = sorted(path.iterdir())
            dirs = [d for d in items if d.is_dir()]
            files = [f for f in items if f.is_file()]
            
            # Show directories
            for i, d in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'└── ' if is_last_dir else '├── '}{d.name}/")
                
                # Count demo files in this directory
                demo_count = len(list(d.glob("*.safetensors")) + list(d.glob("*.pkl")))
                if demo_count > 0:
                    print(f"{prefix}{'    ' if is_last_dir else '│   '}  ({demo_count} demos)")
                
                # Recurse
                extension = "    " if is_last_dir else "│   "
                show_tree(d, prefix + extension, max_depth, current_depth + 1)
            
            # Show count of demo files
            demo_files = [f for f in files if f.suffix in ['.safetensors', '.pkl']]
            if demo_files and current_depth < max_depth - 1:
                print(f"{prefix}└── [{len(demo_files)} demo files]")
        
        print(f"\n{cache_path}/")
        show_tree(cache_path)
        
    else:
        print("❌ Cache directory does not exist")
    
    # Show common environment names for testing
    print("\n" + "="*80)
    print("COMMON ENVIRONMENTS TO TRY:")
    print("="*80)
    print("""
Try loading demos with these environment names:
- ReachTarget
- MovePlate
- PickAndPlace
- Reach
- Push

Example usage:
    from bigym.envs.reach_target import ReachTarget
    from bigym.action_modes import JointPositionActionMode
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata
    
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        control_frequency=50,
        render_mode="human"
    )
    
    demo_store = DemoStore()
    metadata = Metadata.from_env(env)
    demos = demo_store.get_demos(metadata, amount=5, frequency=50)
    """)

if __name__ == "__main__":
    main()