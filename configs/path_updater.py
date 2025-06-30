#!/usr/bin/env python3
"""
Script to recursively update path bases in YAML configuration files.
Automatically detects common path prefixes and replaces them with a new base.
"""

import os
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Set

# Configuration - modify this to set your new base path
NEW_BASE_PATH = "/home/jovyan/workspace"
SEARCH_DIRECTORY = "."  # Current directory, change as needed

class YamlPathUpdater:
    def __init__(self, new_base: str, search_dir: str = "."):
        self.new_base = new_base.rstrip('/')
        self.search_dir = Path(search_dir)
        self.yaml_files = []
        self.all_paths = []

    def find_yaml_files(self) -> List[Path]:
        """Recursively find all YAML files in the directory."""
        yaml_files = []
        for ext in ['*.yaml', '*.yml']:
            yaml_files.extend(self.search_dir.rglob(ext))
        return yaml_files

    def extract_paths_from_value(self, value: Any) -> List[str]:
        """Extract potential file paths from a YAML value."""
        paths = []
        if isinstance(value, str):
            # Check if it looks like an absolute path
            if value.startswith('/') and len(value) > 5:
                paths.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                paths.extend(self.extract_paths_from_value(v))
        elif isinstance(value, list):
            for item in value:
                paths.extend(self.extract_paths_from_value(item))
        return paths

    def collect_all_paths(self) -> List[str]:
        """Collect all paths from all YAML files."""
        all_paths = []
        self.yaml_files = self.find_yaml_files()

        for yaml_file in self.yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        paths = self.extract_paths_from_value(data)
                        all_paths.extend(paths)
            except Exception as e:
                print(f"Warning: Could not read {yaml_file}: {e}")

        return all_paths

    def find_common_base(self, paths: List[str]) -> str:
        """Find the most common base path among all collected paths."""
        if not paths:
            return ""

        # Extract potential bases by looking at path prefixes
        base_candidates = []

        for path in paths:
            parts = path.split('/')
            # Try different prefix lengths
            for i in range(2, len(parts)):  # Start from 2 to avoid just '/'
                base = '/'.join(parts[:i])
                if len(base) > 5:  # Reasonable minimum length
                    base_candidates.append(base)

        if not base_candidates:
            return ""

        # Find the most common base
        base_counts = Counter(base_candidates)
        most_common_base = base_counts.most_common(1)[0][0]

        return most_common_base

    def update_path(self, path: str, old_base: str) -> str:
        """Update a single path by replacing the old base with the new base."""
        if path.startswith(old_base):
            relative_part = path[len(old_base):].lstrip('/')
            return f"{self.new_base}/{relative_part}"
        return path

    def update_yaml_value(self, value: Any, old_base: str) -> Any:
        """Recursively update paths in YAML values."""
        if isinstance(value, str):
            if value.startswith('/') and len(value) > 5:
                return self.update_path(value, old_base)
            return value
        elif isinstance(value, dict):
            return {k: self.update_yaml_value(v, old_base) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.update_yaml_value(item, old_base) for item in value]
        else:
            return value

    def update_yaml_file(self, yaml_file: Path, old_base: str) -> bool:
        """Update paths in a single YAML file."""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                return False

            updated_data = self.update_yaml_value(data, old_base)

            # Write back the updated YAML
            with open(yaml_file, 'w') as f:
                yaml.dump(updated_data, f, default_flow_style=False,
                         sort_keys=False, allow_unicode=True)

            return True

        except Exception as e:
            print(f"Error updating {yaml_file}: {e}")
            return False

    def run(self):
        """Main execution method."""
        print(f"Searching for YAML files in: {self.search_dir.absolute()}")

        # Collect all paths from YAML files
        all_paths = self.collect_all_paths()

        if not all_paths:
            print("No paths found in YAML files.")
            return

        print(f"Found {len(all_paths)} paths across {len(self.yaml_files)} YAML files")

        # Find the common base path
        old_base = "/net/projects2/interp" #self.find_common_base(all_paths)

        if not old_base:
            print("Could not determine a common base path.")
            return

        print(f"Detected common base path: {old_base}")
        print(f"Will replace with: {self.new_base}")

        # Confirm before proceeding
        response = input("Proceed with updates? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

        # Update all YAML files
        updated_count = 0
        for yaml_file in self.yaml_files:
            if self.update_yaml_file(yaml_file, old_base):
                updated_count += 1
                print(f"Updated: {yaml_file}")

        print(f"\nCompleted! Updated {updated_count} files.")

def main():
    updater = YamlPathUpdater(NEW_BASE_PATH, SEARCH_DIRECTORY)
    updater.run()

if __name__ == "__main__":
    main()
