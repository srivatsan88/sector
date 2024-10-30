import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
else:
    print(f"Project root already in sys.path: {project_root}")