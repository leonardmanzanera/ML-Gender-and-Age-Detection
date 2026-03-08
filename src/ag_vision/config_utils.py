import os
import yaml

def load_config(config_path="config/params.yaml"):
    """Loads system topology and parameters."""
    # Find root directory (where launcher is)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    full_path = os.path.join(root_dir, config_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)
