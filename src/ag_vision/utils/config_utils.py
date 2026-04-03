import os
import yaml

def load_config(config_path="config/params.yaml"):
    """Loads system topology and parameters."""
    # Find root directory (where launcher is)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    full_path = os.path.join(root_dir, config_path)
    
    if not os.path.exists(full_path):
        # Fallback just in case
        fallback = os.path.join(os.path.abspath(os.path.join(current_dir, "..", "..")), config_path)
        if os.path.exists(fallback):
            full_path = fallback
            
    with open(full_path, "r") as f:
        return yaml.safe_load(f)
