import os
import urllib.request

def download_file(url, path):
    """Downloads a file with a browser-like User-Agent."""
    if not os.path.exists(path):
        print(f"[*] Downloading : {os.path.basename(path)}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)
        print(f" [+] Success.")
    else:
        # print(f" [V] {os.path.basename(path)} already present.")
        pass

def get_model_path(filename, models_dir="models"):
    """Returns absolute path to a model file, ensuring existence."""
    # src/antigravity/utils/download_utils.py -> needs 3 levels up to root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root_dir, models_dir, filename)
