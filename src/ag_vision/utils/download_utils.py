import os
import urllib.request
import ssl

def download_file(url, path):
    """Downloads a file with a browser-like User-Agent and ignores SSL cert errors."""
    if not os.path.exists(path):
        print(f"[*] Downloading : {os.path.basename(path)}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)
        print(f" [+] Success.")
    else:
        pass

def get_model_path(filename, models_dir="models"):
    """Returns absolute path to a model file, ensuring existence."""
    # src/antigravity/utils/download_utils.py -> needs 3 levels up to root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root_dir, models_dir, filename)
