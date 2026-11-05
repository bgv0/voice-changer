"""Install / uninstall the AMD DirectML patches into a VCClient v1.5.3 folder.

    python install_patch.py "<...>/MMVCServerSIO"            # install
    python install_patch.py "<...>/MMVCServerSIO" --uninstall

Install:
  * copies vc_patches/ (patched modules) next to MMVCServerSIO.exe
  * in base_library.zip, renames pathlib.pyc -> _vcpatch_orig_pathlib.pyc and
    adds pathlib.py (the hook), keeping a backup as base_library.zip.bak
Uninstall restores base_library.zip.bak. Close the voice changer first.
"""
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
app = sys.argv[1]
zip_path = os.path.join(app, "base_library.zip")
bak = zip_path + ".bak"

if "--uninstall" in sys.argv:
    if os.path.exists(bak):
        shutil.copy2(bak, zip_path)
        print("restored", zip_path)
    shutil.rmtree(os.path.join(app, "vc_patches"), ignore_errors=True)
    print("removed vc_patches")
    sys.exit()

if not os.path.exists(os.path.join(app, "MMVCServerSIO.exe")):
    sys.exit("not a VCClient v1.5.3 folder: " + app)

# 1. patched modules
dst = os.path.join(app, "vc_patches")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(os.path.join(HERE, "vc_patches")):
    if f.endswith(".py") and f != "pathlib_hook.py":
        shutil.copy2(os.path.join(HERE, "vc_patches", f), dst)
        print("copied", f)
shutil.copy2(os.path.join(HERE, "launch_voice_changer.ps1"), app)
print("copied launch_voice_changer.ps1")

# 2. hook in base_library.zip (always rebuilt from the untouched backup)
if not os.path.exists(bak):
    shutil.copy2(zip_path, bak)
with zipfile.ZipFile(bak) as src:
    names = src.namelist()
    if "pathlib.pyc" not in names:
        sys.exit("pathlib.pyc not found in base_library.zip - unexpected build")
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for n in names:
            data = src.read(n)
            out.writestr("_vcpatch_orig_pathlib.pyc" if n == "pathlib.pyc" else src.getinfo(n), data)
        with open(os.path.join(HERE, "vc_patches", "pathlib_hook.py"), "rb") as f:
            out.writestr("pathlib.py", f.read())
os.replace(tmp, zip_path)
print("hook installed in", zip_path)
