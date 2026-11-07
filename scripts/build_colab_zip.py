"""Build colab_upload/M2CL_code.zip: the code, configs and scripts the Colab notebooks need.

  python scripts/build_colab_zip.py

The previous zip is kept as colab_upload/M2CL_code_<date of its last change>.zip.
"""
import datetime
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "colab_upload", "M2CL_code.zip")
INCLUDE_DIRS = ("src/m2cl", "configs", "scripts")
EXTRA = ("requirements.txt",)
SKIP = ("__pycache__",)

if os.path.exists(OUT):
    stamp = datetime.date.fromtimestamp(os.path.getmtime(OUT)).isoformat()
    backup = OUT.replace(".zip", f"_{stamp}.zip")
    if not os.path.exists(backup):
        shutil.copy2(OUT, backup)
        print("kept previous zip as", os.path.relpath(backup, ROOT))

files = list(EXTRA)
for d in INCLUDE_DIRS:
    for root, dirs, names in os.walk(os.path.join(ROOT, d)):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for n in names:
            if n.endswith((".py", ".yaml")):
                files.append(os.path.relpath(os.path.join(root, n), ROOT).replace("\\", "/"))
with zipfile.ZipFile(OUT + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        z.write(os.path.join(ROOT, f), f)
os.replace(OUT + ".tmp", OUT)
print(f"wrote {os.path.relpath(OUT, ROOT)} with {len(files)} files")
