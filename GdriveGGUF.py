# Option A: rename existing /content/drive -> backup, then mount
import os, shutil, time
from google.colab import drive

MOUNT_POINT = "/content/drive"
BACKUP = f"/content/drive_backup_{int(time.time())}"
LOCAL_GGUF = "/content/merged_phi2.gguf"
DEST = "/content/drive/MyDrive/merged_phi2.gguf"

# 1) If mount already present, just copy
if os.path.ismount(MOUNT_POINT):
    print("[INFO] /content/drive is already mounted.")
else:
    # If the mountpoint exists and is non-empty, move it aside (safe)
    if os.path.exists(MOUNT_POINT) and os.listdir(MOUNT_POINT):
        print(f"[INFO] {MOUNT_POINT} exists and is non-empty. Moving to backup: {BACKUP}")
        shutil.move(MOUNT_POINT, BACKUP)
        # recreate empty mount dir
        os.makedirs(MOUNT_POINT, exist_ok=True)

    # Now mount
    print("[INFO] Mounting Google Drive...")
    drive.mount(MOUNT_POINT)

# 2) Copy the file to Drive (will create MyDrive if not present)
if os.path.exists(LOCAL_GGUF):
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.copy(LOCAL_GGUF, DEST)
    print(f"[OK] Copied {LOCAL_GGUF} -> {DEST}")
    print("You can now download from https://drive.google.com (My Drive).")
else:
    print("[ERROR] Local file not found:", LOCAL_GGUF)
