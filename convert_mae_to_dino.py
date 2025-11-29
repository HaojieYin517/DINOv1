import os
import torch
import argparse
from collections import OrderedDict

# Allow MAE checkpoints that contain argparse.Namespace
torch.serialization.add_safe_globals([argparse.Namespace])

# ============================================================
# User settings
# ============================================================
CKPT_PATH = "/workspace/mae_patch8_512/checkpoint-299.pth"
OUTPUT_DIR = "/workspace/dino_pretrain_init"
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

basename = os.path.basename(CKPT_PATH)
try:
    epoch = basename.split("-")[1].split(".")[0]
except:
    epoch = "unknown"

output_path = os.path.join(OUTPUT_DIR, f"mae_init_new_epoch{epoch}.pth")

print(f"Loading MAE checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

if "model" not in ckpt:
    raise RuntimeError("Checkpoint missing 'model' key — is this a MAE pretrain ckpt?")

state_dict = ckpt["model"]
new_state = OrderedDict()

KEEP_PREFIXES = [
    "patch_embed",
    "blocks",
    "cls_token",
    "pos_embed",
    "norm",
]

print("Extracting encoder-only weights...")

for k, v in state_dict.items():
    if k.startswith("encoder."):
        new_state[k.replace("encoder.", "")] = v
    else:
        if any(k.startswith(prefix) for prefix in KEEP_PREFIXES):
            new_state[k] = v

torch.save(new_state, output_path)

print(f"\n✅ Saved DINO-ready weights to:")
print(f"   → {output_path}")
print(f"   (Initialized from MAE epoch {epoch})")
