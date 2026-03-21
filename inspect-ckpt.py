import torch

ckpt = torch.load(
    "/mnt/data8tb/Documents/models/repo/MotionGPT3/experiments/motgpt_2optimizer/Stage-1-diffusion/checkpoints/epoch=195-v1.ckpt",
    map_location="cpu",
    weights_only=False  # Safe since this is your own checkpoint
)

print("=== Top-level keys ===")
print(list(ckpt.keys()))

for key in ckpt.keys():
    print(f"\n=== {key} ===")
    val = ckpt[key]
    if isinstance(val, dict):
        print(f"  Type: dict with {len(val)} keys")
        for k in list(val.keys())[:10]:  # Show first 10 keys
            v = val[k]
            if hasattr(v, 'shape'):
                print(f"    {k}: {type(v).__name__} {tuple(v.shape)}")
            else:
                print(f"    {k}: {type(v).__name__}")
        if len(val) > 10:
            print(f"    ... and {len(val) - 10} more keys")
    elif hasattr(val, 'shape'):
        print(f"  Type: {type(val).__name__}, Shape: {val.shape}")
    else:
        print(f"  Type: {type(val).__name__}, Value: {repr(val)[:200]}")