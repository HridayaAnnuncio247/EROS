import wget
import os

if not os.path.exists("sam_vit_b_01ec64.pth"):
    wget.download(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_01ec64.pth"
    )