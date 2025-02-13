import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imageio
import numpy as np

def read_hdr_image(path):
    """
    Reads an HDR image using imageio.
    Converts it to an 8-bit image for further processing.
    (For training, you might want to preserve dynamic range—but this example converts to 8-bit.)
    """
    hdr = imageio.imread(path)
    # If the HDR image is not 8-bit, normalize it to [0, 255]
    if hdr.dtype != np.uint8:
        hdr = (hdr / (np.max(hdr) + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(hdr).convert("RGB")

def strip_ldr(name):
    # Remove the trailing '_LDR' if it exists
    if name.endswith("_LDR"):
        return name[:-4]
    return name

class LDRHDRDataset(Dataset):
    def __init__(self, root_dir, ldr_dir="LDR", hdr_dir="HDR", transform=None):
        super(LDRHDRDataset, self).__init__()
        self.root_dir = root_dir
        self.ldr_path = os.path.join(root_dir, ldr_dir)
        self.hdr_path = os.path.join(root_dir, hdr_dir)
        
        ldr_files = sorted(os.listdir(self.ldr_path))
        hdr_files = sorted(os.listdir(self.hdr_path))
        
        # Strip '_LDR' from LDR base filenames for matching
        ldr_names = {strip_ldr(os.path.splitext(f)[0]) for f in ldr_files}
        hdr_names = {os.path.splitext(f)[0] for f in hdr_files}
        
        common_names = sorted(list(ldr_names.intersection(hdr_names)))
        
        if not common_names:
            raise ValueError("No common image pairs found!")
        
        # Construct full paths
        self.ldr_files = [os.path.join(self.ldr_path, name + "_LDR.png") for name in common_names]
        self.hdr_files = []
        for name in common_names:
            found = False
            for ext in [".hdr", ".exr", ".png", ".jpg"]:
                candidate = os.path.join(self.hdr_path, name + ext)
                if os.path.exists(candidate):
                    self.hdr_files.append(candidate)
                    found = True
                    break
            if not found:
                print(f"Warning: No HDR file found for {name}")
        
        self.transform = transform

    def __len__(self):
        return len(self.ldr_files)
    
    def __getitem__(self, idx):
        # Load LDR image normally with PIL
        ldr_img = Image.open(self.ldr_files[idx]).convert("RGB")
        hdr_path = self.hdr_files[idx]
        # Use imageio to read HDR files (if extension indicates HDR)
        if hdr_path.lower().endswith(('.hdr', '.exr')):
            hdr_img = read_hdr_image(hdr_path)
        else:
            hdr_img = Image.open(hdr_path).convert("RGB")
        
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        else:
            ldr_img = transforms.ToTensor()(ldr_img)
            hdr_img = transforms.ToTensor()(hdr_img)
        return ldr_img, hdr_img
