import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LDRHDRDataset(Dataset):
    def __init__(self, root_dir, ldr_dir="LDR", hdr_dir="HDR", transform=None):
        super(LDRHDRDataset, self).__init__()
        self.root_dir = root_dir
        self.ldr_path = os.path.join(root_dir, ldr_dir)
        self.hdr_path = os.path.join(root_dir, hdr_dir)
        self.ldr_files = sorted(os.listdir(self.ldr_path))
        self.hdr_files = sorted(os.listdir(self.hdr_path))
        self.transform = transform
        if len(self.ldr_files) != len(self.hdr_files):
            raise ValueError("Mismatch between number of LDR and HDR images!")
        
    def __len__(self):
        return len(self.ldr_files)
    
    def __getitem__(self, idx):
        ldr_img_path = os.path.join(self.ldr_path, self.ldr_files[idx])
        hdr_img_path = os.path.join(self.hdr_path, self.hdr_files[idx])
        
        # Open images and convert to RGB
        ldr_img = Image.open(ldr_img_path).convert("RGB")
        hdr_img = Image.open(hdr_img_path).convert("RGB")
        
        if self.transform:
            ldr_img = self.transform(ldr_img)
            hdr_img = self.transform(hdr_img)
        else:
            ldr_img = transforms.ToTensor()(ldr_img)
            hdr_img = transforms.ToTensor()(hdr_img)
            
        return ldr_img, hdr_img
