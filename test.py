import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import LDRHDRDataset
from models.unet import UNet
from utils.utils import save_sample_image
from torchvision import transforms

def main():
    checkpoint_path = "/content/drive/MyDrive/AI_Project_LDR2HDR/checkpoints/checkpoint_epoch_100.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = LDRHDRDataset(root_dir="/content/drive/MyDrive/AI_Project_LDR2HDR/data/val",
                                 ldr_dir="LDR", hdr_dir="HDR", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    os.makedirs("/content/drive/MyDrive/AI_Project_LDR2HDR/results", exist_ok=True)
    with torch.no_grad():
        for idx, (ldr, _) in enumerate(test_loader):
            ldr = ldr.to(device)
            output = model(ldr)
            save_sample_image(output, filename=f"/content/drive/MyDrive/AI_Project_LDR2HDR/results/output_{idx}.png")
    
if __name__ == "__main__":
    main()
