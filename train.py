import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import LDRHDRDataset
from models.unet import UNet
from utils.loss import l1_loss
from utils.utils import save_checkpoint, save_sample_image
from torchvision import transforms
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config("configs/config.yaml")["train"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = LDRHDRDataset(root_dir=config["data_root"],
                            ldr_dir=config.get("ldr_dir", "LDR"),
                            hdr_dir=config.get("hdr_dir", "HDR"),
                            transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=config["num_workers"])
    
    model = UNet(n_channels=3, n_classes=3, bilinear=True).to(device)
    lr = float(config["learning_rate"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_epochs = config["epochs"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for ldr, hdr in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            ldr = ldr.to(device)
            hdr = hdr.to(device)
            
            optimizer.zero_grad()
            outputs = model(ldr)
            loss = l1_loss(outputs, hdr)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * ldr.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        if epoch % config["save_interval"] == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)
            
            model.eval()
            with torch.no_grad():
                sample_ldr, _ = next(iter(dataloader))
                sample_ldr = sample_ldr.to(device)
                sample_output = model(sample_ldr)
                sample_img_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
                save_sample_image(sample_output, filename=sample_img_path)
                
if __name__ == "__main__":
    main()
