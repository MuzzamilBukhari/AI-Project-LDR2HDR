import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    
def save_sample_image(tensor, filename="sample.png"):
    grid = vutils.make_grid(tensor, normalize=True, scale_each=True)
    npimg = grid.cpu().numpy()
    plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))
