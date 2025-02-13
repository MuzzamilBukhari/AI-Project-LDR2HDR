# LDR2HDR in Google Colab – Single Image LDR to HDR Conversion using Deep Learning

This repository provides a complete end-to-end implementation of a deep-learning model for converting a single LDR image into an HDR image. The code is adapted to run in Google Colab with GPU acceleration. In this example, we use the publicly available “HDR-Eye” dataset (from the Single‑Image HDR Reconstruction project by Liu et al. CVPR 2020citeturn1search10) as our free dataset.

## Project Structure

```
LDR2HDR_Colab/
├── configs/
│   └── config.yaml        # Training configuration (hyperparameters, paths)
├── data/
│   ├── train/
│   │   ├── LDR/           # Training LDR images
│   │   └── HDR/           # Corresponding training HDR images
│   └── val/
│       ├── LDR/           # Validation LDR images
│       └── HDR/           # Validation HDR images
├── models/
│   └── unet.py            # U-Net model definition
├── results/               # (Created at runtime) Test output images
├── checkpoints/           # (Created at runtime) Model checkpoints
├── utils/
│   ├── dataset.py         # PyTorch dataset for paired images
│   ├── loss.py            # Loss function (L1 loss)
│   └── utils.py           # Utility functions (saving checkpoints, images)
├── train.py               # Training script
├── test.py                # Inference script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Instructions for Running in Google Colab

### Step 1. Set Up Your Colab Environment

1. **Open Google Colab:**
   - Go to [https://colab.research.google.com](https://colab.research.google.com) and create a new notebook.

2. **Mount Your Google Drive:**
   - This lets you store your data and checkpoints permanently.
   - Add a new code cell at the top of your notebook and run:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Follow the on‑screen instructions to authorize access. Make sure your folder structure matches the paths in `config.yaml` (e.g., `/content/drive/MyDrive/LDR2HDR/`).

3. **Clone the Repository (Optional):**
   - If you have the code on GitHub, you can clone it directly:

     ```bash
     !git clone https://github.com/yourusername/LDR2HDR_Colab.git
     %cd LDR2HDR_Colab
     ```

4. **Install Dependencies:**
   - Run the following cell to install the required packages:

     ```python
     !pip install -r requirements.txt
     ```

### Step 2. Obtain a Public Free Dataset

For this example, we use the “HDR‑Eye” dataset from the Single‑Image HDR Reconstruction project by Liu et al. (CVPR 2020). The testing data is only about 0.52 GB, making it ideal for Colab.

- **Download Instructions:**
  - Visit [http://alex04072000.github.io/SingleHDR/](http://alex04072000.github.io/SingleHDR/) and scroll to the download section.
  - Download the “HDR‑Eye” dataset ZIP file.
  - Upload the ZIP file to your Google Drive (e.g., to a folder `/MyDrive/LDR2HDR/data/`).
  - In Colab, unzip the dataset into the appropriate folders. For example:

    ```python
    !unzip /content/drive/MyDrive/LDR2HDR/data/HDR-Eye.zip -d /content/drive/MyDrive/LDR2HDR/data/
    ```
  - Organize the unzipped files into the following structure:
    - `/content/drive/MyDrive/LDR2HDR/data/train/LDR`
    - `/content/drive/MyDrive/LDR2HDR/data/train/HDR`
    - Similarly for validation (or use a subset for testing).

*(If you prefer another free dataset, for example the “LDR-HDR Pair” dataset on Kaggle, follow the Kaggle API instructions to download it into your drive.)*

### Step 3. Train the Model

1. **Run the Training Script:**
   - In a new code cell, run:

     ```bash
     !python train.py
     ```
   - This script reads `configs/config.yaml`, loads the training data from your Google Drive, and begins training on the GPU.
   - Checkpoints and sample images will be saved in your Drive (e.g., under `/MyDrive/LDR2HDR/checkpoints`).

### Step 4. Test/Run Inference

1. **Run the Testing Script:**
   - After training (or using a pretrained checkpoint), run:

     ```bash
     !python test.py
     ```
   - This script will load the checkpoint, run inference on the validation data, and save output images in `/MyDrive/LDR2HDR/results`.

### Additional Tips

- **GPU/TPU Runtime:**  
  Ensure you have enabled a GPU runtime in Colab by clicking *Runtime* → *Change runtime type* and selecting *GPU*.
  
- **File Paths:**  
  Adjust the paths in `configs/config.yaml` if your folder structure differs.

- **Monitoring Training:**  
  You can add TensorBoard callbacks in the training script to monitor loss and metrics.

---

## Citations

- Liu, Y.-L., Lai, W.-S., Chen, Y.-S., Kao, Y.-L., Yang, M.-H., Chuang, Y.-Y., & Huang, J.-B. (2020). Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. citeturn1search10
