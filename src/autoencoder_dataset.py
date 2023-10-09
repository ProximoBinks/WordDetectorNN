import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AutoencoderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get a list of image file paths (e.g., from the 'img' folder)
        self.image_files = []  # List of image file paths

        # Load your dataset images here

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Load image in grayscale
        if self.transform:
            image = self.transform(image)
        return image
