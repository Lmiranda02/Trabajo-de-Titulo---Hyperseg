import torch
from torch.utils.data import Dataset
from Hyperspectral_image import SpectralImage
from sklearn.decomposition import IncrementalPCA
import numpy as np

class HyperSpectralDataset(Dataset):
    def __init__(self, image_paths, n_components=57):
        self.image_paths = image_paths
        self.n_components = n_components
        self.spectral_images = []
        print("Cargando Dataset Porfavor Espere...")
        for i, img_path in enumerate(self.image_paths):
            print(f"Aplicando PCA a imagen {i+1}/{len(self.image_paths)} ...")
            self.spectral_images.append(self.load_and_process_image(img_path))

    def load_and_process_image(self, img_path):
        spectral_image = SpectralImage(img_path + ".bil", img_path + ".hdr")
        image_data = spectral_image.values
        reshaped_data = image_data.reshape(-1, image_data.shape[2])
        pca = IncrementalPCA(n_components=self.n_components)
        reduced_data = pca.fit_transform(reshaped_data)
        reduced_data = (reduced_data - reduced_data.min()) / (reduced_data.max() - reduced_data.min())
        reduced_data = reduced_data.reshape(image_data.shape[0], image_data.shape[1], self.n_components)
        reduced_data = reduced_data.transpose(2, 0, 1)

        return torch.tensor(reduced_data, dtype=torch.float32)

    def __len__(self):
        return len(self.spectral_images)

    def __getitem__(self, idx):
        return self.spectral_images[idx]