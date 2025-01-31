import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from Hyperspectral_image import SpectralImage
from model2 import UNetAutoencoder
from dataset import HyperSpectralDataset
import argparse

def main(img_path, model_path, n_components):
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar la imagen usando la clase SpectralImage
    img1 = SpectralImage(bilPath=img_path + ".bil", hdrPath=img_path + ".hdr")

    # Cargar el modelo de autoencoder
    autoencoder = UNetAutoencoder(input_channels=n_components).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()

    # Crear el dataset y dataloader
    dataset = HyperSpectralDataset([img_path], n_components=n_components)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Obtener el espacio latente de la imagen
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, latent_space = autoencoder(data)  # Extraer espacio latente
            latent_space = latent_space.cpu().numpy().squeeze()

    # Dimensiones originales de la imagen
    num_rows_img1 = img1.values.shape[0]
    num_cols_img1 = img1.values.shape[1]

    print(f"Dimensiones del espacio latente: {latent_space.shape}")

    # **Visualización Directa del Espacio Latente como Imagen**
    latent_rgb = latent_space[:3, :, :].transpose(1, 2, 0)  # Tomamos 3 canales para visualización
    latent_rgb_normalized = (latent_rgb - latent_rgb.min()) / (latent_rgb.max() - latent_rgb.min())

    img1_bgr = img1.BGR()

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].imshow(latent_rgb_normalized)
    axes[0].set_title("Visualización Directa del Espacio Latente")
    axes[0].axis('off')
    axes[1].imshow(img1_bgr)
    axes[1].set_title("Imagen Original BGR")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # **PCA: Análisis de Varianza**
    latent_reshaped = latent_space.reshape(latent_space.shape[0], -1).T  # Convertimos a 2D (píxeles x características)
    pca = PCA(n_components=10)  # Reducimos a 10 componentes principales
    latent_pca = pca.fit_transform(latent_reshaped)

    print(f"Varianza explicada por PCA: {pca.explained_variance_ratio_}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), pca.explained_variance_ratio_, marker='o')
    plt.xlabel("Número de Componentes Principales")
    plt.ylabel("Varianza Explicada")
    plt.title("Distribución de Varianza en el Espacio Latente (PCA)")
    plt.grid(True)
    plt.show()

    # **t-SNE: Visualización en 2D del Espacio Latente**
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_tsne = tsne.fit_transform(latent_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], s=5, alpha=0.5)
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.title("Visualización t-SNE del Espacio Latente")
    plt.grid(True)
    plt.show()

    # **Clustering en el Espacio Latente**
    n_clusters = int(input("Ingrese el número de clusters a usar para K-Means: "))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_pca)

    clustered_image = cluster_labels.reshape(num_rows_img1, num_cols_img1)

    plt.figure(figsize=(10, 6))
    plt.imshow(clustered_image, cmap='tab10')
    plt.title("Segmentación con K-Means en el Espacio Latente")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluar el espacio latente del autoencoder.')
    parser.add_argument('--img_path', type=str, required=True, help='Ruta de la imagen sin extensión.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta del modelo de autoencoder preentrenado.')
    parser.add_argument('--n_components', type=int, default=57, help='Número de componentes espectrales del autoencoder.')
    args = parser.parse_args()

    main(args.img_path, args.model_path, args.n_components)
