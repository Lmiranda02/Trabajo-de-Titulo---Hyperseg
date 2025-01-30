import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dataset import HyperSpectralDataset
from model import UNetAutoencoder

def extract_latent_space(dataloader, model, device):
    model.eval()
    latent_spaces = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, latent = model(data)
            latent_spaces.append(latent.cpu().numpy())
    latent_spaces = np.concatenate(latent_spaces, axis=0)
    # Aplanar las dimensiones espaciales
    batch_size, channels, height, width = latent_spaces.shape
    latent_spaces = latent_spaces.reshape(batch_size, channels, -1)
    latent_spaces = latent_spaces.transpose(0, 2, 1).reshape(-1, channels)
    return latent_spaces

def apply_clustering(latent_space, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(latent_space)
    return clusters

def visualize_clusters(image, clusters, height, width, n_clusters):
    plt.figure(figsize=(10, 10))
    clusters = clusters.reshape(height, width)
    for i in range(n_clusters):
        mask = clusters == i
        plt.subplot(1, n_clusters, i + 1)
        plt.imshow(image.transpose(1, 2, 0) * mask[:, :, np.newaxis])
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.show()

def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'C:\Users\Marco\Desktop\Universidad Luis\Trabajo de Titulo\Modelos\exp500.pth'
    image_dir = r'C:\Users\Marco\Desktop\Universidad Luis\Trabajo de Titulo\Imagenes Hiperespectrales\Pruebas'
    image_paths = [os.path.splitext(os.path.join(image_dir, f))[0] for f in os.listdir(image_dir) if f.endswith('.bil')]
    
    # Cargar el modelo
    autoencoder = UNetAutoencoder(input_channels=57).to(device)
    checkpoint = torch.load(model_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Cargar el dataset
    dataset = HyperSpectralDataset(image_paths, n_components=57)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Extraer el espacio latente
    latent_space = extract_latent_space(dataloader, autoencoder, device)
    
    # Aplicar clustering
    clusters = apply_clustering(latent_space, n_clusters=4)
    
    # Visualizar clusters
    for i, data in enumerate(dataloader):
        image = data[0].cpu().numpy()
        height, width = image.shape[1], image.shape[2]
        visualize_clusters(image, clusters[i * height * width:(i + 1) * height * width], height, width, 4)
        selected_cluster = int(input("Seleccione el cluster que agrupa las cerezas: "))
        
        # Aplicar máscara
        mask = clusters[i * height * width:(i + 1) * height * width].reshape(height, width) == selected_cluster
        masked_image = image * mask[np.newaxis, :, :]
        plt.imshow(masked_image.transpose(1, 2, 0))
        plt.title('Imagen con máscara aplicada')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()