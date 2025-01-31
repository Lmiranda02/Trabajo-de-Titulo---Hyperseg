import os
import torch
import matplotlib.pyplot as plt
from utils import save_model_and_optimizer
import numpy as np
import time
import torch
import torch.nn as nn

def visualize_reconstructions(dataloader, model, device, n_samples=10, images_per_page=10):
    model.eval()
    data_iter = iter(dataloader)
    
    # Crear listas para almacenar las imágenes originales y reconstruidas
    original_images = []
    reconstructed_images = []
    
    with torch.no_grad():  # Desactivamos el cálculo de gradientes para la evaluación
        try:
            while len(original_images) < n_samples:  # Mientras no tengamos suficientes imágenes
                data = next(data_iter)
                data = data.to(device)
                
                # Reconstrucción
                recon_data, _ = model(data)
                
                # Seleccionar las imágenes del batch y agregarlas a la lista
                for i in range(data.size(0)):  # Iteramos sobre el batch
                    original = data[i].cpu().numpy()  # Convertir a numpy
                    reconstructed = recon_data[i].cpu().numpy()  # Convertir a numpy
                    
                    # Seleccionar las tres primeras componentes principales para simular RGB
                    original_rgb = np.clip(original[:3].transpose(1, 2, 0), 0, 1)  # Cambiar de (C, H, W) a (H, W, C)
                    reconstructed_rgb = np.clip(reconstructed[:3].transpose(1, 2, 0), 0, 1)  # Igual para reconstrucción
                    
                    # Rotar las imágenes 90 grados para mostrarlas en horizontal
                    original_rgb_rot = np.rot90(original_rgb)  # Rotar imagen original
                    reconstructed_rgb_rot = np.rot90(reconstructed_rgb)  # Rotar imagen reconstruida
                    
                    # Almacenar las imágenes
                    original_images.append(original_rgb_rot)
                    reconstructed_images.append(reconstructed_rgb_rot)
                    
                    if len(original_images) >= n_samples:  # Si ya tenemos suficientes imágenes, salir
                        break
        except StopIteration:
            print("Advertencia: El dataloader no tiene suficientes muestras para el número solicitado de imágenes.")
        
        # Ajustar el número de muestras a las disponibles
        n_samples = min(n_samples, len(original_images))
        
        # Dividir las imágenes en páginas
        total_images = len(original_images)
        num_pages = (total_images + images_per_page - 1) // images_per_page
        
        for page in range(num_pages):
            start_idx = page * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)
            current_original_images = original_images[start_idx:end_idx]
            current_reconstructed_images = reconstructed_images[start_idx:end_idx]
            
            # Crear una cuadrícula de subgráficas
            n_rows = len(current_original_images)  # Número de filas es igual al número de imágenes en la página actual
            n_cols = 2  # Dos columnas: una para las imágenes PCA y otra para las reconstruidas
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            
            # Si solo hay una fila de subgráficas
            if n_rows == 1:
                axes = np.array([axes])
            
            # Mostrar las imágenes
            for idx in range(n_rows):
                # Mostrar la imagen original (PCA) en la primera columna
                axes[idx, 0].imshow(current_original_images[idx])
                axes[idx, 0].set_title(f"Original {start_idx + idx + 1} (PCA)")
                axes[idx, 0].axis('off')
                
                # Mostrar la imagen reconstruida en la segunda columna
                axes[idx, 1].imshow(current_reconstructed_images[idx])
                axes[idx, 1].set_title(f"Reconstruida {start_idx + idx + 1}")
                axes[idx, 1].axis('off')
            
            plt.tight_layout()
            plt.show()

def train_autoencoder(train_dataloader, val_dataloader, autoencoder, optimizer, criterion, num_epochs, model_save_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    loss_history = []
    val_loss_history = []
    nmse_train_history = []
    nmse_val_history = []

    start_time = time.time()

    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        total_nmse_train = 0

        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_data, _ = autoencoder(data)

            # Usar la función de pérdida seleccionada (MSE o MAE)
            loss = criterion(recon_data, data)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # NMSE en entrenamiento
            nmse = torch.mean((data - recon_data) ** 2) / torch.mean(data ** 2)
            total_nmse_train += nmse.item()

        avg_loss = total_loss / len(train_dataloader)
        avg_nmse_train = total_nmse_train / len(train_dataloader)
        loss_history.append(avg_loss)
        nmse_train_history.append(avg_nmse_train)

        # Validación
        autoencoder.eval()
        val_loss = 0
        total_nmse_val = 0
        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                recon_data, _ = autoencoder(data)

                loss = criterion(recon_data, data)

                val_loss += loss.item()

                # NMSE en validación
                nmse = torch.mean((data - recon_data) ** 2) / torch.mean(data ** 2)
                total_nmse_val += nmse.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_nmse_val = total_nmse_val / len(val_dataloader)
        val_loss_history.append(avg_val_loss)
        nmse_val_history.append(avg_nmse_val)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, "
              f"NMSE Train: {avg_nmse_train:.6f}, NMSE Val: {avg_nmse_val:.6f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

    # Graficar la curva de loss y NMSE
    plt.figure(figsize=(10, 6))

    # Loss de entrenamiento y validación
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Loss entrenamiento')
    plt.plot(range(1, num_epochs + 1), val_loss_history, marker='x', label='Loss validación')
    plt.xlabel('Época')
    plt.ylabel('Loss promedio')
    plt.title('Curva de aprendizaje y validación')
    plt.grid(True)
    plt.legend()

    # NMSE de entrenamiento y validación
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs + 1), nmse_train_history, marker='o', label='NMSE entrenamiento')
    plt.plot(range(1, num_epochs + 1), nmse_val_history, marker='x', label='NMSE validación')
    plt.xlabel('Época')
    plt.ylabel('NMSE promedio')
    plt.title('Curva de NMSE')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Guardar el modelo y el optimizador
    save_path = os.path.join(model_save_path, f'{model_name}.pth')
    save_model_and_optimizer(autoencoder, optimizer, save_path)

    # Visualizar imágenes
    visualize_reconstructions(val_dataloader, autoencoder, device, n_samples=64, images_per_page=4)
