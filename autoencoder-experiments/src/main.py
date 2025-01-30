import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HyperSpectralDataset
from train import train_autoencoder, PerceptualLoss  # ✅ Importamos PerceptualLoss
import argparse
import os

def run_experiment(batch_size, optimizer_class, learning_rate, model_name, model_version):
    image_dir = r'C:\Users\Marco\Desktop\Universidad Luis\Trabajo de Titulo\Imagenes Hiperespectrales\dataset_final_brillo'
    image_paths = [os.path.splitext(os.path.join(image_dir, f))[0] for f in os.listdir(image_dir) if f.endswith('.bil')]
    dataset = HyperSpectralDataset(image_paths, n_components=57)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Seleccionar el modelo según el argumento `--model_version`
    if model_version == "v1":
        from model import UNetAutoencoder
    elif model_version == "v2":
        from model2 import UNetAutoencoder
    else:
        raise ValueError("Versión de modelo no válida. Usa 'v1' para el modelo original o 'v2' para el mejorado.")

    # Instanciar el autoencoder correcto
    autoencoder = UNetAutoencoder(input_channels=57).to(device)

    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)

    # ✅ Definir la nueva función de pérdida combinada
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)
    
    def combined_loss(recon, target, step=0):
        mse = mse_loss(recon, target)
        perceptual = perceptual_loss(recon, target) if step % 5 == 0 else 0  # Aplicar cada 5 iteraciones
        return mse + 0.1 * perceptual


    num_epochs = 50
    model_save_path = r'C:\Users\Marco\Desktop\Universidad Luis\Trabajo de Titulo\Modelos'
    train_autoencoder(train_dataloader, val_dataloader, autoencoder, optimizer, combined_loss, num_epochs, model_save_path, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run autoencoder experiments.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training and validation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--model_version', type=str, required=True, choices=['v1', 'v2'], help="Versión del modelo ('v1' para el original, 'v2' para el mejorado)")

    args = parser.parse_args()

    optimizer_class = optim.Adam if args.optimizer == 'adam' else optim.SGD
    model_name = input("Ingrese el nombre del modelo: ")

    run_experiment(args.batch_size, optimizer_class, args.learning_rate, model_name, args.model_version)
