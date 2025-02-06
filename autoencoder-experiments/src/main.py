import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HyperSpectralDataset
from train import train_autoencoder
import argparse
import os
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_experiment(batch_size, optimizer_class, learning_rate, criterion_class, model_name, model_version, num_epochs, config):
    image_dir = config['image_dir']
    model_save_path = config['model_save_path']
    
    image_paths = [os.path.splitext(os.path.join(image_dir, f))[0] for f in os.listdir(image_dir) if f.endswith('.bil')]
    dataset = HyperSpectralDataset(image_paths, n_components=57)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diccionario para mapear versiones del modelo a sus respectivas clases
    model_versions = {
        "v1": "model.UNetAutoencoder",
        "v2": "model2.UNetAutoencoder",
        "v3": "model3.UNetAutoencoder",
    }

    if model_version not in model_versions:
        raise ValueError("Versión de modelo no válida. Usa 'v1' para el modelo original o 'v2' para el mejorado.")

    # Importar la clase del modelo dinámicamente
    module_name, class_name = model_versions[model_version].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(module, class_name)

    # Instanciar el autoencoder correcto
    autoencoder = ModelClass(input_channels=57).to(device)

    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    criterion = criterion_class()

    train_autoencoder(train_dataloader, val_dataloader, autoencoder, optimizer, criterion, num_epochs, model_save_path, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run autoencoder experiments.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training and validation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'mae'], help='Loss criterion to use')
    parser.add_argument('--model_version', type=str, required=True, choices=['v1', 'v2'], help="Versión del modelo ('v1' para el original, 'v2' para el mejorado)")
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')

    args = parser.parse_args()

    config = load_config(args.config_path)

    optimizer_class = optim.Adam if args.optimizer == 'adam' else optim.SGD
    criterion_class = nn.MSELoss if args.criterion == 'mse' else nn.L1Loss  

    model_name = input("Ingrese el nombre del modelo: ")

    num_epochs = args.num_epochs if args.num_epochs else config['num_epochs']

    run_experiment(args.batch_size, optimizer_class, args.learning_rate, criterion_class, model_name, args.model_version, num_epochs, config)
