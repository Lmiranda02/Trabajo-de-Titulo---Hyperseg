import torch
import os

def save_model_and_optimizer(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Modelo y optimizador guardados en: {path}")

def load_model_and_optimizer(model_class, optimizer_class, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = model_class().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optimizer_class(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Modelo y optimizador cargados desde: {path}")
    return model, optimizer