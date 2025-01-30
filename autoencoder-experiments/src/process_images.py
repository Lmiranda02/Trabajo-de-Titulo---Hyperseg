import os
import numpy as np
from spectral.io import envi
from Hyperspectral_image import SpectralImage
import argparse

def divide_image(data, hdr_metadata, output_folder, filename, num_divisions=4):
    """
    Divide una imagen hiperespectral en subimágenes y guarda cada subimagen con la extensión correcta.
    
    Parámetros:
    - data: array de datos de la imagen hiperespectral.
    - hdr_metadata: metadatos de la imagen.
    - output_folder: carpeta donde se guardarán las subimágenes.
    - filename: nombre base del archivo para las subimágenes.
    - num_divisions: número de divisiones a realizar en cada dimensión (por defecto, 4).
    """
    
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Dimensiones de cada subimagen
    height, width, num_bands = data.shape
    sub_height = height // num_divisions
    sub_width = width // num_divisions
    
    # Iterar sobre las divisiones para crear y guardar cada subimagen
    for i in range(num_divisions):
        for j in range(num_divisions):
            # Definir las coordenadas de la subimagen
            start_row = i * sub_height
            end_row = (i + 1) * sub_height
            start_col = j * sub_width
            end_col = (j + 1) * sub_width
            
            # Extraer la subimagen
            subimage = data[start_row:end_row, start_col:end_col, :]
            
            # Generar los nombres de los archivos de salida
            output_bil = os.path.join(output_folder, f"{filename}_subimage_{i}_{j}.bil")
            output_hdr = os.path.join(output_folder, f"{filename}_subimage_{i}_{j}.hdr")
            
            # Guardar la subimagen en formato .bil
            envi.save_image(output_hdr, subimage, dtype=subimage.dtype, ext='.bil', force=True, metadata=hdr_metadata)
            print(f"Subimagen guardada en: {output_bil} y {output_hdr}")

def increase_brightness(image, value=50):
    """
    Aumenta el brillo de una imagen sumando un valor a todos los píxeles y truncando en 255.
    
    Parámetros:
    - image: imagen en formato numpy array.
    - value: valor a sumar a cada píxel (por defecto, 50).
    """
    bright_image = np.clip(image + value, 0, 255)
    return bright_image

def process_all_images(main_folder, output_main_folder, num_divisions=4, brightness_value=50):
    """
    Procesa todas las imágenes en la carpeta principal, aumentando su brillo y luego dividiéndolas.
    
    Parámetros:
    - main_folder: carpeta principal que contiene las imágenes.
    - output_main_folder: carpeta donde se guardarán las subimágenes procesadas.
    - num_divisions: número de divisiones a realizar en cada dimensión (por defecto, 4).
    - brightness_value: valor a sumar a cada píxel para aumentar el brillo (por defecto, 50).
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_main_folder):
        os.makedirs(output_main_folder)

    # Recorrer todas las subcarpetas en la carpeta principal
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".bil") and not file.endswith(".bil.bil"):
                bil_path = os.path.join(root, file)
                hdr_path = bil_path + ".hdr"
                
                # Verificar si el archivo .hdr existe
                if os.path.exists(hdr_path):
                    # Cargar la imagen y aumentar el brillo
                    img = envi.open(hdr_path, bil_path)
                    data = img.load()
                    hdr_metadata = envi.read_envi_header(hdr_path)
                    bright_data = increase_brightness(data, value=brightness_value)
                    
                    # Obtener el nombre base del archivo
                    filename = os.path.basename(bil_path).replace('.bil', '')
                    
                    # Dividir la imagen procesada
                    divide_image(bright_data, hdr_metadata, output_main_folder, filename, num_divisions=num_divisions)
                    print(f"Procesamiento completado para {file}")
                else:
                    print(f"Archivo HDR no encontrado para {file}")

def main():
    parser = argparse.ArgumentParser(description='Aumenta el brillo de imágenes hiperespectrales y las divide.')
    parser.add_argument('--main_folder', type=str, required=True, help='Ruta de la carpeta principal que contiene las imágenes.')
    parser.add_argument('--output_folder', type=str, required=True, help='Ruta de la carpeta donde se guardarán las subimágenes procesadas.')
    parser.add_argument('--num_divisions', type=int, default=4, help='Número de divisiones a realizar en cada dimensión (por defecto, 4).')
    parser.add_argument('--brightness_value', type=int, default=50, help='Valor a sumar a cada píxel para aumentar el brillo (por defecto, 50).')
    args = parser.parse_args()

    process_all_images(args.main_folder, args.output_folder, num_divisions=args.num_divisions, brightness_value=args.brightness_value)

if __name__ == "__main__":
    main()
