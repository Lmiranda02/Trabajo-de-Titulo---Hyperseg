import spectral.io.envi as envi
from getpass import getuser
import numpy as np
from variables import wavelengths
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import platform
import os

def normalize(im):
    min, max = im.min(), im.max()
    return (im.astype(float)-min)/(max-min)

def extract_filename(filepath):
    filename = filepath.split("/")[-1]
    if len(filename) > 17:
        filename = "..." + filename[-7:]
    return filename

def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R, G, B = (0,0,0)
    
    if (wavelength < 380) or (wavelength > 750):
        return (0.5, 0.5, 0.5)

    if (wavelength >= 380) and (wavelength < 440):
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        R = 1.0
        G = 0.0
        B = 0.0
        
    if (wavelength >= 380) and (wavelength < 420):
        factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
    elif (wavelength >= 420) and (wavelength < 645):
        factor = 1.0
    elif (wavelength >= 645) and (wavelength <= 750):
        factor = 0.3 + 0.7*(750 - wavelength) / (750 - 645)
        
    R = round(intensity_max * (R * factor)**gamma)
    G = round(intensity_max * (G * factor)**gamma)
    B = round(intensity_max * (B * factor)**gamma)
    
    return (R/255.0, G/255.0, B/255.0)

def simple_mean(l:list)->float:
    return sum(l)/len(l)

class SpectralImage():
    def __init__(self: None, bilPath: str, hdrPath: str) -> None:
        self.bil = bilPath
        self.hdr = hdrPath
        leaf_ref = envi.open(self.hdr, self.bil)
        dats = np.asarray(leaf_ref.load())
        self.values = np.flipud(dats)
        self._bgr_calculated = None  # Inicializado en None
    
    def extract_metadata(self):
        # Extraer los metadatos del archivo .hdr
        hdr = envi.read_envi_header(self.hdr)
        
        # Obtener el nombre base del archivo sin extensión y usarlo para el archivo de metadatos
        image_name = os.path.basename(self.bil).split('.')[0]
        
        # Crear el archivo de salida en la carpeta "EDA" con el nombre de la imagen
        output_dir = "EDA"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{image_name}_metadata.txt")
        
        # Guardar los metadatos en un archivo de texto
        with open(output_file, 'w') as file:
            for key, value in hdr.items():
                file.write(f"{key}: {value}\n")
        
        print(f"Metadatos guardados en: {output_file}")
        return hdr
        
    def extract_red_channel(self):
        # Buscar los índices de las bandas correspondientes al canal rojo
        red_indices = [i for i, wl in enumerate(wavelengths) if 620 <= wl <= 750]
        
        if not red_indices:
            raise ValueError("No se encontraron bandas correspondientes al rango del canal rojo.")
        
        # Promediar las bandas del canal rojo para obtener la imagen del canal rojo
        red_band = np.mean(self.values[:, :, red_indices], axis=2)
        
        # Normalizar la imagen
        red_band_normalized = (red_band - red_band.min()) / (red_band.max() - red_band.min())
        
        return red_band_normalized
    
    def BGR(self):
        # Si ya se ha calculado previamente, simplemente retornar el valor
        if self._bgr_calculated is not None:
            return self._bgr_calculated

        aux_r = np.mean(self.values[0][:, 118:160], axis=1)
        aux_g = np.mean(self.values[0][:, 54:70], axis=1)
        aux_b = np.mean(self.values[0][:, 19:45], axis=1)
        for i in range(1, self.values.shape[0]):
            aux_r = np.concatenate((aux_r, np.mean(self.values[i][:, 118:150], axis=1)))
            aux_g = np.concatenate((aux_g, np.mean(self.values[i][:, 54:80], axis=1)))
            aux_b = np.concatenate((aux_b, np.mean(self.values[i][:, 19:45], axis=1)))
        
        R = normalize(aux_r).reshape(self.values.shape[:2])
        G = normalize(aux_g).reshape(self.values.shape[:2])
        B = normalize(aux_b).reshape(self.values.shape[:2])
        RGB = np.dstack((R, G, B))
        self._bgr_calculated = np.transpose(RGB, (1, 0, 2)).astype(np.float32)
        print(self._bgr_calculated[0,0])
        return self._bgr_calculated

    def applyMask(self):pass
    def reduceChannels(self):pass


if __name__ == "__main__":pass