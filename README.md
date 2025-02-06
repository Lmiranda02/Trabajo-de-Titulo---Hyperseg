# Segmentación de Imágenes Hiperespectrales con Autoencoders y Clustering

Este repositorio contiene una implementación de segmentación de imágenes hiperespectrales mediante **autoencoders convolucionales** y **clustering**. Se emplea un pipeline que combina reducción de dimensionalidad, entrenamiento de modelos de deep learning y análisis en el espacio latente.

---

## 📂 Estructura del Proyecto

```
Trabajo de Titulo - Hyperseg/
│   requirements.txt  # Librerías necesarias para ejecutar el código
│
├───resultados/  # Carpeta con gráficos y reconstrucciones de los experimentos
│   ├───Graficos Experimentos/  # Imágenes de la evolución de la pérdida y desempeño
│   └───Reconstrucciones/  # Resultados de reconstrucción de imágenes hiperespectrales
│
└───src/
    │   config.json  # Configuración del proyecto (directorios y parámetros)
    │   dataset.py  # Manejo del dataset y reducción de dimensionalidad con PCA
    │   evaluar_latente.py  # Evaluación del espacio latente del autoencoder
    │   Hyperspectral_image.py  # Carga y manipulación de imágenes hiperespectrales
    │   main.py  # Script principal para entrenar modelos
    │   model.py  # Definición del modelo U-Net Autoencoder
    │   model2.py  # Autoencoder sin skip connections
    │   pca_kmeans_segmentation.py  # Segmentación con PCA y K-Means
    │   process_images.py  # Preprocesamiento y división de imágenes hiperespectrales
    │   train.py  # Entrenamiento del autoencoder
    │   utils.py  # Funciones auxiliares para guardar/cargar modelos
    │   variables.py  # Parámetros de longitudes de onda
```

---

## 🛠 Instalación

### Requisitos
1. Python 3.8+
2. Instalar dependencias con:
   ```sh
   pip install -r requirements.txt
   ```

---
## Configuración

Antes de ejecutar los scripts, asegúrate de configurar los paths en el archivo `config.json`. Puedes usar el archivo de plantilla `config_template.json` como referencia:

1. Copia el archivo de plantilla:
    ```sh
    cp src/config_template.json src/config.json
    ```

2. Edita el archivo config.json y actualiza los paths según tu entorno local:
    ```json
    {
        "image_dir": "ruta/a/tu/directorio/de/imagenes",
        "model_save_path": "ruta/a/tu/directorio/de/modelos",
        "num_epochs": 50
    }
    ```
---

## 🚀 Uso del Proyecto

### **1️⃣ Entrenar el Autoencoder**
Ejecutar `main.py` para entrenar el autoencoder:
```sh
python main.py --batch_size 3 --optimizer adam --learning_rate 1e-4 --criterion mse --model_version v1 --num_epochs 50
```

- `--model_version` puede ser `v1` (con skip connections) o `v2` (sin skip connections).
- Los modelos entrenados se guardan en la carpeta especificada en `config.json`.

---

### **2️⃣ Evaluar el Espacio Latente**
Después de entrenar un modelo, se puede analizar la representación latente:
```sh
python evaluar_latente.py --img_path "ruta/a/imagen" --model_path "ruta/al/modelo.pth" --model_version v1 --n_components 57
```

Este script:
- Extrae el **espacio latente** de la imagen.
- Aplica **PCA y t-SNE** para visualizar su distribución.
- Ejecuta **K-Means** sobre el espacio latente para segmentación.

---

### **3️⃣ Segmentación con PCA y K-Means**
Si deseas segmentar usando PCA y K-Means en vez del autoencoder:
```sh
python pca_kmeans_segmentation.py --img_path "ruta/a/imagen" --n_components 57
```

Este script:
- Reduce la dimensionalidad con PCA.
- Segmenta la imagen con **K-Means**.
- Permite seleccionar manualmente el **clúster de las cerezas**.

---

## 📊 Resultados y Evaluación
Los resultados generados (gráficos y reconstrucciones) se almacenan en `resultados/`.

---

## 🔍 Próximos Pasos
- Explorar arquitecturas más avanzadas sin skip connections.
- Probar funciones de pérdida alternativas (contrastive loss, perceptual loss).
- Comparar la segmentación con métodos supervisados como redes Fully Convolutional (FCN).

---

## 📄 Créditos
Proyecto desarrollado por **Luis Alberto Miranda De La Guarda** como parte de su **Trabajo de Título en Ingeniería Civil en Computación** en la Universidad de O'Higgins. Profesores guía: **Rodrigo Verschae y Luis Cossio**.

