import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Hyperspectral_image import SpectralImage
import argparse

def main(img_path, n_components):
    # Cargar la imagen usando la clase SpectralImage
    img1 = SpectralImage(bilPath=img_path + ".bil", hdrPath=img_path + ".hdr")

    # Transformar los datos para PCA
    img1_reshape = img1.values.reshape(-1, img1.values.shape[2])
    print(f"El nuevo tamaño de los datos es: {img1_reshape.shape}")

    # Estandarizar los datos
    scaler = StandardScaler()
    img1_reshape_ss = scaler.fit_transform(img1_reshape)

    # Aplicar PCA para reducir a n_components componentes
    pca = PCA(n_components=n_components)
    img1_reshape_ss_pca = pca.fit_transform(img1_reshape_ss)
    explained_variance = pca.explained_variance_ratio_

    # Graficar la varianza explicada acumulativa
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulativa')
    plt.title('Varianza Explicada por PCA')
    plt.grid(True)
    plt.show()

    # Redimensionar los datos a las dimensiones originales para visualización en BGR
    num_rows_img1 = img1.values.shape[0]
    num_cols_img1 = img1.values.shape[1]

    # Seleccionar las primeras tres componentes principales para BGR
    img1_pca_bgr = img1_reshape_ss_pca[:, :3].reshape(num_rows_img1, num_cols_img1, 3)

    # Normalizar para mejorar la visualización
    img1_pca_bgr_normalized = (img1_pca_bgr - img1_pca_bgr.min()) / (img1_pca_bgr.max() - img1_pca_bgr.min())
    img1_pca_bgr_normalized = img1_pca_bgr_normalized.transpose(1, 0, 2)

    # Obtener la imagen original en BGR
    img1_bgr = img1.BGR()

    # Visualizar la imagen generada con PCA y la original
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].imshow(img1_pca_bgr_normalized)
    axes[0].set_title("Imagen Generada con PCA")
    axes[0].axis('off')
    axes[1].imshow(img1_bgr)
    axes[1].set_title("Imagen Original BGR")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Método del codo para determinar el número óptimo de clusters
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(img1_reshape_ss_pca)
        inertias.append(kmeans.inertia_)

    # Graficar la inercia en función del número de clusters
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
    plt.grid(True)
    plt.show()

    # Pedir al usuario que ingrese el número óptimo de clusters según el gráfico del codo
    n_clusters = int(input("Ingrese el número óptimo de clusters según el gráfico del codo: "))

    # Aplicar K-means para clustering con el número de clusters especificado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img1_reshape_ss_pca)

    # Etiquetas de los clusters
    cluster_labels = kmeans.labels_

    # Reorganizar las etiquetas a la forma original
    clustered_image = cluster_labels.reshape(num_rows_img1, num_cols_img1).T

    # Visualizar la imagen con los clusters
    plt.figure(figsize=(18, 5))
    plt.imshow(clustered_image)
    plt.title("Imagen Segmentada con K-means")
    plt.axis('off')
    plt.show()

    # Visualizar cada cluster individualmente
    fig, axes = plt.subplots(1, n_clusters, figsize=(20, 3))
    fig.suptitle("Visualización de Cada Cluster")
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_image = cluster_mask.reshape(num_rows_img1, num_cols_img1).T
        axes[cluster_id].imshow(cluster_image, cmap='gray')
        axes[cluster_id].set_title(f'Cluster {cluster_id}')
        axes[cluster_id].axis('off')
    plt.tight_layout()
    plt.show()

    # Pedir al usuario que seleccione el cluster que resalta mejor los cerezos
    selected_cluster = int(input("Seleccione el cluster que agrupa las cerezas: "))

    # Crear la máscara binaria para el cluster seleccionado y ajustarla a las dimensiones correctas
    mask_cerezas = (cluster_labels == selected_cluster)
    mask_cerezas = mask_cerezas.reshape(img1_bgr.shape[1], img1_bgr.shape[0])
    mask_cerezas = mask_cerezas.T  # Transponer para que coincida con la orientación de la imagen BGR

    # Crear una copia de la imagen original en BGR para aplicar la máscara
    img_bgr_with_mask = img1_bgr.copy()

    # Definir el color que deseas para las cerezas (en formato BGR)
    cereza_color = [0, 255, 0]  # Verde brillante

    # Aplicar el color a los píxeles correspondientes a las cerezas en la máscara
    img_bgr_with_mask[mask_cerezas] = cereza_color

    # Crear un plot con la imagen original y la imagen con la máscara
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Mostrar la imagen original
    axes[0].imshow(img1_bgr)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')

    # Mostrar la imagen con la máscara aplicada
    axes[1].imshow(img_bgr_with_mask)
    axes[1].set_title("Imagen con Cerezos Resaltados")
    axes[1].axis('off')

    # Mostrar la inercia del cluster seleccionado
    selected_cluster_inertia = kmeans.inertia_
    plt.figtext(0.5, 0.01, f'Inercia del Cluster Seleccionado: {selected_cluster_inertia}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Imprimir la inercia del cluster seleccionado
    print(f'Inercia del Cluster Seleccionado: {selected_cluster_inertia}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar imagen hiperespectral con PCA y K-means.')
    parser.add_argument('--img_path', type=str, required=True, help='Ruta de la imagen sin extensión.')
    parser.add_argument('--n_components', type=int, default=57, help='Número de componentes principales para PCA.')
    args = parser.parse_args()

    main(args.img_path, args.n_components)