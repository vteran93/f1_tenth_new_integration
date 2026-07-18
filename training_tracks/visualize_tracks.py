#!/usr/bin/env python3
"""
Script para visualizar las pistas generadas
"""

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


def visualize_all_tracks():
    """Visualiza todas las pistas generadas"""

    tracks_dir = "./tracks"

    if not os.path.exists(tracks_dir):
        print("❌ No se encontró el directorio de pistas. Ejecuta create_tracks.py primero.")
        return

    # Obtener lista de pistas
    track_names = [d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))]

    if not track_names:
        print("❌ No se encontraron pistas en el directorio.")
        return

    print(f"📊 Visualizando {len(track_names)} pistas...")

    # Crear subplot para mostrar todas las pistas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, track_name in enumerate(track_names):
        if i >= len(axes):
            break

        track_image_path = os.path.join(tracks_dir, track_name, f"{track_name}_map.png")

        if os.path.exists(track_image_path):
            # Cargar y mostrar la imagen
            img = Image.open(track_image_path)
            img_array = np.array(img)

            axes[i].imshow(img_array, cmap='gray')
            axes[i].set_title(f"Pista: {track_name.replace('_', ' ').title()}", fontsize=12, pad=10)
            axes[i].axis('off')

            # Añadir información sobre la pista
            height, width = img_array.shape
            track_pixels = np.sum(img_array > 128)  # Píxeles blancos (pista)
            total_pixels = height * width
            track_percentage = (track_pixels / total_pixels) * 100

            axes[i].text(10, height - 20,
                         f"Dimensiones: {width}x{height}\nÁrea pista: {track_percentage:.1f}%",
                         fontsize=9, color='red',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            axes[i].text(0.5, 0.5, f"❌ No encontrada:\n{track_name}",
                         transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f"Error: {track_name}")

    # Ocultar ejes no utilizados
    for j in range(len(track_names), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("🏁 Pistas Generadas para F1TENTH", fontsize=16, y=0.98)
    plt.show()

    print("✅ Visualización completada!")


def show_track_details():
    """Muestra detalles técnicos de cada pista"""

    tracks_dir = "./tracks"
    track_names = [d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))]

    print("\n📋 Detalles de las Pistas Generadas:")
    print("=" * 60)

    for track_name in track_names:
        yaml_path = os.path.join(tracks_dir, track_name, f"{track_name}_map.yaml")
        img_path = os.path.join(tracks_dir, track_name, f"{track_name}_map.png")

        if os.path.exists(yaml_path) and os.path.exists(img_path):
            # Leer configuración YAML
            import yaml
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            # Leer imagen para obtener dimensiones
            img = Image.open(img_path)
            width, height = img.size

            print(f"\n🏁 {track_name.replace('_', ' ').title()}")
            print(f"   📁 Archivos: {track_name}_map.png, {track_name}_map.yaml")
            print(f"   📐 Dimensiones: {width} x {height} píxeles")
            print(f"   📏 Resolución: {config['resolution']} m/píxel")
            print(f"   🌍 Origen: {config['origin']}")
            print(f"   📊 Tamaño real: {width * config['resolution']:.1f} x {height * config['resolution']:.1f} metros")
        else:
            print(f"\n❌ {track_name}: Archivos incompletos")


if __name__ == "__main__":
    print("🏁 Visualizador de Pistas F1TENTH")
    print("=" * 40)

    show_track_details()

    try:
        visualize_all_tracks()
    except Exception as e:
        print(f"❌ Error al visualizar: {e}")
        print("💡 Asegúrate de tener matplotlib instalado: pip install matplotlib")
