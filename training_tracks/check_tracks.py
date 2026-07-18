#!/usr/bin/env python3
"""
Script simple para verificar las pistas generadas
"""

import os
import numpy as np
from PIL import Image


def analyze_track(track_name):
    """Analiza una pista específica"""

    track_dir = f"./tracks/{track_name}"
    img_path = f"{track_dir}/{track_name}_map.png"
    yaml_path = f"{track_dir}/{track_name}_map.yaml"

    if not os.path.exists(img_path):
        print(f"❌ No se encontró la imagen de la pista: {img_path}")
        return False

    if not os.path.exists(yaml_path):
        print(f"❌ No se encontró el archivo YAML: {yaml_path}")
        return False

    # Analizar la imagen
    img = Image.open(img_path)
    img_array = np.array(img)

    height, width = img_array.shape
    track_pixels = np.sum(img_array > 128)  # Píxeles blancos (pista)
    total_pixels = height * width
    track_percentage = (track_pixels / total_pixels) * 100

    print(f"\n🏁 Análisis de la pista: {track_name}")
    print("=" * 50)
    print(f"📐 Dimensiones: {width} x {height} píxeles")
    print(f"🛣️  Píxeles de pista: {track_pixels:,}")
    print(f"📊 Porcentaje de pista: {track_percentage:.2f}%")
    print(f"🗂️  Archivos generados: ✅ PNG, ✅ YAML")

    # Verificar que la pista tiene forma cerrada
    # Contar componentes conectados
    from scipy import ndimage
    labeled, num_features = ndimage.label(img_array > 128)

    if num_features == 1:
        print("🔄 Pista: ✅ Circuito cerrado (1 componente)")
    else:
        print(f"⚠️  Pista: {num_features} componentes separados")

    return True


def check_all_tracks():
    """Verifica todas las pistas generadas"""

    tracks_dir = "./tracks"
    if not os.path.exists(tracks_dir):
        print("❌ No se encontró el directorio de pistas")
        return

    track_names = [d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))]

    print(f"🔍 Verificando {len(track_names)} pistas generadas...")

    success_count = 0
    for track_name in sorted(track_names):
        if analyze_track(track_name):
            success_count += 1

    print(f"\n📊 Resumen: {success_count}/{len(track_names)} pistas generadas correctamente")

    if success_count == len(track_names):
        print("✅ ¡Todas las pistas se generaron exitosamente!")
    else:
        print("⚠️  Algunas pistas tienen problemas")


if __name__ == "__main__":
    check_all_tracks()
