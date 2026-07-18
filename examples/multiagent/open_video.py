"""
Script para abrir el video del mejor episodio en el navegador web.
"""

import os
import webbrowser
from pathlib import Path
import tempfile
import shutil


def open_best_episode_video(experiment_name):
    """
    Abre el video del mejor episodio en el navegador web.
    """
    # Buscar el archivo de video
    base_path = Path("models_deposito") / experiment_name / "eval_runs"
    video_path = base_path / "best_episode.mp4"
    
    if not video_path.exists():
        print(f"❌ Video no encontrado en: {video_path}")
        print("Ejecuta primero: python render_best.py --experiment <name> --mode video")
        return
    
    print(f"🎬 Abriendo video del mejor episodio...")
    print(f"📁 Ubicación: {video_path.absolute()}")
    
    # Obtener información del video
    file_size = video_path.stat().st_size / (1024 * 1024)  # MB
    print(f"📊 Tamaño: {file_size:.1f} MB")
    
    # Cargar resumen si existe
    summary_path = base_path / "mass_eval_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        best = summary.get('best', {})
        print(f"🏆 Mejor episodio:")
        print(f"   Seed: {best.get('seed', 'N/A')}")
        print(f"   Max Lap Progress: {best.get('max_lap_progress', 'N/A'):.3f}")
        print(f"   Retorno: {best.get('return', 'N/A'):.2f}")
        print(f"   Longitud: {best.get('length', 'N/A')} pasos")
        print(f"   Colisiones: {best.get('collisions', 'N/A')}")
    
    # Abrir en el navegador
    video_url = f"file://{video_path.absolute()}"
    print(f"🌐 Abriendo en navegador: {video_url}")
    
    try:
        webbrowser.open(video_url)
        print("✅ Video abierto en el navegador")
        print("💡 Si no se abre automáticamente, copia esta ruta:")
        print(f"   {video_path.absolute()}")
    except Exception as e:
        print(f"❌ Error abriendo navegador: {e}")
        print(f"📂 Puedes abrir manualmente el archivo: {video_path.absolute()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python open_video.py <experiment_name>")
        print("Ejemplo: python open_video.py Spielberg_PPO_Individual_Policy_ProgressRewardAdvanced")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    open_best_episode_video(experiment_name)