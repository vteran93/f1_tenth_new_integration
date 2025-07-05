# SOLUCIÓN DEFINITIVA: PyTorch CPU-only
# Para resolver el bug de memory corruption con CUDA

# Desinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio -y

# Instalar PyTorch CPU-only (versión estable)
pip install torch==2.7.1+cpu torchvision==0.18.1+cpu torchaudio==2.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# O alternativamente (método más simple):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ PyTorch CPU-only instalado"
echo "🔄 Reiniciar entorno virtual después de la instalación"
