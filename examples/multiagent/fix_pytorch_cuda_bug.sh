# DEFINITIVE SOLUTION: PyTorch CPU-only
# To resolve the memory corruption bug with CUDA

# Uninstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y

# Install PyTorch CPU-only (stable version)
pip install torch==2.7.1+cpu torchvision==0.18.1+cpu torchaudio==2.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Or alternatively (simpler method):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ PyTorch CPU-only installed"
echo "🔄 Restart virtual environment