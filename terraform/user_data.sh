#!/bin/bash
# user_data.sh - Script to setup EC2 instance for F1Tenth RL training

set -e

# Log all output for debugging
exec > >(tee /var/log/user-data.log) 2>&1

echo "=========================================="
echo "F1Tenth RL Training Instance Setup"
echo "=========================================="
echo "Starting at: $(date)"

# Update system
echo "🔄 Updating system packages..."
apt-get update
apt-get upgrade -y

# Install basic dependencies
echo "📦 Installing basic dependencies..."
apt-get install -y \
    curl \
    wget \
    git \
    htop \
    tmux \
    vim \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    jq

# Install NVIDIA drivers and CUDA
echo "🎮 Installing NVIDIA drivers and CUDA..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-0
apt-get -y install nvidia-driver-525

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin$${PATH:+:$${PATH}}' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}' >> /home/ubuntu/.bashrc

# Install Python 3.10
echo "🐍 Installing Python 3.10..."
apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Install AWS CLI
echo "☁️  Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Configure AWS CLI region
aws configure set region ${aws_region}

# Create working directories
echo "📁 Setting up working directories..."
mkdir -p /home/ubuntu/f1tenth_training
cd /home/ubuntu/f1tenth_training

# Clone the F1Tenth Gym repository
echo "📥 Cloning F1Tenth Gym repository..."
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -e .
pip install stable-baselines3[extra]
pip install wandb
pip install tensorboard
pip install boto3

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu/f1tenth_training

# Download training scripts from S3
echo "📥 Downloading training scripts from S3..."
mkdir -p /home/ubuntu/f1tenth_training/train_cloud

# Download the SAC cloud training script
echo "📥 Downloading SAC training script..."
aws s3 cp s3://${s3_bucket_name}/scripts/sac_cloud_training.py /home/ubuntu/f1tenth_training/train_cloud/sac_cloud_training.py

# Download the SAC multi-agent cloud training script
echo "📥 Downloading SAC multi-agent training script..."
aws s3 cp s3://${s3_bucket_name}/scripts/sac_multiagent_cloud_training.py /home/ubuntu/f1tenth_training/train_cloud/sac_multiagent_cloud_training.py

# Verify downloads
if [[ -f "/home/ubuntu/f1tenth_training/train_cloud/sac_cloud_training.py" ]]; then
    echo "✅ SAC training script downloaded successfully"
else
    echo "❌ Failed to download SAC training script"
    exit 1
fi

if [[ -f "/home/ubuntu/f1tenth_training/train_cloud/sac_multiagent_cloud_training.py" ]]; then
    echo "✅ SAC multi-agent training script downloaded successfully"
else
    echo "❌ Failed to download SAC multi-agent training script"
    exit 1
fi

# Make training scripts executable
chmod +x /home/ubuntu/f1tenth_training/train_cloud/*.py

# Create systemd service for automatic training
echo "⚙️  Creating systemd service..."
cat > /etc/systemd/system/f1tenth-training.service << SERVICE_EOF
[Unit]
Description=F1Tenth RL Training Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/f1tenth_training/f1tenth_gym
Environment=VIRTUAL_ENV=/home/ubuntu/f1tenth_training/f1tenth_gym/venv
Environment=PATH=/home/ubuntu/f1tenth_training/f1tenth_gym/venv/bin:/usr/local/cuda-12.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64
Environment=S3_BUCKET_NAME=${s3_bucket_name}
Environment=TOTAL_TIMESTEPS=${total_timesteps}
Environment=SAVE_FREQUENCY=${save_frequency}
Environment=TRAINING_TYPE=${training_type}
Environment=AWS_DEFAULT_REGION=${aws_region}
Environment=NUM_AGENTS=2
Environment=AUTO_SHUTDOWN=true
ExecStartPre=/bin/sleep 60
ExecStart=/bin/bash -c 'source venv/bin/activate && cd /home/ubuntu/f1tenth_training/train_cloud && if [ "${training_type}" = "sac_multiagent" ]; then python3 sac_multiagent_cloud_training.py; else python3 sac_cloud_training.py; fi'
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable f1tenth-training.service

echo "✅ F1Tenth RL Training Instance Setup Complete!"
echo "📊 Training will start automatically in ~1 minute"
echo "📈 Monitor with: sudo journalctl -u f1tenth-training -f"
echo "🏁 Instance will auto-shutdown when training completes"
echo "=========================================="
