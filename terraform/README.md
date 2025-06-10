# F1Tenth RL Cloud Training Infrastructure

This repository provides a complete infrastructure setup for training F1Tenth reinforcement learning models on AWS EC2 GPU instances with automatic model saving to S3 and cost-effective spot instance management.

## 🚀 Features

- **Cost-Effective Training**: Uses AWS EC2 spot instances for up to 90% cost savings
- **Automatic Model Backup**: Saves checkpoints to S3 every 50,000 steps by default
- **GPU Optimization**: Supports NVIDIA GPU instances with CUDA acceleration
- **Auto-Resume**: Automatically resumes training from the latest S3 checkpoint
- **Auto-Shutdown**: Automatically terminates instances when training completes
- **Multi-Agent Support**: Supports both single-agent SAC and multi-agent SAC training
- **Monitoring**: Integrated with Weights & Biases and TensorBoard
- **Security**: Encrypted S3 storage and secure IAM roles

## 📋 Prerequisites

### Local Requirements
- AWS CLI installed and configured
- Terraform >= 1.0
- An AWS account with appropriate permissions
- An EC2 key pair for SSH access

### AWS Permissions Required
Your AWS user/role needs permissions for:
- EC2 (launch instances, security groups)
- S3 (create buckets, read/write objects)
- IAM (create roles and policies)
- Spot instances

## 🛠️ Quick Start

### 1. Configure Terraform Variables

Copy the example configuration:
```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your settings:
```hcl
# AWS Configuration
aws_region        = "us-east-1"
availability_zone = "us-east-1a"

# EC2 Configuration
instance_type     = "g4dn.xlarge"  # Most economical GPU instance
key_pair_name     = "your-key-pair-name"

# Project Configuration
project_name      = "f1tenth-rl-training"
s3_bucket_name    = "your-unique-bucket-name"

# Training Configuration
training_type     = "sac"           # or "sac_multiagent"
total_timesteps   = 1000000
save_frequency    = 50000
```

### 2. Deploy Infrastructure

Initialize and deploy with Terraform:
```bash
# Initialize Terraform
terraform init

# Review the deployment plan
terraform plan

# Deploy the infrastructure
terraform apply
```

### 3. Monitor Training

After deployment, you can:

**SSH into the instance:**
```bash
# Use the SSH command from Terraform output
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

**Monitor logs:**
```bash
# View training logs
sudo journalctl -u f1tenth-training -f

# Check training status
sudo systemctl status f1tenth-training
```

**Access TensorBoard:**
```bash
# Open in browser (replace with actual IP)
http://<instance-ip>:6006
```

### 4. Clean Up

When training is complete, destroy the infrastructure:
```bash
terraform destroy
```

## 📁 Project Structure

```
terraform/
├── main.tf                    # Main Terraform configuration
├── user_data.sh              # EC2 instance setup script
├── terraform.tfvars.example  # Configuration template
└── terraform.tfvars          # Your configuration (create this)

train_cloud/
├── sac_cloud_training.py           # Single-agent SAC training
└── sac_multiagent_cloud_training.py # Multi-agent SAC training

examples/
├── sac_example.py             # Local single-agent example
└── sac_example_multi_agent.py # Local multi-agent example
```

## 🏗️ Infrastructure Components

### Script Management Architecture

The infrastructure uses a modern approach for managing training scripts:

- **S3 Script Storage**: Training scripts are uploaded to S3 during deployment
- **Dynamic Downloads**: EC2 instances download the latest scripts from S3 
- **Version Control**: Scripts are versioned using S3 object versioning
- **Dependency Management**: EC2 instances depend on script uploads (using `depends_on`)
- **Maintainability**: No embedded code in `user_data.sh` - scripts remain in source control

This approach provides several advantages:
- ✅ Scripts stay in sync with the repository
- ✅ Easy to update without modifying Terraform files
- ✅ Better separation of concerns
- ✅ Reduced `user_data.sh` complexity
- ✅ Version tracking for all training scripts

### AWS Resources Created

1. **EC2 Spot Instance**
   - Ubuntu 22.04 LTS
   - GPU-enabled instance (g4dn.xlarge by default)
   - 50GB encrypted EBS volume
   - Auto-configured with NVIDIA drivers and CUDA

2. **S3 Bucket**
   - Versioning enabled
   - Server-side encryption (AES256)
   - Stores model checkpoints and training artifacts

3. **IAM Role & Policies**
   - EC2 instance role with S3 access
   - Permissions for instance self-termination
   - Minimal required permissions following best practices

4. **Security Group**
   - SSH access (port 22)
   - TensorBoard access (port 6006)
   - All outbound traffic allowed

### Instance Setup (Automated)

The `user_data.sh` script automatically:
1. Updates the system and installs dependencies
2. Installs NVIDIA drivers and CUDA toolkit
3. Sets up Python 3.10 virtual environment
4. Installs F1Tenth Gym and all dependencies
5. Configures AWS CLI with instance credentials
6. Creates systemd service for automatic training start
7. Downloads and starts the appropriate training script

## 🎯 Training Scripts

### Single-Agent SAC (`sac_cloud_training.py`)

Features:
- Standard SAC algorithm implementation
- S3 checkpoint saving every 50,000 steps
- Automatic resume from latest checkpoint
- GPU acceleration when available
- Weights & Biases integration
- Auto-shutdown after completion

### Multi-Agent SAC (`sac_multiagent_cloud_training.py`)

Features:
- Multiple SAC agents training simultaneously
- Independent replay buffers per agent
- Shared environment experience
- Per-agent checkpoint management
- Coordinated S3 backup strategy
- Scalable to any number of agents

## 🔧 Configuration Options

### Instance Types

| Instance Type | GPU | vCPUs | Memory | GPU Memory | Spot Price (approx.) |
|---------------|-----|-------|--------|------------|---------------------|
| g4dn.xlarge   | T4  | 4     | 16GB   | 16GB       | $0.12/hour         |
| g4dn.2xlarge  | T4  | 8     | 32GB   | 16GB       | $0.23/hour         |
| g5.xlarge     | A10G| 4     | 16GB   | 24GB       | $0.27/hour         |
| g5.2xlarge    | A10G| 8     | 32GB   | 24GB       | $0.41/hour         |

### Training Parameters

Adjust these in `terraform.tfvars`:

```hcl
# Training duration
total_timesteps = 1000000      # Total training steps

# Checkpoint frequency  
save_frequency = 50000         # Save every N steps

# Algorithm selection
training_type = "sac"          # "sac" or "sac_multiagent"

# Multi-agent specific (set in training script)
NUM_AGENTS = 2                 # Number of agents for multi-agent training
```

## 📊 Monitoring and Logging

### Weights & Biases

Training metrics are automatically logged to W&B:
- Training loss and rewards
- Environment statistics
- GPU utilization
- Model performance metrics

### TensorBoard

Access TensorBoard at `http://<instance-ip>:6006` to view:
- Training curves
- Network architecture
- Hyperparameter tracking

### CloudWatch (Optional)

You can enable CloudWatch monitoring for:
- Instance metrics (CPU, GPU, memory)
- Custom training metrics
- Log aggregation

## 💰 Cost Optimization

### Spot Instance Savings
- Spot instances cost ~70-90% less than on-demand
- Auto-termination prevents unexpected charges
- Training resumes automatically from checkpoints

### Storage Optimization
- Old local models are automatically cleaned up
- S3 versioning allows point-in-time recovery
- Lifecycle policies can be added for long-term archival

### Training Efficiency
- Optimized for GPU utilization
- Efficient checkpoint strategy
- Minimal overhead for cloud operations

## 🔒 Security Best Practices

### IAM Security
- Minimal required permissions
- Instance-specific IAM roles
- No hardcoded credentials

### Network Security
- Security groups restrict access
- SSH key-based authentication
- Optional: VPC deployment for enhanced isolation

### Data Security
- S3 server-side encryption
- EBS volume encryption
- Secure credential handling

## 🐛 Troubleshooting

### Common Issues

**Training doesn't start:**
```bash
# Check service status
sudo systemctl status f1tenth-training

# View detailed logs
sudo journalctl -u f1tenth-training -f
```

**GPU not detected:**
```bash
# Check NVIDIA driver installation
nvidia-smi

# Verify CUDA installation
nvcc --version
```

**S3 permissions issues:**
```bash
# Test S3 access
aws s3 ls s3://your-bucket-name

# Check instance profile
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

**High spot instance interruption:**
- Try different availability zones
- Use multiple smaller instances
- Consider on-demand instances for critical training

### Recovery Procedures

**Resume interrupted training:**
Training automatically resumes from the latest S3 checkpoint. No manual intervention required.

**Manual checkpoint download:**
```bash
aws s3 cp s3://your-bucket/checkpoints/sac_latest_checkpoint.json .
aws s3 cp s3://your-bucket/models/sac/model_file.zip .
```

## 📈 Performance Tuning

### GPU Optimization
- Use mixed precision training for faster speeds
- Adjust batch sizes based on GPU memory
- Monitor GPU utilization with `nvidia-smi`

### Training Hyperparameters
- Learning rate: 3e-4 (default, works well)
- Buffer size: 1M (adjust based on available memory)
- Batch size: 256 (optimal for most GPUs)

### S3 Transfer Optimization
- Use multipart uploads for large models
- Consider S3 Transfer Acceleration for global access
- Monitor S3 request costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review AWS CloudWatch logs
3. Open a GitHub issue with detailed information
4. Contact the maintainers

## 🔗 Related Resources

- [F1Tenth Gym Documentation](https://f1tenth-gym.readthedocs.io/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [AWS EC2 Spot Instances Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

---

🏎️ **Happy Training!** 🏎️
