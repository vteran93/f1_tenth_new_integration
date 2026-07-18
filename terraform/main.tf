terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "Availability zone for resources"
  type        = string
  default     = "us-east-1a"
}

variable "instance_type" {
  description = "EC2 instance type with GPU support"
  type        = string
  default     = "g4dn.xlarge" # Most economical GPU instance
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "f1tenth-rl-training"
}

variable "key_pair_name" {
  description = "Name of existing EC2 key pair"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket name for model storage"
  type        = string
}

variable "training_type" {
  description = "Training type: sac or sac_multiagent"
  type        = string
  default     = "sac"
  validation {
    condition     = contains(["sac", "sac_multiagent"], var.training_type)
    error_message = "Training type must be either 'sac' or 'sac_multiagent'."
  }
}

variable "total_timesteps" {
  description = "Total training timesteps"
  type        = number
  default     = 1000000
}

variable "save_frequency" {
  description = "Save model every N steps"
  type        = number
  default     = 50000
}

# Data sources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# S3 Bucket for model storage
resource "aws_s3_bucket" "model_storage" {
  bucket = var.s3_bucket_name

  tags = {
    Name    = "${var.project_name}-models"
    Project = var.project_name
  }
}

# Upload training scripts to S3
resource "aws_s3_object" "sac_training_script" {
  bucket = aws_s3_bucket.model_storage.id
  key    = "scripts/sac_cloud_training.py"
  source = "${path.module}/../train_cloud/sac_cloud_training.py"
  etag   = filemd5("${path.module}/../train_cloud/sac_cloud_training.py")

  tags = {
    Name    = "SAC Training Script"
    Project = var.project_name
  }
}

resource "aws_s3_object" "sac_multiagent_training_script" {
  bucket = aws_s3_bucket.model_storage.id
  key    = "scripts/sac_multiagent_cloud_training.py"
  source = "${path.module}/../train_cloud/sac_multiagent_cloud_training.py"
  etag   = filemd5("${path.module}/../train_cloud/sac_multiagent_cloud_training.py")

  tags = {
    Name    = "SAC Multi-Agent Training Script"
    Project = var.project_name
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "model_storage_versioning" {
  bucket = aws_s3_bucket.model_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "model_storage_encryption" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "model_storage_lifecycle" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    id     = "model_cleanup"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    expiration {
      days = 90 # Delete old models after 90 days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# IAM Role for EC2 to access S3
resource "aws_iam_role" "ec2_s3_role" {
  name = "${var.project_name}-ec2-s3-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-ec2-s3-role"
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "ec2_s3_policy" {
  name = "${var.project_name}-ec2-s3-policy"
  role = aws_iam_role.ec2_s3_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_storage.arn,
          "${aws_s3_bucket.model_storage.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:StopInstances",
          "ec2:TerminateInstances"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_s3_role.name

  tags = {
    Name    = "${var.project_name}-ec2-profile"
    Project = var.project_name
  }
}

# Security Group
resource "aws_security_group" "training_sg" {
  name        = "${var.project_name}-sg"
  description = "Security group for F1Tenth RL training instance"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# Spot Instance Request
resource "aws_spot_instance_request" "training_instance" {
  ami                            = data.aws_ami.ubuntu.id
  instance_type                  = var.instance_type
  key_name                       = var.key_pair_name
  security_groups                = [aws_security_group.training_sg.name]
  wait_for_fulfillment           = true
  spot_type                      = "one-time"
  instance_interruption_behavior = "terminate"

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket_name  = var.s3_bucket_name
    training_type   = var.training_type
    total_timesteps = var.total_timesteps
    save_frequency  = var.save_frequency
    aws_region      = var.aws_region
  }))

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    encrypted   = true
  }

  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  # Ensure training scripts are uploaded to S3 before launching instance
  depends_on = [
    aws_s3_object.sac_training_script,
    aws_s3_object.sac_multiagent_training_script
  ]

  tags = {
    Name    = "${var.project_name}-training-spot"
    Project = var.project_name
    Type    = "training"
  }
}

# Outputs
output "instance_id" {
  description = "ID of the spot instance"
  value       = aws_spot_instance_request.training_instance.spot_instance_id
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = aws_spot_instance_request.training_instance.public_ip
}

output "s3_bucket_name" {
  description = "Name of S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_spot_instance_request.training_instance.public_ip}"
}

output "tensorboard_url" {
  description = "TensorBoard URL"
  value       = "http://${aws_spot_instance_request.training_instance.public_ip}:6006"
}

output "s3_bucket_url" {
  description = "S3 bucket URL for model storage"
  value       = "s3://${aws_s3_bucket.model_storage.id}"
}

output "aws_console_ec2_url" {
  description = "AWS Console URL for EC2 instance"
  value       = "https://${var.aws_region}.console.aws.amazon.com/ec2/v2/home?region=${var.aws_region}#Instances:search=${aws_spot_instance_request.training_instance.spot_instance_id}"
}

output "aws_console_s3_url" {
  description = "AWS Console URL for S3 bucket"
  value       = "https://s3.console.aws.amazon.com/s3/buckets/${aws_s3_bucket.model_storage.id}"
}

output "estimated_hourly_cost" {
  description = "Estimated hourly cost (on-demand pricing)"
  value       = var.instance_type == "g4dn.xlarge" ? "$0.526/hour" : var.instance_type == "g4dn.2xlarge" ? "$0.752/hour" : "Check AWS pricing"
}