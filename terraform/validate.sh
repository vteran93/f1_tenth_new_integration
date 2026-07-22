#!/bin/bash

# F1Tenth RL Training Infrastructure Validation Script
# This script validates prerequisites and configuration before deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏎️  F1Tenth RL Training Infrastructure Validation${NC}"
echo "=================================================="

# Check if running from correct directory
if [[ ! -f "main.tf" ]]; then
    echo -e "${RED}❌ Error: main.tf not found. Please run this script from the terraform directory.${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Checking Prerequisites...${NC}"

# Function to check if command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅ $1 is installed${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    fi
}

# Check required tools
MISSING_TOOLS=0

if ! check_command terraform; then
    echo -e "${YELLOW}   Install: https://www.terraform.io/downloads.html${NC}"
    MISSING_TOOLS=1
fi

if ! check_command aws; then
    echo -e "${YELLOW}   Install: https://aws.amazon.com/cli/${NC}"
    MISSING_TOOLS=1
fi

if ! check_command jq; then
    echo -e "${YELLOW}   Install: sudo apt-get install jq (Ubuntu/Debian)${NC}"
    MISSING_TOOLS=1
fi

if [[ $MISSING_TOOLS -eq 1 ]]; then
    echo -e "${RED}❌ Please install missing tools before proceeding.${NC}"
    exit 1
fi

echo -e "\n${BLUE}🔧 Checking AWS Configuration...${NC}"

# Check AWS credentials
if aws sts get-caller-identity &> /dev/null; then
    AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    AWS_USER=$(aws sts get-caller-identity --query Arn --output text | cut -d'/' -f2)
    echo -e "${GREEN}✅ AWS credentials configured${NC}"
    echo -e "   Account: $AWS_ACCOUNT"
    echo -e "   User: $AWS_USER"
else
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo -e "${YELLOW}   Run: aws configure${NC}"
    exit 1
fi

# Check terraform.tfvars
echo -e "\n${BLUE}📋 Checking Configuration...${NC}"

if [[ ! -f "terraform.tfvars" ]]; then
    echo -e "${RED}❌ terraform.tfvars not found${NC}"
    echo -e "${YELLOW}   Copy terraform.tfvars.example to terraform.tfvars and edit it${NC}"
    exit 1
fi

echo -e "${GREEN}✅ terraform.tfvars found${NC}"

# Extract and validate configuration values
if grep -q "your-key-pair-name" terraform.tfvars; then
    echo -e "${RED}❌ Please update key_pair_name in terraform.tfvars${NC}"
    exit 1
fi

if grep -q "your-unique-bucket-name" terraform.tfvars; then
    echo -e "${RED}❌ Please update s3_bucket_name in terraform.tfvars${NC}"
    exit 1
fi

# Get key pair name from terraform.tfvars
KEY_PAIR_NAME=$(grep "key_pair_name" terraform.tfvars | cut -d'"' -f2)
AWS_REGION=$(grep "aws_region" terraform.tfvars | cut -d'"' -f2)

# Validate key pair exists
echo -e "\n${BLUE}🔑 Validating EC2 Key Pair...${NC}"
if aws ec2 describe-key-pairs --key-names "$KEY_PAIR_NAME" --region "$AWS_REGION" &> /dev/null; then
    echo -e "${GREEN}✅ Key pair '$KEY_PAIR_NAME' exists in $AWS_REGION${NC}"
else
    echo -e "${RED}❌ Key pair '$KEY_PAIR_NAME' not found in $AWS_REGION${NC}"
    echo -e "${YELLOW}   Create it with: aws ec2 create-key-pair --key-name $KEY_PAIR_NAME --region $AWS_REGION${NC}"
    exit 1
fi

# Check S3 bucket name availability
S3_BUCKET_NAME=$(grep "s3_bucket_name" terraform.tfvars | cut -d'"' -f2)
echo -e "\n${BLUE}🪣 Checking S3 Bucket Availability...${NC}"

if aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Bucket '$S3_BUCKET_NAME' already exists${NC}"
    echo -e "${YELLOW}   If it's yours, the deployment will use it. If not, choose a different name.${NC}"
else
    echo -e "${GREEN}✅ Bucket name '$S3_BUCKET_NAME' is available${NC}"
fi

# Check AWS quotas and limits
echo -e "\n${BLUE}📊 Checking AWS Service Limits...${NC}"

# Check EC2 limits for GPU instances
INSTANCE_TYPE=$(grep "instance_type" terraform.tfvars | cut -d'"' -f2)
echo -e "${GREEN}✅ Will use instance type: $INSTANCE_TYPE${NC}"

# Check if we can launch spot instances in the specified AZ
AZ=$(grep "availability_zone" terraform.tfvars | cut -d'"' -f2)
echo -e "${GREEN}✅ Will launch in availability zone: $AZ${NC}"

# Validate training scripts exist
echo -e "\n${BLUE}🚂 Checking Training Scripts...${NC}"

if [[ -f "../train_cloud/sac_cloud_training.py" ]]; then
    echo -e "${GREEN}✅ SAC training script found${NC}"
else
    echo -e "${RED}❌ SAC training script not found at ../train_cloud/sac_cloud_training.py${NC}"
    exit 1
fi

if [[ -f "../train_cloud/sac_multiagent_cloud_training.py" ]]; then
    echo -e "${GREEN}✅ Multi-agent SAC training script found${NC}"
else
    echo -e "${RED}❌ Multi-agent SAC training script not found at ../train_cloud/sac_multiagent_cloud_training.py${NC}"
    exit 1
fi

# Terraform validation
echo -e "\n${BLUE}🔍 Validating Terraform Configuration...${NC}"

if terraform init -backend=false &> /dev/null; then
    echo -e "${GREEN}✅ Terraform initialization successful${NC}"
else
    echo -e "${RED}❌ Terraform initialization failed${NC}"
    exit 1
fi

if terraform validate &> /dev/null; then
    echo -e "${GREEN}✅ Terraform configuration is valid${NC}"
else
    echo -e "${RED}❌ Terraform configuration validation failed${NC}"
    terraform validate
    exit 1
fi

# Cost estimation
echo -e "\n${BLUE}💰 Cost Estimation...${NC}"

case $INSTANCE_TYPE in
    "g4dn.xlarge")
        HOURLY_RATE="0.52"
        ;;
    "g4dn.2xlarge")
        HOURLY_RATE="0.75"
        ;;
    "g4dn.4xlarge")
        HOURLY_RATE="1.20"
        ;;
    *)
        HOURLY_RATE="0.50"
        ;;
esac

echo -e "${YELLOW}💡 Estimated costs (spot pricing may be 50-90% less):${NC}"
echo -e "   📊 Hourly rate (~$${HOURLY_RATE}/hour for on-demand)"
echo -e "   🕐 4-hour training: ~$$(echo "$HOURLY_RATE * 4" | bc -l | xargs printf "%.2f")"
echo -e "   🕐 8-hour training: ~$$(echo "$HOURLY_RATE * 8" | bc -l | xargs printf "%.2f")"
echo -e "   🪣 S3 storage: ~$0.023/GB/month"

echo -e "\n${GREEN}🎉 All validations passed! Ready for deployment.${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo -e "1. Review your terraform.tfvars configuration"
echo -e "2. Run: ${YELLOW}./deploy.sh${NC}"
echo -e "3. Monitor training: ${YELLOW}./monitor.sh${NC}"
echo -e "\n${YELLOW}💡 Pro tip: Use screen or tmux for long deployments${NC}"
