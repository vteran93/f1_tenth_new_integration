#!/bin/bash

# F1Tenth RL Training Infrastructure Quick Setup
# This script helps users get started quickly with the infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏎️  F1Tenth RL Training Infrastructure Quick Setup${NC}"
echo "================================================="

# Check if terraform.tfvars exists
if [[ -f "terraform.tfvars" ]]; then
    echo -e "${YELLOW}⚠️  terraform.tfvars already exists. Skipping creation.${NC}"
else
    echo -e "${BLUE}📋 Creating terraform.tfvars from template...${NC}"
    cp terraform.tfvars.example terraform.tfvars
    echo -e "${GREEN}✅ Created terraform.tfvars${NC}"
    
    echo -e "\n${YELLOW}📝 Please edit terraform.tfvars with your configuration:${NC}"
    echo -e "   1. Set your AWS region and availability zone"
    echo -e "   2. Replace 'your-key-pair-name' with your EC2 key pair name"
    echo -e "   3. Replace 'your-unique-bucket-name-f1tenth-models' with a globally unique S3 bucket name"
    echo -e "   4. Adjust instance type and training parameters as needed"
    echo ""
    
    # Open editor if available
    if command -v code &> /dev/null; then
        echo -e "${BLUE}💡 Opening terraform.tfvars in VS Code...${NC}"
        code terraform.tfvars
    elif command -v nano &> /dev/null; then
        echo -e "${BLUE}💡 Opening terraform.tfvars in nano...${NC}"
        nano terraform.tfvars
    else
        echo -e "${YELLOW}💡 Please edit terraform.tfvars with your preferred editor${NC}"
    fi
fi

echo -e "\n${BLUE}🔧 Next Steps:${NC}"
echo -e "1. Ensure AWS CLI is configured: ${YELLOW}aws configure${NC}"
echo -e "2. Create EC2 key pair if needed: ${YELLOW}aws ec2 create-key-pair --key-name YOUR_KEY_NAME${NC}"
echo -e "3. Validate configuration: ${YELLOW}./validate.sh${NC}"
echo -e "4. Deploy infrastructure: ${YELLOW}./deploy.sh${NC}"
echo -e "5. Monitor training: ${YELLOW}./monitor.sh${NC}"

echo -e "\n${GREEN}🎯 Quick Commands:${NC}"
echo -e "   ${YELLOW}./validate.sh${NC}  - Validate prerequisites and configuration"
echo -e "   ${YELLOW}./deploy.sh${NC}    - Deploy the infrastructure"
echo -e "   ${YELLOW}./monitor.sh${NC}   - Monitor training progress"
echo -e "   ${YELLOW}terraform destroy${NC} - Clean up resources when done"

echo -e "\n${BLUE}📚 Documentation:${NC}"
echo -e "   📖 Full guide: README.md"
echo -e "   🏗️  Architecture: See main.tf for detailed infrastructure"
echo -e "   🚂 Training scripts: ../train_cloud/"

echo -e "\n${YELLOW}💡 Pro Tips:${NC}"
echo -e "   🔹 Use spot instances to save 50-90% on costs"
echo -e "   🔹 Monitor costs with AWS Cost Explorer"
echo -e "   🔹 Set billing alerts in AWS console"
echo -e "   🔹 Use screen/tmux for long-running operations"

echo -e "\n${GREEN}✨ Setup complete! Edit terraform.tfvars and run ./validate.sh to continue.${NC}"
