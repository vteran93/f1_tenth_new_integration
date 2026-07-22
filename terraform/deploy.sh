#!/bin/bash
# F1Tenth Cloud Training Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TERRAFORM_DIR="$SCRIPT_DIR"

print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "🏎️  F1Tenth RL Cloud Training Deployment"
    echo "============================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform >= 1.0"
        exit 1
    fi
    
    # Check if aws cli is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install AWS CLI"
        exit 1
    fi
    
    # Check if aws is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Run 'aws configure' first"
        exit 1
    fi
    
    # Check terraform version
    TERRAFORM_VERSION=$(terraform version -json | jq -r '.terraform_version')
    print_success "Terraform version: $TERRAFORM_VERSION"
    
    # Check AWS identity
    AWS_IDENTITY=$(aws sts get-caller-identity --query 'Arn' --output text)
    print_success "AWS Identity: $AWS_IDENTITY"
}

check_terraform_vars() {
    if [ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
        print_warning "terraform.tfvars not found. Creating from template..."
        cp "$TERRAFORM_DIR/terraform.tfvars.example" "$TERRAFORM_DIR/terraform.tfvars"
        print_info "Please edit terraform.tfvars with your configuration and run this script again"
        print_info "Required variables: key_pair_name, s3_bucket_name"
        exit 1
    fi
    
    # Check for required variables
    if ! grep -q "key_pair_name" "$TERRAFORM_DIR/terraform.tfvars" || grep -q "your-key-pair-name" "$TERRAFORM_DIR/terraform.tfvars"; then
        print_error "Please set key_pair_name in terraform.tfvars"
        exit 1
    fi
    
    if ! grep -q "s3_bucket_name" "$TERRAFORM_DIR/terraform.tfvars" || grep -q "your-unique-bucket-name" "$TERRAFORM_DIR/terraform.tfvars"; then
        print_error "Please set s3_bucket_name in terraform.tfvars"
        exit 1
    fi
    
    print_success "terraform.tfvars configuration looks good"
}

terraform_init() {
    print_info "Initializing Terraform..."
    cd "$TERRAFORM_DIR"
    terraform init
    print_success "Terraform initialized"
}

terraform_plan() {
    print_info "Creating Terraform plan..."
    cd "$TERRAFORM_DIR"
    terraform plan -out=tfplan
    print_success "Terraform plan created"
}

terraform_apply() {
    print_info "Applying Terraform configuration..."
    cd "$TERRAFORM_DIR"
    terraform apply tfplan
    print_success "Infrastructure deployed successfully!"
}

show_outputs() {
    print_info "Retrieving deployment information..."
    cd "$TERRAFORM_DIR"
    
    echo ""
    print_header
    echo "🎉 Deployment Complete!"
    echo ""
    
    INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "Not available")
    SSH_COMMAND=$(terraform output -raw ssh_command 2>/dev/null || echo "Not available")
    TENSORBOARD_URL=$(terraform output -raw tensorboard_url 2>/dev/null || echo "Not available")
    S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "Not available")
    
    echo -e "${GREEN}📊 Instance Information:${NC}"
    echo "  Public IP: $INSTANCE_IP"
    echo "  S3 Bucket: $S3_BUCKET"
    echo ""
    
    echo -e "${GREEN}🔗 Access Information:${NC}"
    echo "  SSH: $SSH_COMMAND"
    echo "  TensorBoard: $TENSORBOARD_URL"
    echo ""
    
    echo -e "${GREEN}📈 Monitoring:${NC}"
    echo "  Training Logs: sudo journalctl -u f1tenth-training -f"
    echo "  Service Status: sudo systemctl status f1tenth-training"
    echo "  GPU Status: nvidia-smi"
    echo ""
    
    echo -e "${YELLOW}⏱️  Training will start automatically in ~5-10 minutes after instance boot${NC}"
    echo -e "${YELLOW}🛑 Instance will auto-shutdown when training completes${NC}"
}

monitor_training() {
    print_info "Connecting to monitor training..."
    cd "$TERRAFORM_DIR"
    
    INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null)
    SSH_KEY=$(grep "key_pair_name" terraform.tfvars | cut -d'"' -f2)
    
    if [ "$INSTANCE_IP" = "Not available" ] || [ -z "$INSTANCE_IP" ]; then
        print_error "Could not get instance IP. Is the infrastructure deployed?"
        exit 1
    fi
    
    print_info "Connecting to $INSTANCE_IP..."
    print_info "You can monitor training with: sudo journalctl -u f1tenth-training -f"
    
    ssh -i ~/.ssh/${SSH_KEY}.pem ubuntu@${INSTANCE_IP}
}

destroy_infrastructure() {
    print_warning "This will destroy all infrastructure including:"
    print_warning "- EC2 instance"
    print_warning "- Security groups"
    print_warning "- IAM roles"
    print_warning "Note: S3 bucket and models will be preserved"
    echo ""
    
    read -p "Are you sure you want to destroy the infrastructure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        print_info "Destroying infrastructure..."
        cd "$TERRAFORM_DIR"
        terraform destroy -auto-approve
        print_success "Infrastructure destroyed"
    else
        print_info "Destruction cancelled"
    fi
}

show_help() {
    echo "F1Tenth Cloud Training Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    - Deploy the complete infrastructure"
    echo "  plan      - Show what will be deployed (dry run)"
    echo "  monitor   - SSH into the training instance"
    echo "  destroy   - Destroy all infrastructure"
    echo "  status    - Show current deployment status"
    echo "  help      - Show this help message"
    echo ""
    echo "Before first use:"
    echo "1. Configure AWS CLI: aws configure"
    echo "2. Edit terraform.tfvars with your settings"
    echo "3. Run: $0 deploy"
}

show_status() {
    cd "$TERRAFORM_DIR"
    
    if [ ! -f "terraform.tfstate" ]; then
        print_info "No infrastructure deployed"
        return
    fi
    
    print_info "Current infrastructure status:"
    terraform show -json | jq -r '.values.root_module.resources[] | select(.type == "aws_spot_instance_request") | .values | "Instance: \(.spot_instance_id // "pending") (\(.instance_type))"'
    
    INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "Not available")
    if [ "$INSTANCE_IP" != "Not available" ]; then
        print_success "Instance accessible at: $INSTANCE_IP"
        echo "  TensorBoard: http://$INSTANCE_IP:6006"
    fi
}

# Main script logic
case "${1:-}" in
    "deploy")
        print_header
        check_prerequisites
        check_terraform_vars
        terraform_init
        terraform_plan
        terraform_apply
        show_outputs
        ;;
    "plan")
        print_header
        check_prerequisites
        check_terraform_vars
        terraform_init
        terraform_plan
        print_info "Plan created. Run '$0 deploy' to apply changes."
        ;;
    "monitor")
        monitor_training
        ;;
    "destroy")
        destroy_infrastructure
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
