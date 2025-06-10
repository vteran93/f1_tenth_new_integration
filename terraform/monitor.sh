#!/bin/bash
# Training Progress Monitor Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "📊 F1Tenth Training Progress Monitor"
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

check_aws_config() {
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured"
        exit 1
    fi
}

get_s3_bucket() {
    if [ -f "terraform.tfvars" ]; then
        BUCKET=$(grep "s3_bucket_name" terraform.tfvars | cut -d'"' -f2)
        if [ "$BUCKET" != "your-unique-bucket-name" ] && [ -n "$BUCKET" ]; then
            echo "$BUCKET"
            return
        fi
    fi
    
    if [ -f "terraform.tfstate" ]; then
        BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null)
        if [ -n "$BUCKET" ] && [ "$BUCKET" != "Not available" ]; then
            echo "$BUCKET"
            return
        fi
    fi
    
    echo ""
}

show_latest_checkpoints() {
    local bucket=$1
    
    print_info "Checking for latest checkpoints..."
    
    # Check for SAC single-agent checkpoint
    if aws s3 ls "s3://$bucket/checkpoints/sac_latest_checkpoint.json" &>/dev/null; then
        print_success "SAC Single-Agent Training Found"
        
        CHECKPOINT_INFO=$(aws s3 cp "s3://$bucket/checkpoints/sac_latest_checkpoint.json" - 2>/dev/null)
        if [ $? -eq 0 ]; then
            TIMESTEPS=$(echo "$CHECKPOINT_INFO" | jq -r '.timesteps_completed // "unknown"')
            TIMESTAMP=$(echo "$CHECKPOINT_INFO" | jq -r '.timestamp // "unknown"')
            echo "  📈 Progress: $TIMESTEPS timesteps"
            echo "  🕐 Last Update: $TIMESTAMP"
        fi
        echo ""
    fi
    
    # Check for SAC multi-agent checkpoints
    MULTIAGENT_CHECKPOINTS=$(aws s3 ls "s3://$bucket/checkpoints/" | grep "sac_multiagent_agent_" | wc -l)
    if [ "$MULTIAGENT_CHECKPOINTS" -gt 0 ]; then
        print_success "SAC Multi-Agent Training Found ($MULTIAGENT_CHECKPOINTS agents)"
        
        for checkpoint in $(aws s3 ls "s3://$bucket/checkpoints/" | grep "sac_multiagent_agent_" | awk '{print $4}'); do
            AGENT_ID=$(echo "$checkpoint" | grep -o "agent_[0-9]*" | cut -d'_' -f2)
            
            CHECKPOINT_INFO=$(aws s3 cp "s3://$bucket/checkpoints/$checkpoint" - 2>/dev/null)
            if [ $? -eq 0 ]; then
                TIMESTEPS=$(echo "$CHECKPOINT_INFO" | jq -r '.timesteps_completed // "unknown"')
                TIMESTAMP=$(echo "$CHECKPOINT_INFO" | jq -r '.timestamp // "unknown"')
                echo "  🤖 Agent $AGENT_ID: $TIMESTEPS timesteps (updated: $TIMESTAMP)"
            fi
        done
        echo ""
    fi
    
    if [ "$MULTIAGENT_CHECKPOINTS" -eq 0 ] && ! aws s3 ls "s3://$bucket/checkpoints/sac_latest_checkpoint.json" &>/dev/null; then
        print_warning "No checkpoints found. Training may not have started yet."
    fi
}

show_model_history() {
    local bucket=$1
    
    print_info "Recent model saves:"
    
    # SAC models
    SAC_MODELS=$(aws s3 ls "s3://$bucket/models/sac/" --recursive | tail -5)
    if [ -n "$SAC_MODELS" ]; then
        echo -e "${GREEN}📁 SAC Models (last 5):${NC}"
        echo "$SAC_MODELS" | while read -r line; do
            echo "  $line"
        done
        echo ""
    fi
    
    # Multi-agent models
    MULTIAGENT_MODELS=$(aws s3 ls "s3://$bucket/models/sac_multiagent/" --recursive | tail -10)
    if [ -n "$MULTIAGENT_MODELS" ]; then
        echo -e "${GREEN}📁 Multi-Agent Models (last 10):${NC}"
        echo "$MULTIAGENT_MODELS" | while read -r line; do
            echo "  $line"
        done
        echo ""
    fi
}

show_instance_status() {
    print_info "Checking EC2 instance status..."
    
    if [ -f "terraform.tfstate" ]; then
        INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null)
        INSTANCE_IP=$(terraform output -raw instance_public_ip 2>/dev/null)
        
        if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "Not available" ]; then
            INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "unknown")
            
            case "$INSTANCE_STATE" in
                "running")
                    print_success "Instance $INSTANCE_ID is running"
                    echo "  🌐 Public IP: $INSTANCE_IP"
                    echo "  📊 TensorBoard: http://$INSTANCE_IP:6006"
                    echo "  🔗 SSH: ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@$INSTANCE_IP"
                    ;;
                "stopped"|"stopping")
                    print_warning "Instance $INSTANCE_ID is stopped/stopping"
                    echo "  Training may have completed or been interrupted"
                    ;;
                "terminated"|"terminating")
                    print_info "Instance $INSTANCE_ID is terminated/terminating"
                    echo "  Training completed and instance auto-shutdown"
                    ;;
                *)
                    print_warning "Instance $INSTANCE_ID status: $INSTANCE_STATE"
                    ;;
            esac
        else
            print_warning "No active instance found"
        fi
    else
        print_warning "No Terraform state found. Infrastructure not deployed?"
    fi
    echo ""
}

show_costs_estimate() {
    local bucket=$1
    
    print_info "Estimated costs (last 24 hours):"
    
    # Note: This is a simplified cost estimation
    # For accurate costs, use AWS Cost Explorer or CloudWatch metrics
    
    if [ -f "terraform.tfvars" ]; then
        INSTANCE_TYPE=$(grep "instance_type" terraform.tfvars | cut -d'"' -f2 || echo "g4dn.xlarge")
        
        case "$INSTANCE_TYPE" in
            "g4dn.xlarge")
                SPOT_PRICE="~\$0.12/hour"
                ;;
            "g4dn.2xlarge")
                SPOT_PRICE="~\$0.23/hour"
                ;;
            "g5.xlarge")
                SPOT_PRICE="~\$0.27/hour"
                ;;
            *)
                SPOT_PRICE="varies"
                ;;
        esac
        
        echo "  💰 Instance ($INSTANCE_TYPE): $SPOT_PRICE"
    fi
    
    # S3 storage estimation
    S3_SIZE=$(aws s3 ls "s3://$bucket" --recursive --human-readable --summarize 2>/dev/null | grep "Total Size" | awk '{print $3 " " $4}' || echo "unknown")
    echo "  💾 S3 Storage: $S3_SIZE (~\$0.023/GB/month)"
    
    echo "  📝 Note: Use AWS Cost Explorer for accurate billing"
    echo ""
}

download_models() {
    local bucket=$1
    
    print_info "Available models to download:"
    
    # List recent models
    echo "Recent SAC models:"
    aws s3 ls "s3://$bucket/models/sac/" | tail -3
    
    echo ""
    echo "Recent Multi-Agent models:"
    aws s3 ls "s3://$bucket/models/sac_multiagent/" --recursive | tail -5
    
    echo ""
    read -p "Enter the model filename to download (or 'latest' for most recent): " model_choice
    
    if [ "$model_choice" = "latest" ]; then
        # Download latest SAC model
        LATEST_MODEL=$(aws s3 ls "s3://$bucket/models/sac/" | tail -1 | awk '{print $4}')
        if [ -n "$LATEST_MODEL" ]; then
            print_info "Downloading latest model: $LATEST_MODEL"
            aws s3 cp "s3://$bucket/models/sac/$LATEST_MODEL" "./models/"
            print_success "Downloaded to ./models/$LATEST_MODEL"
        else
            print_warning "No models found"
        fi
    elif [ -n "$model_choice" ]; then
        # Try to find and download specific model
        if aws s3 ls "s3://$bucket/models/sac/$model_choice" &>/dev/null; then
            aws s3 cp "s3://$bucket/models/sac/$model_choice" "./models/"
            print_success "Downloaded $model_choice"
        elif aws s3 ls "s3://$bucket/models/sac_multiagent/" --recursive | grep "$model_choice" &>/dev/null; then
            # Find full path for multi-agent model
            FULL_PATH=$(aws s3 ls "s3://$bucket/models/sac_multiagent/" --recursive | grep "$model_choice" | awk '{print $4}')
            aws s3 cp "s3://$bucket/models/sac_multiagent/$FULL_PATH" "./models/"
            print_success "Downloaded $model_choice"
        else
            print_error "Model not found: $model_choice"
        fi
    fi
}

watch_training() {
    local bucket=$1
    
    print_info "Starting training monitor (press Ctrl+C to stop)..."
    
    while true; do
        clear
        print_header
        show_instance_status
        show_latest_checkpoints "$bucket"
        echo "🔄 Refreshing in 30 seconds..."
        sleep 30
    done
}

show_help() {
    echo "F1Tenth Training Progress Monitor"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status    - Show current training status (default)"
    echo "  watch     - Continuously monitor training progress"
    echo "  models    - List and download available models"
    echo "  costs     - Show estimated costs"
    echo "  help      - Show this help message"
    echo ""
}

# Main script logic
check_aws_config

BUCKET=$(get_s3_bucket)
if [ -z "$BUCKET" ]; then
    print_error "Could not determine S3 bucket name"
    print_info "Make sure terraform.tfvars is configured or infrastructure is deployed"
    exit 1
fi

print_info "Using S3 bucket: $BUCKET"
echo ""

case "${1:-status}" in
    "status")
        print_header
        show_instance_status
        show_latest_checkpoints "$BUCKET"
        show_model_history "$BUCKET"
        ;;
    "watch")
        watch_training "$BUCKET"
        ;;
    "models")
        mkdir -p models
        download_models "$BUCKET"
        ;;
    "costs")
        show_costs_estimate "$BUCKET"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
