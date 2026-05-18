#!/bin/bash
set -euo pipefail

# ── AWS EC2 Deployment ──
# Prerequisites: AWS CLI configured, SSH key pair

REGION="${AWS_REGION:-ap-southeast-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium"
KEY_NAME="${KEY_NAME:-fl-backend-key}"
SECURITY_GROUP="${SECURITY_GROUP:-fl-backend-sg}"

echo "=== Deploying Unified FL Backend to AWS ==="

# 1. Create security group
aws ec2 create-security-group \
    --group-name "$SECURITY_GROUP" \
    --description "Unified FL Backend" \
    --region "$REGION" 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-name "$SECURITY_GROUP" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-name "$SECURITY_GROUP" \
    --protocol tcp --port 8080 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-name "$SECURITY_GROUP" \
    --protocol tcp --port 50051 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

# 2. Find latest Ubuntu AMI
AMI=$(aws ssm get-parameters \
    --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
    --region "$REGION" \
    --query 'Parameters[0].Value' \
    --output text)

echo "  AMI: $AMI"
echo "  Instance: $INSTANCE_TYPE"
echo "  Region: $REGION"

# 3. Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --user-data "#!/bin/bash
apt-get update && apt-get install -y docker.io docker-compose-plugin curl
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu
" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=unified-fl-backend}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance: $INSTANCE_ID"

# 4. Wait for running
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Deployment Complete ==="
echo "  Instance: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  REST API: http://$PUBLIC_IP:8080/health"
echo "  gRPC: $PUBLIC_IP:50051"
echo ""
echo "Next steps:"
echo "  1. SSH: ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo "  2. Copy project: scp -r unified-fl-backend/ ubuntu@$PUBLIC_IP:~/"
echo "  3. Start: cd unified-fl-backend && docker compose up -d"
echo "  4. Verify: curl http://$PUBLIC_IP:8080/health"
