#!/bin/bash
set -euo pipefail

# ── Tencent Cloud (腾讯云) CVM Deployment ──
# Prerequisites: Tencent Cloud CLI (tccli) configured

REGION="${TC_REGION:-ap-shanghai}"
ZONE="${TC_ZONE:-ap-shanghai-2}"
INSTANCE_TYPE="${TC_INSTANCE_TYPE:-S5.MEDIUM4}"
SECURITY_GROUP="${TC_SG:-fl-backend-sg}"

echo "=== Deploying Unified FL Backend to Tencent Cloud ==="

# 1. Create security group
tccli cvm CreateSecurityGroup \
    --GroupName "$SECURITY_GROUP" \
    --Description "Unified FL Backend" \
    --Region "$REGION" 2>/dev/null || true

SG_ID=$(tccli cvm DescribeSecurityGroups \
    --Filters '[{"Name":"security-group-name","Values":["'$SECURITY_GROUP'"]}]' \
    --Region "$REGION" \
    --query 'SecurityGroupSet[0].SecurityGroupId' \
    --output text 2>/dev/null)

if [ -n "$SG_ID" ]; then
    # Open ports
    for PORT in 22 8080 50051; do
        tccli vpc CreateSecurityGroupPolicies \
            --SecurityGroupId "$SG_ID" \
            --SecurityGroupPolicySet.Ingress.0.Protocol TCP \
            --SecurityGroupPolicySet.Ingress.0.Port "$PORT" \
            --SecurityGroupPolicySet.Ingress.0.CidrBlock "0.0.0.0/0" \
            --SecurityGroupPolicySet.Ingress.0.Action ALLOW \
            --Region "$REGION" 2>/dev/null || true
    done
    echo "  Security Group: $SG_ID"
fi

# 2. Find latest Ubuntu image
IMAGE_ID=$(tccli cvm DescribeImages \
    --Filters '[{"Name":"image-name","Values":["Ubuntu 24.04 LTS 64位"]},{"Name":"image-type","Values":["PUBLIC"]}]' \
    --Region "$REGION" \
    --query 'ImageSet | sort_by(@, &CreationTime) | [-1].ImageId' \
    --output text)

echo "  Image: $IMAGE_ID"
echo "  Instance: $INSTANCE_TYPE"
echo "  Region: $REGION"

# 3. Create instance
INSTANCE_ID=$(tccli cvm RunInstances \
    --ImageId "$IMAGE_ID" \
    --InstanceType "$INSTANCE_TYPE" \
    --Placement.Zone "$ZONE" \
    --SecurityGroupIds '["'$SG_ID'"]' \
    --LoginSettings.Password "$(openssl rand -base64 16)" \
    --InstanceChargeType POSTPAID_BY_HOUR \
    --Region "$REGION" \
    --query 'InstanceIdSet[0]' \
    --output text)

echo "  Instance: $INSTANCE_ID"

# 4. Wait and get IP
echo "  Waiting for instance..."
sleep 30

PUBLIC_IP=$(tccli cvm DescribeInstances \
    --InstanceIds '["'$INSTANCE_ID'"]' \
    --Region "$REGION" \
    --query 'InstanceSet[0].PublicIpAddresses[0]' \
    --output text)

echo ""
echo "=== Deployment Complete ==="
echo "  Instance: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  REST API: http://$PUBLIC_IP:8080/health"
echo "  gRPC: $PUBLIC_IP:50051"
echo ""
echo "Next steps:"
echo "  1. SSH: ssh ubuntu@$PUBLIC_IP"
echo "  2. Copy project: scp -r unified-fl-backend/ ubuntu@$PUBLIC_IP:~/"
echo "  3. Start: cd unified-fl-backend && docker compose up -d"
echo "  4. Verify: curl http://$PUBLIC_IP:8080/health"
