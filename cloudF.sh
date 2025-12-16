#!/bin/bash

# Script to install Cloudflared on Fedora/RHEL/CentOS systems
# Run with sudo: sudo bash install_cloudflared.sh

set -e  # Exit on any error

echo "=== Installing Cloudflared on Fedora ==="

# 1) Ensure required tools are installed
echo "Step 1: Installing required tools..."
dnf install -y curl gnupg

# 2) Add Cloudflare's RPM repository
echo "Step 2: Adding Cloudflare RPM repository..."
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.repo | tee /etc/yum.repos.d/cloudflare-main.repo


# 3)--------
# This requires dnf config-manager
# Add cloudflared.repo to config-manager
# Stable
sudo dnf config-manager --add-repo https://pkg.cloudflare.com/cloudflared.repo
# Nightly
sudo dnf config-manager --add-repo https://next.pkg.cloudflare.com/cloudflared.repo

# install cloudflared
sudo dnf install cloudflared

# 4) Install cloudflared
echo "Step 4: Installing cloudflared..."
dnf install -y cloudflared

# 5) Verify installation
echo "Step 5: Verifying installation..."
cloudflared --version

echo ""
echo "=== Cloudflared installed successfully! ==="
echo "Run 'cloudflared --help' for usage information"
