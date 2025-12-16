#!/bin/bash

# Script to install Cloudflared on Fedora/RHEL/CentOS systems
# Run with sudo: sudo bash install_cloudflared.sh

set -e  # Exit on any error
sudo dnf install -y dnf-plugins-core
sudo tee /etc/yum.repos.d/cloudflare-cloudflared.repo > /dev/null <<'EOF'
[cloudflare-cloudflared]
name=Cloudflare Cloudflared
baseurl=https://pkg.cloudflare.com/cloudflared/rpm
enabled=1
gpgcheck=1
gpgkey=https://pkg.cloudflare.com/cloudflare.gpg
EOF

sudo dnf install -y cloudflared


cloudflared --version
