#!/bin/bash

# Script to install Cloudflared on Fedora/RHEL/CentOS systems
# Run with sudo: sudo bash install_cloudflared.sh

set -e  # Exit on any error
sudo dnf install -y curl
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/
cloudflared --version


