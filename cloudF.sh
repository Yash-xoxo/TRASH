#!/bin/bash

# Script to install Cloudflared on Fedora/RHEL/CentOS systems
# Run with sudo: sudo bash install_cloudflared.sh

set -e  # Exit on any error
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo \
https://pkg.cloudflare.com/cloudflared-ascii.repo

sudo dnf install -y cloudflared


cloudflared --version
