#!/bin/bash

# Script to install Cloudflared on Fedora/RHEL/CentOS systems
# Run with sudo: sudo bash install_cloudflared.sh

set -e  # Exit on any error
sudo dnf install -y curl
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/
cloudflared --version


sudo cloudflared service install eyJhIjoiODI1ZTMyMWMxMjg3MDNjYWNlNzg5ODM4MmJlNjhiMTYiLCJ0IjoiZGRlMzJkZDQtMjE2NS00Y2JhLWJkNDktZjYyMDM1MDQ4ZDM4IiwicyI6IllUTXpaamc1TXpndE1ESmxOaTAwWVRsakxUbGtZalF0TUdRNE1qVmpPV1ZoWVRneCJ9
