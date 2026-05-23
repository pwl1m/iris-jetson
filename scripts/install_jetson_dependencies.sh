#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  docker.io \
  docker-compose-plugin \
  python3-pip \
  python3-venv \
  v4l-utils \
  ffmpeg \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav \
  jq \
  htop \
  tmux \
  rsync \
  openssh-server

sudo usermod -aG docker "$USER"

echo "Reinicie a sessao ou faca reboot para o grupo docker valer."
echo "Confirme JetPack/L4T com: cat /etc/nv_tegra_release"

