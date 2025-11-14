#!/bin/bash
# Deployment script for DigitalOcean

set -e

echo "🚀 Deploying Llama-3.2-1B on DigitalOcean..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create cache directory
mkdir -p cache

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please create it from .env.example"
    echo "You need to add your HuggingFace token to access the Llama model"
    exit 1
fi

# Build and run
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting service..."
docker-compose up -d

echo "✅ Deployment complete!"
echo ""
echo "Service is running at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo ""
echo "Check logs with: docker-compose logs -f"
echo "Stop service with: docker-compose down"
