# Quick Deployment Guide for DigitalOcean Droplet

## Why Droplet Instead of App Platform?

App Platform has build time/memory limits that make it difficult to deploy large ML models.
A traditional Droplet gives you full control and sufficient resources.

## Step-by-Step Deployment

### 1. Create a Droplet

1. Go to: https://cloud.digitalocean.com/droplets/new
2. Choose:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic
   - **CPU**: Regular (4GB RAM / 2 vCPU - $24/mo) OR Premium (4GB RAM / 2 vCPU - $42/mo)
   - **Datacenter**: Choose closest to you
   - **Authentication**: SSH Key (recommended) or Password
3. Click **Create Droplet**
4. Wait ~60 seconds for it to boot
5. **Copy the IP address**

### 2. Connect to Your Droplet

```bash
# From PowerShell or terminal
ssh root@YOUR_DROPLET_IP
```

### 3. Clone and Deploy

```bash
# Install git if needed
apt-get update
apt-get install -y git

# Clone your repository
git clone https://github.com/michaelnmanlangit/llama-depl.git
cd llama-depl

# Create .env file with your token
cat > .env << 'EOF'
HF_TOKEN=your_huggingface_token_here
API_PORT=8000
API_HOST=0.0.0.0
MODEL_ID=meta-llama/Llama-3.2-1B
USE_QUANTIZATION=true
MAX_WORKERS=2
TIMEOUT=120
EOF

# Run the deployment script
chmod +x deploy.sh
./deploy.sh
```

### 4. Wait for Model Download

The first time will take 5-10 minutes to:
- Install Docker
- Download the model (~1.2GB)
- Start the service

### 5. Test Your API

```bash
# Check if it's running
curl http://localhost:8000/health

# Test generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am", "max_new_tokens": 50}'
```

### 6. Access from Outside

Your API will be available at:
- **API**: `http://YOUR_DROPLET_IP:8000`
- **Docs**: `http://YOUR_DROPLET_IP:8000/docs`

## Pricing Comparison

| Option | RAM | Cost/Month | Best For |
|--------|-----|------------|----------|
| Regular 4GB | 4GB | $24 | Development/Testing |
| Premium 4GB | 4GB | $42 | Better CPU performance |
| Regular 8GB | 8GB | $48 | Production (recommended) |

**Recommendation**: Start with **Regular 4GB ($24/mo)** or **Regular 8GB ($48/mo)** for better performance.

## Configure Firewall (Important!)

```bash
# On your droplet
ufw allow 22    # SSH
ufw allow 8000  # API
ufw enable
```

## Monitor Your Service

```bash
# View logs
docker-compose logs -f

# Check memory usage
docker stats

# Restart if needed
docker-compose restart
```

## Common Issues

**Out of Memory?**
- Upgrade to 8GB droplet ($48/mo)
- Or reduce max_new_tokens in requests

**Slow responses?**
- Normal for CPU inference
- First request after startup is slowest
- Consider 8GB droplet for better performance

## Next Steps After Deployment

1. **Add SSL**: Use Nginx + Let's Encrypt
2. **Domain**: Point your domain to the droplet IP
3. **Monitoring**: Set up monitoring alerts
4. **Backups**: Enable DigitalOcean backups

## Need Help?

Check the main README.md or deployment logs:
```bash
docker-compose logs
```
