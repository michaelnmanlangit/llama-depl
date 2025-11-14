# 🚀 Quick Deployment Guide for DigitalOcean

You've already completed the HuggingFace setup! Here's what to do next:

## ✅ Prerequisites Complete
- [x] HuggingFace account created
- [x] Access token generated: `hf_KEsRi...wSUQ`
- [x] Llama-3.2-1B model access approved
- [x] `.env` file configured

## 📦 What You Have
All files are ready in: `c:\Users\manla\Downloads\data\llama-deployment\`

## 🎯 Next Steps

### Option 1: Deploy to DigitalOcean (Recommended)

1. **Create DigitalOcean Droplet**
   - Go to https://cloud.digitalocean.com/droplets/new
   - Choose: **Ubuntu 22.04 LTS**
   - Select: **$49/month plan (4GB RAM, 1 vCPU)**
   - Choose a datacenter region near you
   - Add your SSH key
   - Click "Create Droplet"

2. **Upload Files to Droplet**
   ```bash
   # From your local machine (PowerShell)
   # Replace YOUR_DROPLET_IP with actual IP
   scp -r c:\Users\manla\Downloads\data\llama-deployment root@YOUR_DROPLET_IP:/root/
   ```

3. **SSH into Droplet**
   ```bash
   ssh root@YOUR_DROPLET_IP
   ```

4. **Deploy**
   ```bash
   cd /root/llama-deployment
   chmod +x deploy.sh
   ./deploy.sh
   ```

5. **Access Your API**
   - API: `http://YOUR_DROPLET_IP:8000`
   - Docs: `http://YOUR_DROPLET_IP:8000/docs`
   - Health: `http://YOUR_DROPLET_IP:8000/health`

### Option 2: Test Locally First (Windows)

Since you're on Windows, you can test locally with Docker Desktop:

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and start Docker Desktop

2. **Open PowerShell in the deployment folder**
   ```powershell
   cd c:\Users\manla\Downloads\data\llama-deployment
   ```

3. **Build and Run**
   ```powershell
   docker-compose build
   docker-compose up
   ```

4. **Test the API**
   - Open browser: http://localhost:8000/docs
   - Or use the test script:
   ```powershell
   python test_api.py
   ```

## 🧪 Quick API Test

Once deployed, test with curl or browser:

```bash
# Health check
curl http://YOUR_IP:8000/health

# Generate text
curl -X POST http://YOUR_IP:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Hello, I am\", \"max_new_tokens\": 50}"
```

## 📊 Expected Performance

**On 4GB DigitalOcean Droplet:**
- First request: 60-120s (model loading)
- Subsequent requests: 5-30s depending on length
- Memory usage: ~2.5-3GB
- Concurrent requests: 1-2 recommended

## 🔧 Troubleshooting

**Model download takes long?**
- Normal! The model is ~1.2GB quantized
- First startup takes 2-5 minutes

**Out of memory?**
- Reduce `max_new_tokens` in requests (try 128 instead of 512)
- Restart container: `docker-compose restart`

**Can't connect?**
- Check firewall: `sudo ufw allow 8000`
- Verify container is running: `docker ps`

## 💰 Cost Estimate

**Monthly:**
- DigitalOcean Droplet (4GB): $49/month
- Bandwidth: Included (400GB)
- **Total: ~$49/month**

## 🎉 You're Ready!

All configuration is complete. Choose your deployment option above and you're good to go!

Need help? Check the main README.md for detailed documentation.
