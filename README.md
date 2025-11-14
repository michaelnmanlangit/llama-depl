# Llama-3.2-1B Deployment on DigitalOcean

Deploy Meta's Llama-3.2-1B model as a web service on DigitalOcean with optimized RAM usage.

## 📋 Requirements

### Recommended DigitalOcean Droplet
- **Plan**: $49/month (Basic)
- **RAM**: 4GB 
- **vCPU**: 1 Dedicated vCPU
- **Bandwidth**: 400GB
- **OS**: Ubuntu 22.04 LTS

### Prerequisites
1. A DigitalOcean account
2. A HuggingFace account with access to Llama-3.2-1B model
3. HuggingFace token (get it from: https://huggingface.co/settings/tokens)

## 🚀 Quick Start

### Step 1: Request Model Access
1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B
2. Click "Request Access" and accept the license terms
3. Wait for approval (usually quick)

### Step 2: Get HuggingFace Token
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Copy the token

### Step 3: Deploy on DigitalOcean

SSH into your droplet:
```bash
ssh root@your-droplet-ip
```

Clone/upload this repository:
```bash
git clone <your-repo> llama-deployment
cd llama-deployment
```

Create `.env` file:
```bash
cp .env.example .env
nano .env  # Add your HuggingFace token
```

Run deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

## 🔧 Configuration

### Memory Optimization
The deployment uses **4-bit quantization** which reduces memory usage by ~75%:
- Original model: ~4.8GB (BF16)
- Quantized model: ~1.2GB (4-bit)
- Total runtime usage: ~2.5-3GB (including overhead)

This leaves comfortable headroom on a 4GB droplet.

### Environment Variables
Edit `.env` file:
```env
HF_TOKEN=hf_your_token_here
API_PORT=8000
MODEL_ID=meta-llama/Llama-3.2-1B
USE_QUANTIZATION=true
```

## 📡 API Usage

### Health Check
```bash
curl http://your-droplet-ip:8000/health
```

### Generate Text
```bash
curl -X POST http://your-droplet-ip:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The key to life is",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

### Python Example
```python
import requests

response = requests.post(
    "http://your-droplet-ip:8000/generate",
    json={
        "prompt": "Explain quantum computing in simple terms",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

print(response.json()["generated_text"])
```

### JavaScript Example
```javascript
fetch('http://your-droplet-ip:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Write a short poem about technology',
    max_new_tokens: 150,
    temperature: 0.8
  })
})
.then(r => r.json())
.then(data => console.log(data.generated_text));
```

## 📊 Monitoring

View logs:
```bash
docker-compose logs -f
```

Check stats:
```bash
curl http://your-droplet-ip:8000/stats
```

Monitor memory:
```bash
docker stats llama-3.2-1b-api
```

## 🛡️ Security Recommendations

1. **Firewall**: Configure UFW to only allow necessary ports
```bash
sudo ufw allow 22    # SSH
sudo ufw allow 8000  # API
sudo ufw enable
```

2. **Reverse Proxy**: Use Nginx with SSL for production
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Authentication**: Add API key authentication

## 🔄 Updates

Update the service:
```bash
git pull
docker-compose down
docker-compose build
docker-compose up -d
```

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce `max_new_tokens` in requests
- Restart the container: `docker-compose restart`
- Check available memory: `free -h`

### Model Loading Issues
- Verify HuggingFace token is correct
- Ensure you have access to the model
- Check logs: `docker-compose logs`

### Slow Performance
- This is normal for CPU inference on small droplets
- For better performance, consider:
  - Upgrading to 8GB RAM droplet
  - Using GPU-enabled droplet
  - Reducing `max_new_tokens`

## 💰 Cost Optimization

**Monthly Costs:**
- 4GB Droplet: $49/month
- Bandwidth: Included (400GB)
- Total: **~$49/month**

**Alternatives:**
- 2GB droplet ($21/mo): May work but tight on memory
- 8GB droplet ($84/mo): Better performance, more headroom

## 📚 Additional Resources

- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [DigitalOcean Pricing](https://www.digitalocean.com/pricing)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Transformers Library](https://huggingface.co/docs/transformers)

## 📄 License

This deployment setup is provided as-is. The Llama-3.2-1B model is governed by the Llama 3.2 Community License. Please review and comply with Meta's terms.

## ⚠️ Important Notes

1. **Model Access**: You MUST get approval from Meta/HuggingFace to use this model
2. **RAM**: 4GB is minimum; performance will be limited but functional
3. **CPU Inference**: Slower than GPU but works for moderate traffic
4. **Production**: Add authentication, monitoring, and caching for production use
