# 🧬 Evo2 Variant Pathogenicity Prediction API - Deployment Guide

A comprehensive guide to deploying the Extended Evo2 Variant Pathogenicity Prediction API on Modal.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Initial Setup](#-initial-setup)
- [Deployment Steps](#-deployment-steps)
- [Production Configuration](#-production-configuration)
- [Testing Your Deployment](#-testing-your-deployment)
- [Monitoring & Management](#-monitoring--management)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## ✅ Prerequisites

Before deploying, ensure you have:

- **Python 3.12+** installed
- **Git** installed
- **Modal account** ([Sign up for free](https://modal.com))
- **Basic understanding** of Python and REST APIs

### System Requirements

- **Local machine**: Any OS (Windows, macOS, Linux)
- **Modal account**: Free tier available (includes GPU credits)
- **Internet connection**: Required for deployment and API calls

---

## 🚀 Initial Setup

### Step 1: Install Modal CLI

```bash
# Install Modal Python package
pip install modal

# Verify installation
modal --version
```

### Step 2: Authenticate with Modal

Choose one of the following authentication methods:

#### Option A: Browser OAuth (Recommended for Development)

```bash
modal token new
```

This will:
1. Open your browser
2. Prompt you to sign in with GitHub, Google, or Email
3. Automatically save your credentials locally

#### Option B: API Token (Recommended for CI/CD)

1. Go to [Modal Settings → API Tokens](https://modal.com/settings)
2. Click **"Create Token"**
3. Copy the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
4. Set environment variables:

```bash
# Linux/macOS
export MODAL_TOKEN_ID="ak-xxxxxxxx"
export MODAL_TOKEN_SECRET="as-xxxxxxxx"

# Windows (PowerShell)
$env:MODAL_TOKEN_ID="ak-xxxxxxxx"
$env:MODAL_TOKEN_SECRET="as-xxxxxxxx"

# Windows (CMD)
set MODAL_TOKEN_ID=ak-xxxxxxxx
set MODAL_TOKEN_SECRET=as-xxxxxxxx
```

### Step 3: Verify Authentication

```bash
# Test your authentication
modal app list
```

If successful, you'll see a list of your Modal apps (may be empty initially).

---

## 📦 Deployment Steps

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Review Configuration

Open `main.py` and verify the following:

```python
# App name (change if needed)
app = modal.App("extended-evo2-snv-pathogenicity", image=evo2_image)

# GPU configuration
@app.cls(
    gpu="H100",                    # NVIDIA H100 GPU
    volumes={mount_path: volume},   # Model cache volume
    max_containers=3,              # Max parallel instances
    retries=2,                     # Auto-retry on failure
    scaledown_window=120           # Keep warm for 2 minutes
)
```

**Note**: H100 GPUs are premium resources. For testing, you can temporarily use `gpu="A10G"` or `gpu="T4"` (cheaper but slower).

### Step 3: Deploy the API

```bash
modal deploy main.py
```

**First deployment will take 10-15 minutes** because it needs to:
1. Build the Docker container image
2. Install all dependencies (CUDA, PyTorch, Evo2, etc.)
3. Download the Evo2 7B model (~14GB)
4. Create the HuggingFace cache volume

You'll see output like:

```
✓ Building image...
✓ Installing dependencies...
✓ Downloading model...
✓ Created app: extended-evo2-snv-pathogenicity
✓ Deployed Evo2Model.analyze_single_variant
  ↳ https://your-workspace--evo2-snv-pathogenicity-evo2model-analyze-single-variant.modal.run
```

**Save the endpoint URL** - you'll need it for API calls!

### Step 4: Verify Deployment

```bash
# List all deployed apps
modal app list

# View app details
modal app show extended-evo2-snv-pathogenicity

# View logs
modal app logs extended-evo2-snv-pathogenicity --follow
```

---

## 🔐 Production Configuration

### Setting Up API Key Authentication

For production use, secure your endpoint with API key authentication:

#### Step 1: Create Modal Secret

```bash
# Create a secret to store your API key
modal secret create evo2-api-key MODAL_API_KEY=your-secret-api-key-here
```

**Important**: Choose a strong, random API key. You can generate one:

```bash
# Linux/macOS
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### Step 2: Update main.py (Optional - Already Implemented)

The API already includes API key authentication. The `verify_api_key()` function will:
- Allow requests without API key if `MODAL_API_KEY` environment variable is not set (development mode)
- Require valid API key if `MODAL_API_KEY` is set (production mode)

#### Step 3: Use Secret in Deployment

Update your Modal class to use the secret:

```python
@app.cls(
    gpu="H100",
    volumes={mount_path: volume},
    max_containers=3,
    retries=2,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("evo2-api-key")]  # Add this line
)
class Evo2Model:
    ...
```

Then redeploy:

```bash
modal deploy main.py
```

#### Step 4: Test API Key Authentication

```bash
# Without API key (should fail in production mode)
curl -X POST "https://your-endpoint-url.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"variant_position": 43119628, "alternative": "G", "genome": "hg38", "chromosome": "chr17"}'

# With API key (should succeed)
curl -X POST "https://your-endpoint-url.modal.run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{"variant_position": 43119628, "alternative": "G", "genome": "hg38", "chromosome": "chr17"}'
```

---

## 🧪 Testing Your Deployment

### Test 1: Basic SNV Analysis

```bash
curl -X POST "https://your-endpoint-url.modal.run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "variant_position": 43119628,
    "alternative": "G",
    "genome": "hg38",
    "chromosome": "chr17",
    "mutation_type": "SNV"
  }'
```

**Expected Response:**

```json
{
  "position": 43119628,
  "chromosome": "chr17",
  "genome": "hg38",
  "reference": "A",
  "alternative": "G",
  "delta_score": -0.001234,
  "prediction": "Likely pathogenic",
  "classification_confidence": 0.85,
  "mutation_type": "SNV"
}
```

### Test 2: Deletion Analysis

```bash
curl -X POST "https://your-endpoint-url.modal.run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "variant_position": 43119628,
    "alternative": "-",
    "genome": "hg38",
    "chromosome": "chr17",
    "mutation_type": "DELETION"
  }'
```

**Expected Response:**

```json
{
  "reference": "T",
  "alternative": "",
  "delta_score": 0.00019371509552001953,
  "prediction": "Likely benign",
  "classification_confidence": 1.0,
  "mutation_type": "DELETION",
  "position": 43119628,
  "chromosome": "chr17",
  "genome": "hg38"
}
```

### Test 3: Insertion Analysis

```bash
curl -X POST "https://your-endpoint-url.modal.run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "variant_position": 43119628,
    "alternative": "ACGT",
    "genome": "hg38",
    "chromosome": "chr17",
    "mutation_type": "INSERTION"
  }'
```

```json
{
  "reference": "T",
  "alternative": "ACGT",
  "delta_score": -2.288818359375e-05,
  "prediction": "Likely benign",
  "classification_confidence": 0.9925746321678162,
  "mutation_type": "INSERTION",
  "position": 43119628,
  "chromosome": "chr17",
  "genome": "hg38"
}
```

### Test 4: Python Client Example

```python
import requests

# Your endpoint URL
url = "https://your-endpoint-url.modal.run"

# Your API key
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key-here"
}

# Request payload
payload = {
    "variant_position": 43119628,
    "alternative": "G",
    "genome": "hg38",
    "chromosome": "chr17",
    "mutation_type": "SNV"
}

# Make request
response = requests.post(url, json=payload, headers=headers)
response.raise_for_status()

# Parse response
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Delta Score: {result['delta_score']:.6f}")
print(f"Confidence: {result['classification_confidence']:.2%}")
```

---

## 📊 Monitoring & Management

### View Logs

```bash
# Stream logs in real-time
modal app logs evo2-snv-pathogenicity --follow

# View recent logs
modal app logs evo2-snv-pathogenicity --tail 100

# View logs for specific function
modal function logs evo2-snv-pathogenicity::Evo2Model::analyze_single_variant
```

### Monitor Usage

```bash
# View app status
modal app show evo2-snv-pathogenicity

# View running containers
modal container list

# View volume usage
modal volume list
```

### Dashboard Access

Visit [Modal Dashboard](https://modal.com/apps) to:
- View real-time metrics
- Monitor GPU usage
- Track API calls
- View error rates
- Check costs

---

## 🔄 Updating Your Deployment

### Redeploy After Code Changes

```bash
# Standard redeploy (uses cached image if possible)
modal deploy main.py

# Force rebuild (rebuilds container image)
modal deploy main.py --force-build
```

### Update Dependencies

1. Update `requirements.txt`
2. Redeploy:

```bash
modal deploy main.py --force-build
```

### Change Configuration

1. Edit `main.py` (e.g., change GPU type, max containers)
2. Redeploy:

```bash
modal deploy main.py
```

---

## 🐛 Troubleshooting

### Issue: Deployment Fails with "Build Timeout"

**Solution**: Increase build timeout or use a smaller GPU for testing.

```python
# In main.py, temporarily use a smaller GPU
@app.cls(
    gpu="A10G",  # Instead of H100
    ...
)
```

### Issue: "CUDA Out of Memory"

**Solution**: The model is too large for the selected GPU.

- Use H100 GPU (recommended)
- Or reduce model size (if using custom model)

### Issue: "Model Download Fails"

**Solution**: Check HuggingFace access and retry.

```bash
# Force rebuild to retry download
modal deploy main.py --force-build
```

### Issue: "UCSC API Error"

**Solution**: Verify chromosome format.

- ✅ Correct: `"chr17"`, `"chrX"`
- ❌ Incorrect: `"17"`, `"X"`

### Issue: "API Key Not Working"

**Solution**: Verify secret is attached and environment variable is set.

```bash
# Check if secret exists
modal secret list

# Verify secret contents (will show masked value)
modal secret show evo2-api-key

# Check if secret is attached to app
modal app show evo2-snv-pathogenicity
```

### Issue: "Endpoint Returns 404"

**Solution**: Verify endpoint URL and deployment status.

```bash
# List all endpoints
modal app list

# Get endpoint URL
modal app show evo2-snv-pathogenicity
```

### Issue: Slow First Request (Cold Start)

**Solution**: This is normal. The container needs to:
1. Start up (~10-30 seconds)
2. Load model into GPU memory (~30-60 seconds)

**Mitigation**: Use `scaledown_window` to keep containers warm:

```python
@app.cls(
    ...
    scaledown_window=300,  # Keep warm for 5 minutes
)
```

---

## 📡 API Reference

### Endpoint

```
POST https://your-workspace--evo2-snv-pathogenicity-evo2model-analyze-single-variant.modal.run
```

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-API-Key` | Conditional | Required if `MODAL_API_KEY` is set |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `variant_position` | `int` | Yes | Genomic position (1-based) |
| `alternative` | `str` | Yes | Alternative allele (see mutation types below) |
| `genome` | `str` | Yes | Genome assembly (e.g., `"hg38"`, `"hg19"`) |
| `chromosome` | `str` | Yes | Chromosome (e.g., `"chr17"`, `"chr1"`) |
| `mutation_type` | `str` | No | Mutation type (see below). Default: `"SNV"` |
| `reference` | `str` | No | Reference allele (auto-detected if not provided) |

### Mutation Types

#### SNV (Single Nucleotide Variant)
- `alternative`: Single nucleotide (`"A"`, `"C"`, `"G"`, or `"T"`)
- Example: `{"mutation_type": "SNV", "alternative": "G"}`

#### DELETION
- `alternative`: `"-"` or empty string `""`
- `reference`: Nucleotide(s) to delete
- Example: `{"mutation_type": "DELETION", "alternative": "-", "reference": "A"}`

#### INSERTION
- `alternative`: Sequence to insert (e.g., `"ACGT"`)
- `reference`: Nucleotide before insertion point
- Example: `{"mutation_type": "INSERTION", "alternative": "ACGT", "reference": "A"}`

#### DUPLICATION
- `alternative`: Number of copies (e.g., `"2"`, `"3"`) or the duplicated sequence itself (e.g., `"ATCGATCG"`)
- `reference`: Sequence to duplicate
- Example: `{"mutation_type": "DUPLICATION", "alternative": "2", "reference": "ATCG"}`

#### MICROSATELLITE
- `alternative`: Expanded/contracted repeat sequence (e.g., `"CAGCAGCAG"` for expansion)
- `reference`: Repeat unit (e.g., `"CAG"`)
- Example: `{"mutation_type": "MICROSATELLITE", "alternative": "CAGCAGCAG", "reference": "CAG"}`

#### INDEL (Insertion-Deletion)
- `alternative`: Sequence to insert
- `reference`: Sequence to delete
- Example: `{"mutation_type": "INDEL", "alternative": "ACGT", "reference": "AT"}`

#### INVERSION
- `alternative`: Optional - if provided, should match reversed reference
- `reference`: Sequence to reverse
- Example: `{"mutation_type": "INVERSION", "reference": "ATCG"}`

#### TRANSLOCATION
- `alternative`: Sequence to insert at new location
- `reference`: Sequence to remove from original location
- Example: `{"mutation_type": "TRANSLOCATION", "alternative": "ATCG", "reference": "GCAT"}`

### Response Schema

```json
{
  "position": 43119628,
  "chromosome": "chr17",
  "genome": "hg38",
  "reference": "A",
  "alternative": "G",
  "delta_score": -0.001234,
  "prediction": "Likely pathogenic",
  "classification_confidence": 0.85,
  "mutation_type": "SNV"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `position` | `int` | Genomic position |
| `chromosome` | `str` | Chromosome identifier |
| `genome` | `str` | Genome assembly |
| `reference` | `str` | Reference allele |
| `alternative` | `str` | Alternative allele |
| `delta_score` | `float` | Log-likelihood difference (negative = pathogenic) |
| `prediction` | `str` | `"Likely pathogenic"` or `"Likely benign"` |
| `classification_confidence` | `float` | Confidence score (0.0 to 1.0) |
| `mutation_type` | `str` | Type of mutation analyzed |

### Error Responses

#### 400 Bad Request
```json
{
  "detail": "For SNV, alternative must be a single nucleotide (A, C, G, or T)"
}
```

#### 401 Unauthorized
```json
{
  "detail": "API key required. Please provide X-API-Key header."
}
```

#### 403 Forbidden
```json
{
  "detail": "Invalid API key."
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Failed to fetch genome sequence from UCSC API: 500"
}
```

---

## 💰 Cost Considerations

### GPU Pricing (Approximate)

| GPU Type | Cost per Hour | Use Case |
|----------|---------------|----------|
| H100 | ~$4-8/hour | Production (fastest) |
| A100 | ~$2-4/hour | Production (good balance) |
| A10G | ~$1-2/hour | Development/testing |
| T4 | ~$0.50/hour | Light testing |

### Cost Optimization Tips

1. **Use appropriate GPU**: Use A10G or T4 for development
2. **Set scaledown_window**: Keep containers warm to avoid cold starts
3. **Monitor usage**: Check Modal dashboard regularly
4. **Set max_containers**: Limit parallel instances to control costs

---

## 🔗 Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Evo2 Paper](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)
- [UCSC Genome Browser API](https://genome.ucsc.edu/goldenPath/help/api.html)

---

## 📝 Quick Reference Commands

```bash
# Deploy
modal deploy main.py

# View logs
modal app logs evo2-snv-pathogenicity --follow

# List apps
modal app list

# Show app details
modal app show evo2-snv-pathogenicity

# Create secret
modal secret create evo2-api-key MODAL_API_KEY=your-key

# View secrets
modal secret list

# Force rebuild
modal deploy main.py --force-build

# Delete app (careful!)
modal app stop evo2-snv-pathogenicity
```

---

## ✅ Deployment Checklist

- [ ] Modal CLI installed and authenticated
- [ ] `main.py` reviewed and configured
- [ ] API deployed successfully
- [ ] Endpoint URL saved
- [ ] API key secret created (for production)
- [ ] Test request successful
- [ ] Logs monitored
- [ ] Dashboard access verified
- [ ] Frontend configured with endpoint URL

---

**Need Help?** Check the [Troubleshooting](#-troubleshooting) section or visit the [Modal Community](https://modal.com/community).
