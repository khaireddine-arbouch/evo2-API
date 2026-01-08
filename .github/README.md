# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and system monitoring.

## Workflows

### System Health Check (`system-health-check.yml`)

This workflow monitors the health of the deployed Evo2 Variant Pathogenicity Prediction API.

**Features:**
- Automatically runs every 6 hours
- Can be manually triggered via workflow_dispatch
- Runs on code changes to backend files
- Tests the API endpoint with a real variant analysis request
- Reports system status

**Setup Instructions:**

1. **Add GitHub Secrets:**
   - Go to your repository Settings → Secrets and variables → Actions
   - Add the following secrets:
     - `MODAL_API_ENDPOINT`: Your Modal API endpoint URL
       - Example: `https://your-workspace--evo2-snv-pathogenicity-evo2model-analyze-single-variant.modal.run`
     - `MODAL_API_KEY`: Your API key (if authentication is enabled)
       - Optional: Only required if your API uses authentication

2. **Get Your API Endpoint:**
   ```bash
   modal app show extended-evo2-snv-pathogenicity
   ```
   Look for the endpoint URL in the output.

3. **Test the Workflow:**
   - Go to Actions tab in your repository
   - Select "System Health Check" workflow
   - Click "Run workflow" to manually trigger it

**Workflow Behavior:**
- ✅ **Healthy**: API responds with 200 status and valid JSON
- ⚠️ **Auth Failed**: API is reachable but authentication failed
- ❌ **Unhealthy**: API is unreachable, timed out, or returned an error

**Manual Testing:**

You can also test the health check script locally:

```bash
export MODAL_API_ENDPOINT="https://your-endpoint-url.modal.run"
export MODAL_API_KEY="your-api-key"  # Optional

python .github/scripts/health_check.py
```

## Workflow Status Badge

To add a status badge to your README, add this markdown:

```markdown
![System Health Check](https://github.com/your-username/your-repo/workflows/System%20Health%20Check/badge.svg)
```

Replace `your-username` and `your-repo` with your actual GitHub username and repository name.
