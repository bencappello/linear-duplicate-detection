# Linear Duplicate Detection - Automation Setup Guide

This guide will help you automate your Linear duplicate detection system using Chroma Cloud and GitHub Actions.

## 🚀 Complete Setup Guide

### Phase 1: Set up Chroma Cloud (5 minutes)

1. **Sign up for Chroma Cloud**
   - Go to [Chroma Cloud](https://www.trychroma.com/)
   - Sign up for the **Starter plan** (free with $5 credits)
   - Create a new database/tenant

2. **Get your Chroma Cloud credentials**
   - Copy your `CHROMA_API_URL` (e.g., `https://api.trychroma.com`)
   - Copy your `CHROMA_API_KEY` from the dashboard
   - Save these for the next step

### Phase 2: Configure GitHub Repository Secrets (5 minutes)

1. **Go to your GitHub repository**
   - Navigate to **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**

2. **Add the following secrets:**
   ```
   OPENAI_API_KEY: your-openai-api-key
   LINEAR_API_KEY: your-linear-api-token
   LINEAR_TEAM_ID: your-linear-team-id
   CHROMA_API_URL: your-chroma-api-url
   CHROMA_API_KEY: your-chroma-api-key
   ```

3. **Get your API keys if you don't have them:**
   - **OpenAI**: Go to https://platform.openai.com/api-keys
   - **Linear**: Go to Linear Settings → API → Create personal API key
   - **Linear Team ID**: Check your current `.env` file or Linear URL

### Phase 3: Initial Setup and Testing (10 minutes)

1. **Test the automation locally first** (recommended):
   ```bash
   # Set up environment variables
   export OPENAI_API_KEY=your-openai-api-key
   export LINEAR_API_KEY=your-linear-api-token
   export LINEAR_TEAM_ID=your-team-id
   export CHROMA_API_URL=your-chroma-url
   export CHROMA_API_KEY=your-chroma-key
   
   # Test the cloud script
   python ai_flag_duplicates_cloud.py --tickets BUG-123 --dry-run
   ```

2. **Push your changes to GitHub:**
   ```bash
   git add .
   git commit -m "Add automated duplicate detection with Chroma Cloud"
   git push origin main
   ```

3. **Trigger the initial setup:**
   - Go to **Actions** tab in your GitHub repository
   - Find the **Linear Duplicate Detection** workflow
   - Click **Run workflow**
   - Leave inputs empty and click **Run workflow**

### Phase 4: Configure Automation Options

#### Option A: Daily Scheduled Runs (Recommended)
The workflow is configured to run daily at 1 AM UTC. It will:
1. Fetch latest Linear issues
2. Update ChromaDB with new issues
3. Process only tickets created in the last 24 hours (efficient!)

#### Option B: Manual Triggers
You can manually trigger the workflow:
1. Go to **Actions** → **Linear Duplicate Detection**
2. Click **Run workflow**
3. Optionally specify specific tickets: `BUG-123 BUG-456 BUG-789`
4. Enable **dry-run mode** to test without making changes

#### Option C: Webhook Triggers (Advanced)
For real-time processing, you can set up Linear webhooks:
1. Add a webhook trigger to the workflow
2. Configure Linear to send webhook on issue creation
3. Process tickets immediately when they're created

## 🔧 How It Works

### Automated Daily Process:
1. **1 AM UTC**: GitHub Actions triggers the workflow
2. **Fetch Issues**: Downloads latest Linear issues
3. **Update ChromaDB**: Adds new issues to Chroma Cloud
4. **Process Recent**: Only processes tickets from last 24 hours
5. **Flag Duplicates**: Adds labels and comments to duplicates

### Manual Process:
1. **Trigger**: You manually run the workflow
2. **Custom Tickets**: Specify which tickets to process
3. **Dry Run**: Test mode available
4. **Immediate**: Runs right away

## 📊 Monitoring and Logs

### Check Workflow Status:
- Go to **Actions** tab in GitHub
- Click on the latest **Linear Duplicate Detection** run
- View logs for each step

### View Results:
- Check Linear for new `ai-flagged-duplicate` labels
- Look for AI comments on tickets
- Monitor linked issues

### Troubleshooting:
- Failed runs will upload logs as artifacts
- Check **Actions** → **Failed Run** → **Artifacts**
- Common issues:
  - API key expiration
  - Rate limiting
  - Network connectivity

## 💰 Cost Estimate

### Chroma Cloud (Monthly):
- **Free tier**: $5 credit (sufficient for 1 month)
- **Usage**: ~$0.10-0.50/month for 500-2000 tickets
- **Storage**: ~$0.10/month for embeddings

### GitHub Actions (Monthly):
- **Free tier**: 2,000 minutes/month
- **Usage**: ~30 minutes/month (1 min/day)
- **Cost**: $0 (well within free tier)

### OpenAI API (Monthly):
- **Embeddings**: ~$0.50-2.00/month
- **GPT-4-mini**: ~$1.00-5.00/month
- **Total**: ~$1.50-7.00/month

## 🎯 Optimization Tips

### For Better Performance:
1. **Process Recent Only**: Use scheduled runs for efficiency
2. **Batch Processing**: Process multiple tickets at once
3. **Confidence Tuning**: Adjust threshold based on results

### For Better Accuracy:
1. **Monitor Results**: Check false positives/negatives
2. **Adjust Prompts**: Modify LLM prompts if needed
3. **Confidence Threshold**: Start with 65%, adjust as needed

### For Cost Optimization:
1. **Scheduled Runs**: More efficient than real-time
2. **Batch Processing**: Fewer API calls
3. **Filtering**: Only process relevant tickets

## 🔧 Configuration Options

### Environment Variables:
```bash
# Required
OPENAI_API_KEY=your-openai-key
LINEAR_API_KEY=your-linear-key
CHROMA_API_URL=your-chroma-url
CHROMA_API_KEY=your-chroma-key

# Optional
LINEAR_TEAM_ID=your-team-id
CONFIDENCE_THRESHOLD=65
MAX_CANDIDATES=10
```

### Workflow Customization:
- **Schedule**: Edit cron in `.github/workflows/duplicate-detection.yml`
- **Timeout**: Adjust `timeout-minutes` for large batches
- **Retention**: Change log retention days

## 🚨 Important Notes

### Security:
- Never commit API keys to Git
- Use GitHub Secrets for all sensitive data
- Rotate keys regularly

### Rate Limits:
- GitHub Actions: 2,000 minutes/month free
- Linear API: 1,000 requests/hour
- OpenAI API: Variable by plan

### Maintenance:
- Monitor workflow success/failure
- Check ChromaDB storage usage
- Update dependencies periodically

## 🎉 You're Done!

Your duplicate detection system is now fully automated! It will:
- ✅ Run daily automatically
- ✅ Process only new tickets efficiently
- ✅ Scale to handle 100-300 tickets/day
- ✅ Cost less than $10/month
- ✅ Require minimal maintenance

### Next Steps:
1. **Monitor**: Check the first few runs
2. **Adjust**: Tune confidence threshold if needed
3. **Scale**: Add more processing power if needed
4. **Optimize**: Fine-tune based on results

### Need Help?
- Check the **Actions** logs for errors
- Review the **troubleshooting** section above
- Test locally with `--dry-run` first 