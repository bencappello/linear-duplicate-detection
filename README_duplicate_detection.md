# Linear AI Duplicate Detection System

An AI-powered system to automatically detect and flag duplicate tickets in Linear using semantic similarity search and GPT-4 analysis. This system is designed for bug bounty programs where maintaining chronological ticket integrity is crucial.

## 🎯 What This System Does

1. **Fetches Linear Issues**: Downloads all tickets from your Linear workspace
2. **Semantic Search**: Uses OpenAI embeddings to find similar tickets based on content
3. **AI Analysis**: GPT-4 analyzes similarities and determines duplicates with confidence scores
4. **Automated Actions**: Labels duplicates, links related issues, and adds detailed comments
5. **Chronological Protection**: Ensures newer tickets are never marked as duplicates of older ones (critical for bug bounty programs)

## 📁 Project Structure

```
bug_bounty/
├── ai_flag_duplicates_improved.py    # Main duplicate detection script (CURRENT VERSION)
├── fetch_linear_issues.py           # Script to fetch Linear issues to JSON
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── README_duplicate_detection.md    # This file
├── improvements_summary.md          # Technical improvements documentation
├── ai_flag_duplicates_updated.py    # Previous version (deprecated)
└── ai_flag_duplicates.py           # Original version (deprecated)
```

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone/download the project
cd /path/to/bug_bounty

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the template and add your API keys:

```bash
# Copy the template
cp .env.example .env

# Edit .env with your actual API keys
```

Your `.env` file should contain:

```bash
# Required API Keys
CHROMA_OPENAI_API_KEY=your-openai-api-key-here
LINEAR_OAUTH_ACCESS_TOKEN=your-linear-api-token-here

# Optional Configuration
LINEAR_TEAM_ID=your-team-id  # defaults to the ID in script
CHROMA_DB_PATH=/path/to/chromadb  # defaults to ~/.chroma/bug-bounty
CONFIDENCE_THRESHOLD=65  # 0-100, default 65
MAX_CANDIDATES=10  # number of similar tickets to analyze, default 10
```

**To get your API keys:**
- **OpenAI**: Go to https://platform.openai.com/api-keys
- **Linear**: Go to Linear Settings > API > Create personal API key

### 3. Fetch Your Linear Issues

```bash
# Download all Linear issues to JSON file
python fetch_linear_issues.py

# This creates ~/linear_issues.json with all your tickets
```

### 4. Initialize the System

```bash
# Seed ChromaDB with your issues (one-time setup)
python ai_flag_duplicates_improved.py --seed

# This creates embeddings for all tickets and stores them locally
```

### 5. Test the System

```bash
# Test on a single ticket (dry run)
python ai_flag_duplicates_improved.py --tickets BUG-123 --dry-run

# Process a single ticket for real
python ai_flag_duplicates_improved.py --tickets BUG-123
```

## 📖 Detailed Usage

### Fetching Linear Issues

The `fetch_linear_issues.py` script downloads all issues from your Linear workspace:

```bash
# Basic usage - saves to ~/linear_issues.json
python fetch_linear_issues.py

# Custom output location
python fetch_linear_issues.py /path/to/custom/issues.json

# Filter by specific team (optional)
export LINEAR_TEAM_ID=your-team-id
python fetch_linear_issues.py
```

**When to re-fetch:**
- New tickets have been created in Linear
- You want to refresh ticket data (descriptions, states, etc.)
- You're switching to a different team or workspace

### ChromaDB Seeding

The seeding process creates embeddings for semantic similarity search:

```bash
# Initial seeding (one-time setup)
python ai_flag_duplicates_improved.py --seed

# Update existing entries (after re-fetching Linear data)
python ai_flag_duplicates_improved.py --seed --update-existing

# Custom issues file location
python ai_flag_duplicates_improved.py --seed --issues-file /path/to/issues.json
```

### Processing Tickets
### (THIS IS WHAT YOU WILL USE 99% OF THE TIME)

#### Single and Multi Ticket Processing
```bash
# Process one ticket
python ai_flag_duplicates_improved.py --tickets BUG-123

# Process multiple specific tickets
python ai_flag_duplicates_improved.py --tickets BUG-123 BUG-456 BUG-789
```

### Other Processing Options
### (Mostly for debugging purposes)


#### Bulk Processing
#### (don't do this because it will reprocess all the old tickets)
```bash
# Process all tickets (don't do this because it will reprocess all the old tickets)
python ai_flag_duplicates_improved.py --all

# Process with custom confidence threshold
python ai_flag_duplicates_improved.py --all --confidence-threshold 80
```

#### Dry Run Mode
```bash
# See what would happen without making changes in Linear
python ai_flag_duplicates_improved.py --tickets BUG-123 --dry-run
python ai_flag_duplicates_improved.py --all --dry-run
```

### Advanced Options

```bash
# Debug mode with detailed logging
python ai_flag_duplicates_improved.py --tickets BUG-123 --log-level DEBUG

# Log to file
python ai_flag_duplicates_improved.py --all --log-file duplicate_detection.log

# Combine multiple options
python ai_flag_duplicates_improved.py \
    --all \
    --dry-run \
    --log-level INFO \
    --log-file run_$(date +%Y%m%d_%H%M%S).log \
    --confidence-threshold 70
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_OPENAI_API_KEY` | OpenAI API key (required) | None |
| `LINEAR_OAUTH_ACCESS_TOKEN` | Linear API token (required) | None |
| `LINEAR_TEAM_ID` | Filter issues by team ID | Auto-detected |
| `CHROMA_DB_PATH` | ChromaDB storage location | `~/.chroma/bug-bounty` |
| `CONFIDENCE_THRESHOLD` | Min confidence for duplicates (0-100) | 65 |
| `MAX_CANDIDATES` | Max similar tickets to analyze | 10 |

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--seed` | Seed ChromaDB with issues |
| `--update-existing` | Update existing ChromaDB entries |
| `--tickets BUG-123 BUG-456` | Process specific tickets |
| `--all` | Process all tickets |
| `--issues-file PATH` | Custom issues JSON file path |
| `--dry-run` | Test without making changes |
| `--log-level DEBUG` | Set logging level |
| `--log-file FILE` | Log to file |
| `--confidence-threshold 80` | Override confidence threshold |

## 🎯 Common Workflows

### 1. Initial Setup (New Teammate)
```bash
# 1. Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your actual API keys

# 3. Fetch Linear data
python fetch_linear_issues.py

# 4. Seed ChromaDB
python ai_flag_duplicates_improved.py --seed

# 5. Test on a few tickets
python ai_flag_duplicates_improved.py --tickets BUG-1 BUG-2 BUG-3 --dry-run
```

### 2. Regular Maintenance (Processing New Tickets)
```bash
# 1. Update Linear data
python fetch_linear_issues.py

# 2. Update ChromaDB with new tickets
python ai_flag_duplicates_improved.py --seed --update-existing

# 3. Process new tickets (example: BUG-559 through BUG-780)
python ai_flag_duplicates_improved.py --tickets BUG-559 BUG-560 BUG-561
# OR process all new tickets at once
```

### 3. Bulk Backfill (One-time Process)
```bash
# Process all existing tickets
python ai_flag_duplicates_improved.py --all --log-file backfill_$(date +%Y%m%d).log
```

## 📊 Understanding the Output

### Console Output
```
2024-01-10 10:15:23 - INFO - Processing ticket: BUG-123
2024-01-10 10:15:24 - INFO - Found 8 potential duplicate candidates
2024-01-10 10:15:24 - INFO - Date filtering reduced candidates from 12 to 8
2024-01-10 10:15:26 - INFO - LLM identified 1 duplicates above confidence threshold
2024-01-10 10:15:27 - INFO - Successfully added label to issue BUG-123
2024-01-10 10:15:28 - INFO - Successfully linked BUG-123 as duplicate of BUG-456
==================================================
SUMMARY
==================================================
Total tickets processed: 150
Total duplicates found: 23
✅ Processing completed
```

### Linear Comments
The system adds professional comments to tickets:

```
🤖 AI Duplicate Detection

This ticket appears to be a duplicate of BUG-456.

Confidence: 85%

Analysis: Both tickets describe the same API timeout issue with identical error messages and stack traces

Please review this assessment. If incorrect, you can unlink the issues.
```

### Linear Actions Taken
- **Label**: Adds "ai-flagged-duplicate" label to tickets
- **Link**: Creates "related" relationships between duplicate tickets
- **Comment**: Adds detailed explanation of the duplicate detection

## 🔍 How It Works

### 1. Semantic Similarity Search
- Uses OpenAI's `text-embedding-3-small` model to create embeddings
- Stores embeddings in ChromaDB for fast vector search
- Finds tickets with similar titles and descriptions

### 2. Chronological Protection
- Only considers tickets created **before** the current ticket
- Prevents newer tickets from being marked as duplicates of older ones
- Critical for bug bounty programs where first reporter gets credit

### 3. AI Analysis
- GPT-4-mini analyzes semantic similarities
- Considers exact duplicates, root cause duplicates, and pattern duplicates
- Provides confidence scores and detailed explanations

### 4. Conservative Approach
- Default 65% confidence threshold prevents false positives
- Dry run mode for testing
- All actions are reversible in Linear

## 🛠️ Troubleshooting

### Common Issues

**1. "API keys must be set as environment variables"**
- Make sure you copied `.env.example` to `.env` and added your actual API keys
- Verify the keys are not expired or invalid
- Check that the `.env` file is in the project root directory

**2. "Issues file not found"**
- Run `python fetch_linear_issues.py` first
- Check if `~/linear_issues.json` exists
- Use `--issues-file` to specify custom path

**3. "Could not find internal ID for identifier"**
- The ticket might not exist in your fetched issues
- Re-run `python fetch_linear_issues.py` to update your data
- Check if the ticket ID is correct

**4. "GraphQL errors" or "400 Bad Request"**
- Linear API token might be expired or invalid
- Check your `LINEAR_OAUTH_ACCESS_TOKEN` in `.env`
- Verify you have the correct permissions in Linear

### Debug Mode

For detailed troubleshooting, use debug mode:

```bash
python ai_flag_duplicates_improved.py --tickets BUG-123 --log-level DEBUG --dry-run
```

This will show:
- Detailed API requests and responses
- ChromaDB query results
- LLM prompt and response
- Step-by-step processing logic

## 📈 Performance & Costs

### Processing Speed
- **ChromaDB seeding**: ~1-2 minutes for 500 tickets
- **Individual ticket**: ~7-8 seconds (including AI analysis)
- **Bulk processing**: ~15-30 minutes for 500 tickets

### API Costs (Approximate)
- **OpenAI Embeddings**: ~$0.0001 per ticket
- **GPT-4-mini Analysis**: ~$0.001 per ticket
- **Total**: ~$0.0011 per ticket processed

### Rate Limiting
- Built-in delays between API calls
- Configurable rate limiting
- Automatic retry logic for failed requests

## 🔒 Security & Privacy

- API keys stored in local `.env` file (not in code)
- ChromaDB stored locally (not in cloud)
- No ticket data sent to external services except OpenAI for analysis
- All actions logged for audit trail

## 🔄 Maintenance

### Regular Tasks
1. **Update Linear data**: Run `fetch_linear_issues.py` periodically
2. **Update ChromaDB**: Use `--seed --update-existing` after fetching new data
3. **Process new tickets**: Run duplicate detection on new tickets
4. **Monitor logs**: Check for errors or API issues

### Monitoring
- Check log files for errors
- Monitor OpenAI API usage
- Review false positives/negatives in Linear
- Adjust confidence threshold if needed

## 📚 Additional Resources

- **improvements_summary.md**: Technical details about improvements made
- **Linear API Documentation**: https://developers.linear.app/
- **OpenAI API Documentation**: https://platform.openai.com/docs
- **ChromaDB Documentation**: https://docs.trychroma.com/

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run with `--log-level DEBUG` for detailed output
3. Check log files for error messages
4. Verify API keys and permissions
5. Test with `--dry-run` mode first

The system is designed to be safe and reversible - all actions can be undone manually in Linear if needed. 