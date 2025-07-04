#!/usr/bin/env python3
"""
Helper script to process only recent tickets (last 24 hours) for scheduled runs.
This is more efficient than processing all tickets every time.
"""

import json
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


def load_issues(issues_file: str = "linear_issues.json") -> List[dict]:
    """Load Linear issues from JSON file."""
    try:
        with open(issues_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Issues file {issues_file} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {issues_file}: {e}")
        return []


def get_recent_tickets(issues: List[dict], hours: int = 24) -> List[str]:
    """Get ticket IDs created in the last N hours."""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_tickets = []
    
    for issue in issues:
        if not issue.get('createdAt') or not issue.get('identifier'):
            continue
            
        try:
            # Parse the creation date (ISO format from Linear)
            created_at = datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))
            
            # Remove timezone info for comparison (assuming UTC)
            created_at = created_at.replace(tzinfo=None)
            
            if created_at >= cutoff_time:
                recent_tickets.append(issue['identifier'])
                
        except ValueError as e:
            print(f"Warning: Could not parse date for {issue.get('identifier', 'unknown')}: {e}")
            continue
    
    return recent_tickets


def main():
    """Main function to process recent tickets."""
    print("Processing recent tickets...")
    
    # Load issues
    issues = load_issues()
    if not issues:
        print("No issues found, exiting.")
        return 1
    
    # Get recent tickets (last 24 hours)
    recent_tickets = get_recent_tickets(issues, hours=24)
    
    if not recent_tickets:
        print("No recent tickets found in the last 24 hours.")
        return 0
    
    print(f"Found {len(recent_tickets)} recent tickets: {', '.join(recent_tickets[:5])}")
    if len(recent_tickets) > 5:
        print(f"... and {len(recent_tickets) - 5} more")
    
    # Process the recent tickets
    cmd = [
        sys.executable, 
        "ai_flag_duplicates_improved.py", 
        "--tickets"
    ] + recent_tickets
    
    print(f"Running: {' '.join(cmd[:3])} --tickets {' '.join(recent_tickets)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Recent tickets processed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing recent tickets: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 