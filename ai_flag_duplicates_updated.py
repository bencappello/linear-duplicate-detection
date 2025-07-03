import json
import openai
import os
import chromadb
from chromadb.utils import embedding_functions
import requests
import re
import argparse
from typing import List, Dict, Any, Optional

class Config:
    """
    Configuration class to hold all settings and secrets.
    """
    def __init__(self):
        self.openai_api_key = os.environ.get('CHROMA_OPENAI_API_KEY')
        self.linear_api_key = os.environ.get('LINEAR_OAUTH_ACCESS_TOKEN')
        self.linear_team_id = "d9c4b135-c5f6-4eec-a504-085f1c4e13dc" # Your Linear Team ID
        self.chroma_db_path = "/home/daytona/chroma-bug-bounty-demo"
        self.chroma_collection_name = "linear_bugs"
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4-1106-preview"
        self.issues_json_path = '/home/daytona/linear_issues.json'
        self.confidence_threshold = 65
        self.max_candidates = 10 # Number of similar tickets to fetch

        if not self.openai_api_key or not self.linear_api_key:
            raise ValueError("API keys for OpenAI and Linear must be set as environment variables.")

class LinearAPI:
    """
    Handles all interactions with the Linear GraphQL API.
    """
    def __init__(self, config: Config):
        self.config = config
        self.api_url = "https://api.linear.app/graphql"
        self.headers = {
            "Authorization": f"Bearer {self.config.linear_api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, query: str) -> Dict[str, Any]:
        """Helper to make requests to the Linear API."""
        try:
            response = requests.post(self.api_url, json={"query": query}, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Linear API: {e}")
            return {}

    def get_internal_id(self, identifier: str, all_issues: List[Dict[str, Any]]) -> Optional[str]:
        """Finds the internal Linear ID for a given public identifier."""
        for issue in all_issues:
            if issue.get('identifier') == identifier:
                return issue.get('id')
        return None

    def get_or_create_label_id(self, label_name: str) -> Optional[str]:
        """Gets the ID of a label by name, creating it if it doesn't exist."""
        query = f'''query {{ team(id: \"{self.config.linear_team_id}\") {{ labels {{ nodes {{ id name }} }} }} }}'''
        data = self._make_request(query)
        if data and 'data' in data and 'team' in data['data'] and data['data']['team']:
            for label in data['data']['team']['labels']['nodes']:
                if label['name'] == label_name:
                    return label['id']

        # If not found, create it
        mutation = f'''mutation {{ labelCreate(input: {{ name: \"{label_name}\", teamId: \"{self.config.linear_team_id}\" }}) {{ label {{ id }} }} }}'''
        data = self._make_request(mutation)
        if data and 'data' in data and 'labelCreate' in data['data']:
            return data['data']['labelCreate']['label']['id']
        return None

    def add_label_to_issue(self, issue_id: str, label_id: str):
        """Adds a label to an issue, preserving existing labels."""
        query = f'''query {{ issue(id: \"{issue_id}\") {{ labels {{ nodes {{ id }} }} }} }}'''
        data = self._make_request(query)
        if not (data and 'data' in data and 'issue' in data['data']):
            return

        current_label_ids = [label['id'] for label in data['data']['issue']['labels']['nodes']]
        if label_id not in current_label_ids:
            current_label_ids.append(label_id)
            mutation = f'''mutation {{ issueUpdate(id: \"{issue_id}\", input: {{ labelIds: {json.dumps(current_label_ids)} }}) {{ success }} }}'''
            self._make_request(mutation)
            print(f"Successfully added label to issue {issue_id}.")

    def add_comment_to_issue(self, issue_id: str, comment: str):
        """Adds a comment to a Linear issue."""
        safe_comment = json.dumps(comment)
        mutation = f'''mutation {{ commentCreate(input: {{ issueId: \"{issue_id}\", body: {safe_comment} }}) {{ success }} }}'''
        self._make_request(mutation)
        print(f"Successfully added comment to issue {issue_id}.")

    def link_issues(self, source_issue_id: str, related_issue_id: str):
        """Marks one issue as a duplicate of another."""
        mutation = f'''mutation {{ issueRelationCreate(input: {{ issueId: \"{source_issue_id}\", relatedIssueId: \"{related_issue_id}\", type: "duplicate" }}) {{ success }} }}'''
        self._make_request(mutation)
        print(f"Successfully linked {source_issue_id} as duplicate of {related_issue_id}.")


def load_all_issues(filepath: str) -> List[Dict[str, Any]]:
    """Loads all Linear issues from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading issues file: {e}")
        return []

def seed_chroma_with_linear_issues(config: Config, issues: List[Dict[str, Any]]):
    """
    Seeds the ChromaDB with all Linear issues. Run this once initially.
    """
    print("Seeding ChromaDB with Linear issues...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.openai_api_key,
        model_name=config.embedding_model
    )
    chroma_client = chromadb.PersistentClient(path=config.chroma_db_path)
    collection = chroma_client.get_or_create_collection(
        name=config.chroma_collection_name,
        embedding_function=openai_ef
    )

    documents = []
    metadatas = []
    ids = []
    for issue in issues:
        if issue.get('title') and issue.get('description'):
            documents.append(f"{issue['title']}\n{issue['description']}")
            metadatas.append({'linear_id': issue['identifier']})
            ids.append(issue['identifier'])

    # Add to collection in batches if necessary (for very large datasets)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully seeded ChromaDB with {len(ids)} issues.")


def find_duplicate_candidates(config: Config, ticket_text: str, ticket_id: str) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB to find the most similar tickets.
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.openai_api_key,
        model_name=config.embedding_model
    )
    chroma_client = chromadb.PersistentClient(path=config.chroma_db_path)
    collection = chroma_client.get_collection(name=config.chroma_collection_name, embedding_function=openai_ef)

    query_results = collection.query(
        query_texts=[ticket_text],
        n_results=config.max_candidates + 1, # +1 to account for self-matching
        include=["metadatas"]
    )

    candidates = []
    if not query_results['metadatas']:
        return []

    for meta in query_results['metadatas'][0]:
        match_id = meta.get('linear_id')
        if match_id and match_id != ticket_id:
            candidates.append({'id': match_id})
    return candidates[:config.max_candidates]


def analyze_duplicates_with_llm(config: Config, new_ticket: Dict[str, Any], candidates: List[Dict[str, Any]], all_issues_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uses an LLM to analyze candidate tickets and determine if they are duplicates.
    """
    if not candidates:
        return []

    candidate_details = ""
    for i, c in enumerate(candidates):
        issue_details = all_issues_map.get(c['id'], {})
        candidate_details += f"{i+1}. ID: {c['id']}\nTitle: {issue_details.get('title', 'N/A')}\nDescription: {issue_details.get('description', 'N/A')}\n"

    prompt = f"""
    You are an expert bug triage assistant. For each candidate ticket below, determine if it is a duplicate of the new ticket.
    Consider exact duplicates and pattern-based duplicates (e.g., same root cause, same API endpoint).

    For each candidate, reply on a single line in this format:
    [ID]: Yes, [confidence score as a percentage], [Detailed multi-sentence explanation]
    or
    [ID]: No, [confidence score as a percentage], [Short explanation]

    Example:
    BUG-123: Yes, 90, This is a duplicate because both tickets describe the same API contract violation and reference the same endpoint and error code.
    BUG-456: No, 10, This is not a duplicate because it concerns a different feature.

    Strictly follow this format. If you answer Yes, provide a detailed explanation.

    New Ticket:
    ID: {new_ticket['identifier']}
    Title: {new_ticket['title']}
    Description: {new_ticket['description']}

    Candidate Tickets:
    {candidate_details}

    For each candidate, is it a duplicate of the new ticket? Reply in the format above.
    """

    client = openai.OpenAI(api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    output = response.choices[0].message.content
    print('--- RAW LLM OUTPUT ---')
    print(output)
    print('--- END RAW LLM OUTPUT ---')

    duplicates = []
    for line in output.split('\n'):
        if not line.strip() or ':' not in line:
            continue
        try:
            candidate_id_part, rest_part = line.split(':', 1)
            candidate_id = candidate_id_part.strip()

            if re.search(r'yes', rest_part, re.IGNORECASE):
                parts = [p.strip() for p in rest_part.split(',', 2)]
                if len(parts) == 3:
                    confidence_str = re.search(r'\d+', parts[1])
                    confidence = int(confidence_str.group(0)) if confidence_str else 0
                    explanation = parts[2]
                    if confidence >= config.confidence_threshold:
                        duplicates.append({'id': candidate_id, 'confidence': confidence, 'explanation': explanation})
        except (ValueError, IndexError) as e:
            print(f"Could not parse LLM output line: '{line}'. Error: {e}")

    return duplicates


def process_ticket(ticket_id: str, config: Config, linear_api: LinearAPI, all_issues: List[Dict[str, Any]], all_issues_map: Dict[str, Any]):
    """
    Main processing logic for a single ticket.
    """
    print(f"\n--- Processing Ticket: {ticket_id} ---")
    new_ticket = all_issues_map.get(ticket_id)
    if not new_ticket:
        print(f"Could not find ticket {ticket_id} in the loaded issues.")
        return

    new_ticket_text = f"{new_ticket.get('title', '')}\n{new_ticket.get('description', '')}"
    if not new_ticket_text.strip():
        print(f"Ticket {ticket_id} has no title or description. Skipping.")
        return

    # 1. Find similar candidates from ChromaDB
    candidates = find_duplicate_candidates(config, new_ticket_text, ticket_id)
    print(f"Found {len(candidates)} potential duplicate candidates for {ticket_id}.")

    # 2. Use LLM to analyze candidates
    duplicates = analyze_duplicates_with_llm(config, new_ticket, candidates, all_issues_map)
    print(f"LLM identified {len(duplicates)} duplicates for {ticket_id} above the confidence threshold.")

    # 3. Take action in Linear
    new_ticket_internal_id = linear_api.get_internal_id(ticket_id, all_issues)
    if not new_ticket_internal_id:
        print(f"Could not find internal ID for {ticket_id}. Cannot update Linear.")
        return

    if duplicates:
        label_id = linear_api.get_or_create_label_id('ai-flagged-duplicate')
        if label_id:
            linear_api.add_label_to_issue(new_ticket_internal_id, label_id)

        for dup in duplicates:
            related_internal_id = linear_api.get_internal_id(dup['id'], all_issues)
            if related_internal_id:
                # Mark the new ticket as a duplicate of the older one
                linear_api.link_issues(new_ticket_internal_id, related_internal_id)
                comment = (
                    f"🤖 **AI Duplicate Detection**\n\n"
                    f"This ticket may be a duplicate of **{dup['id']}**.\n\n"
                    f"**Confidence Score:** {dup['confidence']}%\n\n"
                    f"**Reason:** {dup['explanation']}"
                )
                linear_api.add_comment_to_issue(new_ticket_internal_id, comment)
            else:
                print(f"Could not find internal ID for duplicate candidate {dup['id']}.")
    else:
        comment = "🤖 AI Duplicate Detection: No potential duplicates were found for this ticket."
        linear_api.add_comment_to_issue(new_ticket_internal_id, comment)
        print(f"No duplicates found. Added a comment to {ticket_id}.")


def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Find and flag duplicate tickets in Linear.")
    parser.add_argument('--seed', action='store_true', help="Seed the ChromaDB with issues from the JSON file.")
    parser.add_argument('--tickets', nargs='+', help="A list of ticket identifiers to process (e.g., BUG-123 BUG-456).")
    parser.add_argument('--all', action='store_true', help="Process all tickets from the JSON file.")

    args = parser.parse_args()

    config = Config()
    all_issues = load_all_issues(config.issues_json_path)
    if not all_issues:
        print("Exiting due to issues with loading data.")
        return

    if args.seed:
        seed_chroma_with_linear_issues(config, all_issues)
        return

    linear_api = LinearAPI(config)
    all_issues_map = {issue['identifier']: issue for issue in all_issues}

    tickets_to_process = []
    if args.tickets:
        tickets_to_process = args.tickets
    elif args.all:
        tickets_to_process = [issue['identifier'] for issue in all_issues]

    if not tickets_to_process:
        print("No tickets specified. Use --tickets, --all, or --seed.")
        parser.print_help()
        return

    for ticket_id in tickets_to_process:
        process_ticket(ticket_id, config, linear_api, all_issues, all_issues_map)


if __name__ == "__main__":
    main()
