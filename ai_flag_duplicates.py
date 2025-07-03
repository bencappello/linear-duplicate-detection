import json
import openai
import os
import chromadb
from chromadb.utils import embedding_functions
import requests
import re

def extract_bug_number(identifier):
    match = re.search(r'BUG-(\d+)', identifier)
    return int(match.group(1)) if match else None

# Load all issues and select BUG-74 as the new ticket
test_id = 'BUG-74'
with open('/home/daytona/linear_issues.json', 'r') as f:
    issues = json.load(f)
id_to_issue = {issue['identifier']: issue for issue in issues}
new_ticket = id_to_issue[test_id]
new_ticket_text = f"{new_ticket['title']}\n{new_ticket['description']}"

openai_api_key = os.environ.get('CHROMA_OPENAI_API_KEY')
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="/home/daytona/chroma-bug-bounty-demo")
collection = chroma_client.get_or_create_collection(
    name="linear_bugs",
    embedding_function=openai_ef
)

query_results = collection.query(
    query_texts=[new_ticket_text],
    n_results=21,
    include=["documents", "metadatas"]
)
candidates = []
for i in range(len(query_results['documents'][0])):
    meta = query_results['metadatas'][0][i]
    match_id = meta['linear_id'] if meta else 'Unknown'
    if match_id != test_id:
        candidate_issue = id_to_issue.get(match_id, {})
        candidates.append({
            'id': match_id,
            'title': candidate_issue.get('title', ''),
            'description': candidate_issue.get('description', '')
        })
    if len(candidates) == 20:
        break

# Improved prompt for verbose reasoning
prompt = f"""
You are an expert bug triage assistant. For each candidate ticket below, determine if it is a duplicate of the new ticket. Consider both exact and pattern-based duplicates (e.g., same root cause, same API contract violation, same HTTP method/status code issue, etc).

For each candidate, reply on a single line in this format:
[ID]: Yes, [confidence score], [Detailed multi-sentence explanation]
or
[ID]: No, [confidence score], [Short explanation]

Example:
BUG-123: Yes, 90, This is a duplicate because both tickets describe the same API contract violation. Both reference the same endpoint and error code, and the steps to reproduce are nearly identical. The only difference is the resource ID, but the underlying bug is the same.
BUG-456: No, 0, This is not a duplicate because it is about a different endpoint.

Strictly follow this format for every candidate. Do not use brackets or any other punctuation. If you answer Yes, provide a detailed, multi-sentence explanation referencing specific fields, patterns, or context that justify the duplicate status.

New Ticket:
ID: {test_id}
Title: {new_ticket['title']}
Description: {new_ticket['description']}

Candidate Tickets:
"""
for i, c in enumerate(candidates):
    prompt += f"{i+1}. ID: {c['id']}\nTitle: {c['title']}\nDescription: {c['description']}\n"
prompt += """
For each candidate, is it a duplicate of the new ticket? Reply in the format above.
"""

client = openai.OpenAI(api_key=openai_api_key)
response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1200
)
output = response.choices[0].message.content

print('--- RAW LLM OUTPUT FOR BUG-74 ---')
print(output)
print('--- END RAW LLM OUTPUT ---')

duplicates = []
new_bug_number = extract_bug_number(test_id)
for line in output.split('\n'):
    if not line.strip() or ':' not in line:
        continue
    parts = line.split(':', 1)
    candidate_id = parts[0].strip().replace('ID', '').replace('.', '').strip()
    rest = parts[1].strip()
    yes_match = re.search(r'yes', rest, re.IGNORECASE)
    conf = None
    if yes_match:
        after_yes = rest[yes_match.end():]
        conf_match = re.search(r'(\d{1,3})', after_yes)
        if conf_match:
            conf = int(conf_match.group(1))
    candidate_bug_number = extract_bug_number(candidate_id)
    if (yes_match and conf is not None and conf > 65 and
        candidate_bug_number is not None and candidate_bug_number < new_bug_number):
        explanation = rest.split(',', 2)[-1].strip(' ,') if ',' in rest else ''
        duplicates.append({'id': candidate_id, 'confidence': conf, 'explanation': explanation})

LINEAR_API_URL = "https://api.linear.app/graphql"
access_token = os.environ.get('LINEAR_OAUTH_ACCESS_TOKEN')
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

def get_internal_id(identifier):
    for issue in issues:
        if issue['identifier'] == identifier:
            return issue['id']
    return None

def get_current_label_ids(issue_id):
    query = f'''query {{ issue(id: \"{issue_id}\") {{ labels {{ nodes {{ id name }} }} }} }}'''
    resp = requests.post(LINEAR_API_URL, json={"query": query}, headers=headers)
    return [label['id'] for label in resp.json()['data']['issue']['labels']['nodes']]

def get_label_id(label_name):
    query = '''query { team(id: \"d9c4b135-c5f6-4eec-a504-085f1c4e13dc\") { labels { nodes { id name } } } }'''
    resp = requests.post(LINEAR_API_URL, json={"query": query}, headers=headers)
    for label in resp.json()['data']['team']['labels']['nodes']:
        if label['name'] == label_name:
            return label['id']
    mutation = f'''mutation {{ labelCreate(input: {{ name: \"{label_name}\", teamId: \"d9c4b135-c5f6-4eec-a504-085f1c4e13dc\" }}) {{ label {{ id }} }} }}'''
    resp = requests.post(LINEAR_API_URL, json={"query": mutation}, headers=headers)
    return resp.json()['data']['labelCreate']['label']['id']

def add_label_preserve(issue_id, label_name):
    current_label_ids = get_current_label_ids(issue_id)
    label_id = get_label_id(label_name)
    if label_id not in current_label_ids:
        current_label_ids.append(label_id)
    mutation = f'''mutation {{ issueUpdate(id: \"{issue_id}\", input: {{ labelIds: {json.dumps(current_label_ids)} }}) {{ success }} }}'''
    requests.post(LINEAR_API_URL, json={"query": mutation}, headers=headers)

def add_comment(issue_id, comment):
    safe_comment = comment.replace('"', '\\"').replace('\n', '\\n')
    mutation = f'''mutation {{ commentCreate(input: {{ issueId: \"{issue_id}\", body: \"{safe_comment}\" }}) {{ success }} }}'''
    requests.post(LINEAR_API_URL, json={"query": mutation}, headers=headers)

def link_issues(issue_id, related_id):
    mutation = f'''mutation {{ issueRelationCreate(input: {{ issueId: \"{issue_id}\", relatedIssueId: \"{related_id}\", type: related }}) {{ success }} }}'''
    requests.post(LINEAR_API_URL, json={"query": mutation}, headers=headers)

bug74_internal_id = get_internal_id('BUG-74')

if duplicates:
    add_label_preserve(bug74_internal_id, 'ai-flagged-duplicate')
    for dup in duplicates:
        related_internal_id = get_internal_id(dup['id'])
        if related_internal_id:
            link_issues(bug74_internal_id, related_internal_id)
        comment = (
            f"🤖 AI Duplicate Detection\n\n"
            f"This ticket may be a duplicate of {dup['id']}\n\n"
            f"Similarity Score: {dup['confidence']}%\n\n"
            f"Reason: {dup['explanation']}"
        )
        add_comment(bug74_internal_id, comment)
    print(f"Added label (preserved others), linked, and commented for {len(duplicates)} duplicate(s) on BUG-74.")
else:
    add_comment(bug74_internal_id, "No potential duplicates were found for this ticket by the AI system.")
    print("No duplicates found. Comment added to BUG-74.")
