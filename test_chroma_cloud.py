#!/usr/bin/env python3
"""
Debug script to test Chroma Cloud authentication with v2 API.
"""
import chromadb
from chromadb.config import Settings
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_API_URL = os.environ.get('CHROMA_API_URL')
CHROMA_TENANT_ID = os.environ.get('CHROMA_TENANT_ID')
CHROMA_DATABASE = os.environ.get('CHROMA_DATABASE')
CHROMA_API_KEY = os.environ.get('CHROMA_API_KEY')

print("Testing Chroma Cloud Authentication with v2 API")
print("=" * 50)
print(f"API URL: {CHROMA_API_URL}")
print(f"Tenant ID: {CHROMA_TENANT_ID}")
print(f"Database: {CHROMA_DATABASE}")
print(f"API Key: {CHROMA_API_KEY[:10]}..." if CHROMA_API_KEY else "None")
print()

# Test 1: Raw HTTP request to v2 version endpoint
print("Test 1: Raw HTTP request to v2 version endpoint")
try:
    response = requests.get(
        f"https://{CHROMA_API_URL}/api/v2/version",
        headers={'X-Chroma-Token': CHROMA_API_KEY}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 2: Try Authorization header with v2
print("Test 2: Raw HTTP request with Authorization header (v2)")
try:
    response = requests.get(
        f"https://{CHROMA_API_URL}/api/v2/version",
        headers={'Authorization': f'Bearer {CHROMA_API_KEY}'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 3: Check if tenant exists (v2)
print("Test 3: Check tenant endpoint (v2)")
try:
    response = requests.get(
        f"https://{CHROMA_API_URL}/api/v2/tenants/{CHROMA_TENANT_ID}",
        headers={'X-Chroma-Token': CHROMA_API_KEY}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 4: List tenants (v2)
print("Test 4: List all tenants (v2)")
try:
    response = requests.get(
        f"https://{CHROMA_API_URL}/api/v2/tenants",
        headers={'X-Chroma-Token': CHROMA_API_KEY}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 5: Try with CloudClient (should use v2 internally)
print("Test 5: CloudClient with your tenant/database")
try:
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT_ID,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY
    )
    version = client.get_version()
    print(f"✅ SUCCESS: Connected, version: {version}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 6: Try HttpClient with v2 endpoint
print("Test 6: HttpClient with v2 endpoint")
try:
    client = chromadb.HttpClient(
        host=f"https://{CHROMA_API_URL}/api/v2",
        tenant=CHROMA_TENANT_ID,
        database=CHROMA_DATABASE,
        headers={'X-Chroma-Token': CHROMA_API_KEY}
    )
    version = client.get_version()
    print(f"✅ SUCCESS: Connected, version: {version}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 7: Simple health check
print("Test 7: Health check endpoint")
try:
    response = requests.get(
        f"https://{CHROMA_API_URL}/api/v2/health",
        headers={'X-Chroma-Token': CHROMA_API_KEY}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()
print("Debug complete. Focus on v2 API endpoints.") 