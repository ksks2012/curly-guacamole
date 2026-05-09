"""
Test Notion token via GET /v1/users/me.

Usage:
    python testing/test_notion_token.py <token>
    # or set NOTION_TOKEN env var and run without argument
"""

import os
import sys
import requests

from rag.ingest.notion.client import NotionClient

def print_database_properties(token: str, database_id: str):
    url = f"https://api.notion.com/v1/databases/{database_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28"
    }

    res = requests.get(url, headers=headers)
    data = res.json()

    for prop_name, prop_info in data["properties"].items():
        print(prop_name, ":", prop_info["type"])
    

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NOTION_TOKEN", "")
    database_id = sys.argv[2] if len(sys.argv) > 2 else os.getenv("NOTION_DATABASE_ID", "")

    if not token:
        print("Usage: python testing/test_notion_token.py <token>")
        print("       or set NOTION_TOKEN environment variable")
        sys.exit(1)

    client = NotionClient(token)
    try:
        result = client.test_connection()
        print("Connection OK")
        print(f"  type : {result.get('type', '?')}")
        print(f"  id   : {result.get('id', '?')}")
        print(f"  name : {result.get('name', '?')}")
        bot = result.get("bot", {})
        owner = bot.get("owner", {})
        print(f"  owner: {owner.get('type', '?')}")
    except Exception as exc:
        print(f"Connection FAILED: {exc}")
        sys.exit(1)

    try:
        print("\nTesting database properties retrieval...")
        print_database_properties(token, database_id)
    except Exception as exc:
        print(f"Database properties retrieval FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
