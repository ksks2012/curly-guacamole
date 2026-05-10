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

def retrieve_database(token: str, database_id: str):
    # ./data/database.json
    url = f"https://api.notion.com/v1/databases/{database_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2026-03-11",
        "Content-Type": "application/json"
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()

def query_data_source(token: str, data_source_id: str):
    # ./data/data_source.json
    url = f"https://api.notion.com/v1/data_sources/{data_source_id}/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2026-03-11",
        "Content-Type": "application/json"
    }
    payload = {}
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()

def retrieve_page_as_markdown(token: str, page_id: str):
    # ./data/page_markdown.json
    url = f"https://api.notion.com/v1/pages/{page_id}/markdown"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2026-03-11"
    }

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.text

def retrieve_block_children(token: str, block_id: str):
    # page_id also works for block_id since pages are blocks in Notion's data model
    # ./data/block_children.json
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2026-03-11"
    }

    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()

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

    # obtain data_source_id for query test
    try:
        print("\nTesting database properties retrieval...")
        database_data = retrieve_database(token, database_id)
        print(f"Retrieved database properties: {database_data}")
    except Exception as exc:
        print(f"Database properties retrieval FAILED: {exc}")
        sys.exit(1)

    # obtain page_id for markdown retrieval test
    try:
        print("\nTesting data source query...")
        data_source_id = sys.argv[3] if len(sys.argv) > 3 else os.getenv("NOTION_DATA_SOURCE_ID", "")
        if not data_source_id:
            print("Usage: python testing/test_notion_token.py <token> <database_id> <data_source_id>")
            print("       or set NOTION_DATA_SOURCE_ID environment variable")
            sys.exit(1)
        data_source_data = query_data_source(token, data_source_id)
        print(f"Retrieved data from data source: {data_source_data}")
    except Exception as exc:
        print(f"Data source query FAILED: {exc}")
        sys.exit(1)

    # obtain page's markdown content
    try:
        print("\nTesting page markdown retrieval...")
        page_id = sys.argv[4] if len(sys.argv) > 4 else os.getenv("NOTION_PAGE_ID", "")
        if not page_id:
            print("Usage: python testing/test_notion_token.py <token> <database_id> <data_source_id> <page_id>")
            print("       or set NOTION_PAGE_ID environment variable")
            sys.exit(1)
        markdown = retrieve_page_as_markdown(token, page_id)
        print(f"Retrieved markdown content for page {page_id}:\n{markdown[:500]}...")  # print first 500 chars
    except Exception as exc:
        print(f"Page markdown retrieval FAILED: {exc}")
        sys.exit(1)

    try:
        print("\nTesting block children retrieval...")
        block_id = sys.argv[5] if len(sys.argv) > 5 else os.getenv("NOTION_BLOCK_ID", "")
        if not block_id:
            print("Usage: python testing/test_notion_token.py <token> <database_id> <data_source_id> <page_id> <block_id>")
            print("       or set NOTION_BLOCK_ID environment variable")
            sys.exit(1)
        block_children = retrieve_block_children(token, block_id)
        print(f"Retrieved children for block {block_id}: {block_children}")
    except Exception as exc:
        print(f"Block children retrieval FAILED: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
