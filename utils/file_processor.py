import csv
import configparser
import json
import os
import pprint
import yaml

from typing import Mapping, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Reader
def read_yaml(filename: str) -> Mapping :
    data = {}
    try:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(e)
    pprint.pprint(data)
    return data

def read_json(filename: str) -> Mapping :
    data = {}
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)

    # pprint.pprint(data)
    return data

# Writer
def write_yaml(filename: str, data: Mapping):
    pass

def write_json(filename: str, data: Mapping):
    with open(filename, "w", encoding="utf8") as fw:
        jsonString = json.dumps(data, ensure_ascii=False)
        fw.writelines(jsonString)

def write_csv(filename: str, data: list):
    with open(f'{filename}.csv', 'w', newline='', encoding="utf8") as csvfile:
        if len(data) == 0:
            return
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def read_ini(filename: str) -> Mapping:
    config = configparser.ConfigParser()
    config.read(filename)

    return config


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Loads a PDF and splits it into chunks with enriched metadata.

    Each chunk carries:
      - source     : absolute path of the PDF (set by PyPDFLoader)
      - page        : 0-based page number (set by PyPDFLoader)
      - filename    : basename of the PDF file
      - total_pages : total number of pages in the PDF
      - chunk_id    : sequential index across all chunks (0-based)
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    total_pages = len(pages)
    filename = os.path.basename(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["filename"] = filename
        chunk.metadata["total_pages"] = total_pages
        chunk.metadata["chunk_id"] = idx

    return chunks