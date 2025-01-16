from typing import ClassVar, Optional
from pydantic import Field, BaseModel
import pdfplumber
import urllib3
import time
import os
import io

from state import (
    SearchPapersInput,
)


from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class CoreAPIWrapper(BaseModel):
    """Simple wrapper around the CORE API."""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    api_key: ClassVar[str] = os.environ["CORE_API_KEY"]

    top_k_results: int = Field(description = "Top k results obtained by running a query on Core", default = 1)

    def _get_search_response(self, query: str) -> dict:
        http = urllib3.PoolManager()

        # Retry mechanism to handle transient errors
        max_retries = 5    
        for attempt in range(max_retries):
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs", 
                headers={"Authorization": f"Bearer {self.api_key}"}, 
                fields={"q": query, "limit": self.top_k_results}
            )
            if 200 <= response.status < 300:
                return response.json()
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx response from CORE API: {response.status} {response.data}")

    def search(self, query: str) -> str:
        response = self._get_search_response(query)
        results = response.get("results", [])
        if not results:
            return "No relevant results were found"

        # Format the results in a string
        docs = []
        for result in results:
            published_date_str = result.get('publishedDate') or result.get('yearPublished', '')
            authors_str = ' and '.join([item['name'] for item in result.get('authors', [])])
            docs.append((
                f"* ID: {result.get('id', '')},\n"
                f"* Title: {result.get('title', '')},\n"
                f"* Published Date: {published_date_str},\n"
                f"* Authors: {authors_str},\n"
                f"* Abstract: {result.get('abstract', '')},\n"
                f"* Paper URLs: {result.get('sourceFulltextUrls') or result.get('downloadUrl', '')}"
            ))
        return "\n-----\n".join(docs)



@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Search for scientific papers using the CORE API.

    Example:
    {"query": "Attention is all you need", "max_papers": 1}

    Returns:
        A list of the relevant papers found with the corresponding relevant information.
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"Error performing paper search: {e}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Download a specific scientific paper from a given URL.

    Example:
    {"url": "https://sample.pdf"}

    Returns:
        The paper content.
    """
    try:        
        http = urllib3.PoolManager(
            cert_reqs='CERT_NONE',
        )
        
        # Mock browser headers to avoid 403 error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        max_retries = 5
        for attempt in range(max_retries):
            response = http.request('GET', url, headers=headers)
            if 200 <= response.status < 300:
                pdf_file = io.BytesIO(response.data)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                raise Exception(f"Got non 2xx when downloading paper: {response.status_code} {response.text}")
    except Exception as e:
        return f"Error downloading paper: {e}"

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """Ask for human feedback. You should call this tool when encountering unexpected errors."""
    return input(question)

tools = [search_papers, download_paper, ask_human_feedback]