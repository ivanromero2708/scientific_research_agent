# agent_tools.py

from typing import Optional, List
from pydantic import Field, BaseModel, field_validator, ConfigDict
import pdfplumber
import urllib3
import time
import os
import io
import re
from dotenv import load_dotenv
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from langchain_core.tools import tool
from state import SearchPapersInput

load_dotenv()

class CoreAPIWrapper(BaseModel):
    """Wrapper para la API de CORE con manejo avanzado de errores."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    base_url: str = Field(
        default="https://api.core.ac.uk/v3",
        description="Endpoint principal de la API de CORE"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("CORE_API_KEY", ""),
        description="API Key para autenticación en CORE"
    )
    top_k_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número máximo de resultados a retornar"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        if not value:
            raise ValueError("CORE_API_KEY no encontrada en variables de entorno.")
        return value

    def _execute_api_request(self, query: str) -> dict:
        http = urllib3.PoolManager(
            retries=urllib3.Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504]
            )
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ScientificResearchAgent/2.0 (contact: tu@email.com)",
            "Accept": "application/json"
        }

        try:
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs",
                headers=headers,
                fields={
                    "q": query,
                    "limit": self.top_k_results,
                    "sort": "relevance:desc"
                },
                timeout=15.0
            )
            
            if response.status == 200:
                return response.json()
            elif response.status == 429:
                return {"error": "Límite de tasa excedido. Espere antes de hacer nuevas consultas."}
            else:
                return {"error": f"Error HTTP {response.status}"}
            
        except Exception as e:
            return {"error": f"Error de conexión: {str(e)}"}

    def _filter_relevant_results(self, results: list, query: str) -> list:
        keywords = [kw.lower() for kw in re.split(r'\W+', query) if kw]
        filtered = []
        for paper in results:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            if any(kw in title or kw in abstract for kw in keywords):
                filtered.append(paper)
        return filtered

    def search(self, query: str) -> dict:
        try:
            response = self._execute_api_request(query)
            
            if "error" in response:
                return {
                    "status": "error",
                    "error_type": "API Error",
                    "message": response["error"]
                }
            
            raw_results = response.get("results", [])
            filtered_results = self._filter_relevant_results(raw_results, query)
            
            if not filtered_results:
                return {
                    "status": "success",
                    "results_count": 0,
                    "papers": [],
                    "message": "No se encontraron resultados relevantes."
                }

            formatted_results = []
            for paper in filtered_results[:self.top_k_results]:
                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in paper.get("authors", [])
                ]
                
                formatted_results.append({
                    "title": paper.get("title", "Sin título"),
                    "id": paper.get("id"),
                    "publication_date": paper.get("publishedDate") or paper.get("yearPublished"),
                    "authors": authors[:5],
                    "urls": paper.get("sourceFulltextUrls", []),
                    "abstract": (paper.get("abstract") or "")[:500]
                })
            
            return {
                "status": "success",
                "results_count": len(formatted_results),
                "papers": formatted_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e)
            }

@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 3) -> dict:
    """
    Busca artículos científicos en la API de CORE según la consulta proporcionada.
    Ejemplo de entrada: {"query": "machine learning in healthcare", "max_papers": 5}
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return {
            "status": "error",
            "error_type": "Critical Error",
            "message": f"Fallo en search_papers: {str(e)}"
        }

@tool("download-paper")
def download_paper(url: str) -> dict:
    """
    Descarga y extrae texto de un documento científico en PDF, dado su URL.

    Args:
        url: URL válida de un documento PDF

    Retorna:
        dict: Resultado con estructura:
        {
            "status": "success"|"error",
            "url": str,
            "pages_processed": int,
            "content": str,
            "warnings": List[str]
        }
    """
    result_template = {
        "status": "success",
        "url": url,
        "pages_processed": 0,
        "content": "",
        "warnings": []
    }

    try:
        if not re.match(r'^https?://', url):
            raise ValueError("URL debe usar HTTP/HTTPS")
        
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            timeout=urllib3.Timeout(connect=15.0, read=30.0)
        )
        
        for attempt in range(3):
            try:
                response = http.request(
                    'GET',
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                    }
                )
                
                if response.status != 200:
                    raise ConnectionError(f"HTTP Error {response.status}")
                    
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower():
                    raise ValueError(f"Contenido no es PDF: {content_type}")
                    
                with pdfplumber.open(io.BytesIO(response.data)) as pdf:
                    result_template["pages_processed"] = len(pdf.pages)
                    text_content = []
                    
                    for page in pdf.pages[:50]:
                        text = page.extract_text() or ""
                        text_content.append(text)
                    
                    full_text = "\n".join(text_content)
                    result_template["content"] = full_text[:15000]
                    
                    if len(full_text) > 15000:
                        result_template["warnings"].append(
                            "Texto truncado a 15,000 caracteres"
                        )
                    
                    return result_template

            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
                
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "url": url
        }

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """
    Solicita retroalimentación/confirmación humana sobre la pregunta especificada.
    
    Args:
        question: Pregunta a mostrar al usuario
        
    Returns:
        str: Respuesta del usuario
    """
    ctx = get_script_run_ctx()
    
    if ctx and hasattr(st, 'session_state'):
        key = f"human_feedback_{hash(question)}"
        if key not in st.session_state:
            with st.chat_message("assistant"):
                st.markdown(f"**Confirmación Requerida:**\n{question}")
                st.session_state[key] = st.text_input("Tu respuesta:", key=key)
                
                if st.button("Enviar Respuesta", key=f"{key}_button"):
                    st.rerun()
            
            st.stop()
        
        return st.session_state.get(key, "")
    
    return input(f"\n[FEEDBACK REQUERIDO] {question}\nTu respuesta: ")

# Lista de herramientas disponibles para el agente
tools = [
    search_papers,
    download_paper,  # ¡Ahora está incluida!
    ask_human_feedback
]