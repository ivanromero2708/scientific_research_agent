from typing import Optional, List
from pydantic import Field, BaseModel, field_validator, ConfigDict
import pdfplumber
import urllib3
import time
import os
import io
import re
from streamlit.runtime.scriptrunner import get_script_run_ctx

from state import SearchPapersInput
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class CoreAPIWrapper(BaseModel):
    """Wrapper mejorado para la API de CORE con manejo avanzado de errores"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    base_url: str = Field(
        default="https://api.core.ac.uk/v3",
        description="Endpoint principal de la API de CORE"
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("CORE_API_KEY"),
        description="API Key para autenticación en CORE"
    )
    top_k_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número máximo de resultados a retornar"
    )

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Valida que la API Key esté configurada"""
        if not value:
            raise ValueError("CORE_API_KEY no encontrada en variables de entorno")
        return value

    def _execute_api_request(self, query: str) -> dict:
        """Ejecuta una solicitud a la API con reintentos inteligentes"""
        http = urllib3.PoolManager(
            retries=urllib3.Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ScientificResearchAgent/2.0 (+https://github.com/ivanromero2708/scientific_research_agent)"
        }

        try:
            response = http.request(
                'GET',
                f"{self.base_url}/search/outputs",
                headers=headers,
                fields={
                    "q": query,
                    "limit": self.top_k_results,
                    "sort": "publishedDate:desc"
                },
                timeout=10.0
            )
            
            if response.status == 200:
                return response.json()
            
            return {"error": f"Error HTTP {response.status}"}
            
        except Exception as e:
            return {"error": f"Error de conexión: {str(e)}"}

    def search(self, query: str) -> dict:
        """Realiza una búsqueda académica estructurada"""
        try:
            response = self._execute_api_request(query)
            
            if "error" in response:
                return {
                    "status": "error",
                    "error_type": "API Error",
                    "message": response["error"]
                }
            
            results = response.get("results", [])
            if not results:
                return {
                    "status": "success",
                    "results_count": 0,
                    "papers": []
                }

            formatted_results = []
            for paper in results[:self.top_k_results]:
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
    Busca artículos científicos en CORE API. 
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
    Descarga y extrae texto de documentos científicos en PDF.
    Ejemplo de entrada: {"url": "https://example.com/paper.pdf"}
    """
    result_template = {
        "status": "success",
        "url": url,
        "pages_processed": 0,
        "content": "",
        "warnings": []
    }

    try:
        # Validación de URL
        if not re.match(r'^https?://', url):
            raise ValueError("URL debe usar HTTP/HTTPS")
            
        # Configuración de conexión segura
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            timeout=urllib3.Timeout(connect=15.0, read=30.0)
        )
        
        # Descarga con reintentos
        for attempt in range(3):
            try:
                response = http.request('GET', url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                if response.status != 200:
                    raise ConnectionError(f"HTTP Error {response.status}")
                    
                # Verificar tipo de contenido
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower():
                    raise ValueError(f"Contenido no es PDF: {content_type}")
                    
                # Procesar PDF
                with pdfplumber.open(io.BytesIO(response.data)) as pdf:
                    result_template["pages_processed"] = len(pdf.pages)
                    text_content = []
                    
                    for page in pdf.pages[:50]:  # Limitar a 50 páginas
                        text = page.extract_text() or ""
                        text_content.append(text)
                    
                    full_text = "\n".join(text_content)
                    result_template["content"] = full_text[:15000]  # Limitar tamaño
                    
                    if len(full_text) > 15000:
                        result_template["warnings"].append("Texto truncado a 15,000 caracteres")
                    
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
    Solicita feedback humano. Útil para confirmaciones o ambigüedades.
    Ejemplo: {"question": "¿Debería priorizar resultados recientes o históricos?"}
    """
    ctx = get_script_run_ctx()
    
    # Modo Streamlit
    if ctx and hasattr(st, 'session_state'):
        key = f"human_feedback_{hash(question)}"
        if key not in st.session_state:
            with st.chat_message("assistant"):
                st.markdown(f"**Asistente requiere confirmación:**\n{question}")
                st.session_state[key] = st.text_input("Tu respuesta:", key=key)
                st.rerun()
        return st.session_state.get(key, "")
        
    # Modo CLI/Consola
    return input(f"\n[FEEDBACK REQUERIDO] {question}\nTu respuesta: ")

# Lista de herramientas disponibles para el agente
tools = [search_papers, download_paper, ask_human_feedback]

