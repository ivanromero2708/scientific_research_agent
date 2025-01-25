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

# Import específico de Streamlit si se necesitan funciones de feedback
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from langchain_core.tools import tool

# Import de la clase de validación para el input de busqueda
from state import SearchPapersInput

# Carga variables de entorno (CORE_API_KEY, etc.)
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
        """Valida que la API Key esté configurada."""
        if not value:
            raise ValueError("CORE_API_KEY no encontrada en variables de entorno.")
        return value

    def _execute_api_request(self, query: str) -> dict:
        """Ejecuta una solicitud a la API de CORE con reintentos y control de errores."""
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
                method='GET',
                url=f"{self.base_url}/search/outputs",
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
            else:
                return {"error": f"Error HTTP {response.status}"}
            
        except Exception as e:
            return {"error": f"Error de conexión: {str(e)}"}

    def search(self, query: str) -> dict:
        """
        Realiza una búsqueda académica estructurada en CORE.

        Retorna un dict con:
          - status ("success" o "error")
          - results_count (número de papers)
          - papers (lista de papers con título, autores, etc.)
        """
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
                    "authors": authors[:5],  # Limita la lista de autores
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

    Retorna un dict con:
      - status ("success" o "error")
      - pages_processed (número de páginas procesadas)
      - content (texto extraído hasta 15,000 caracteres)
      - warnings (si se trunca texto)
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
        
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            timeout=urllib3.Timeout(connect=15.0, read=30.0)
        )
        
        # Reintentos básicos de descarga
        for attempt in range(3):
            try:
                response = http.request(
                    method='GET',
                    url=url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                    }
                )
                
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
                    
                    # Extraer texto de las primeras 50 páginas
                    for page in pdf.pages[:50]:
                        text = page.extract_text() or ""
                        text_content.append(text)
                    
                    full_text = "\n".join(text_content)
                    # Limitar a 15,000 caracteres
                    result_template["content"] = full_text[:15000]
                    
                    if len(full_text) > 15000:
                        result_template["warnings"].append(
                            "Texto truncado a 15,000 caracteres"
                        )
                    
                    return result_template

            except Exception as e:
                # Si es el último intento, relanzar la excepción
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))  # Exponencial backoff
                
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
    - Modo Streamlit: se muestra una entrada de texto para el usuario.
    - Modo CLI/Consola: usa `input()`.

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
                # Forzamos un rerun para que se actualice la UI inmediatamente
                st.rerun()
        return st.session_state.get(key, "")
    
    # Modo CLI/Consola
    return input(f"\n[FEEDBACK REQUERIDO] {question}\nTu respuesta: ")


# Lista de herramientas disponibles para el agente.
tools = [
    search_papers,
    download_paper,
    ask_human_feedback
]
