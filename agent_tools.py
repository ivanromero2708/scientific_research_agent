from typing import ClassVar, Optional, List
from pydantic import Field, BaseModel, field_validator
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
    """Wrapper para la API de CORE con manejo mejorado de errores"""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    api_key: ClassVar[str] = Field(default=os.getenv("CORE_API_KEY"), description="API key para CORE")
    top_k_results: int = Field(description="Número máximo de resultados a obtener", default=1, ge=1, le=10)

    @field_validator('api_key')
    def check_api_key(cls, value):
        if not value:
            raise ValueError("CORE_API_KEY no está configurada en las variables de entorno")
        return value

    def _get_search_response(self, query: str) -> dict:
        """Realiza una búsqueda con reintentos y manejo de errores"""
        http = urllib3.PoolManager()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ResearchAgent/1.0 (+https://github.com/ivanromero2708/scientific_research_agent)"
        }

        for attempt in range(5):
            try:
                response = http.request(
                    'GET',
                    f"{self.base_url}/search/outputs",
                    headers=headers,
                    fields={"q": query, "limit": self.top_k_results},
                    timeout=10.0
                )
                
                if response.status == 200:
                    return response.json()
                elif 400 <= response.status < 500:
                    return {"error": f"Error del cliente: {response.status}"}
                else:
                    raise urllib3.exceptions.HTTPError(f"Estado HTTP inesperado: {response.status}")
                    
            except urllib3.exceptions.HTTPError as e:
                if attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"Fallo al conectar con CORE API después de 5 intentos: {str(e)}")

    def search(self, query: str) -> str:
        """Procesa y formatea los resultados de la búsqueda"""
        try:
            response = self._get_search_response(query)
            if "error" in response:
                return response["error"]
            
            results = response.get("results", [])
            if not results:
                return "No se encontraron resultados relevantes"

            formatted_results = []
            for result in results[:self.top_k_results]:
                authors = '; '.join([f"{a.get('name', '')}" for a in result.get("authors", [])][:5])
                abstract = result.get("abstract", "")[:500] + "..." if len(result.get("abstract", "")) > 500 else result.get("abstract", "")
                
                formatted_results.append(
                    "\n".join([
                        f"📄 **Título**: {result.get('title', 'Desconocido')}",
                        f"🆔 ID: `{result.get('id', '')}`",
                        f"📅 Fecha: {result.get('publishedDate', result.get('yearPublished', 'N/D'))}",
                        f"👥 Autores: {authors}",
                        f"🔗 Enlaces: {', '.join(result.get('sourceFulltextUrls', []))}",
                        f"📝 Resumen: {abstract}\n"
                    ])
                )
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"🚨 Error en la búsqueda: {str(e)}"

@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Busca artículos científicos usando la API de CORE. Ejemplo: {"query": "machine learning", "max_papers": 3}"""
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"🚨 Error crítico en search_papers: {str(e)}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Descarga y extrae texto de un PDF científico. Ejemplo: {"url": "https://ejemplo.com/paper.pdf"}"""
    try:
        # Validación de URL
        if not re.match(r'^https?://', url):
            return "❌ URL inválida: debe comenzar con http:// o https://"
            
        # Configuración de la solicitud
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            timeout=urllib3.Timeout(connect=10.0, read=30.0)
        )
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf, text/html'
        }

        # Descarga con reintentos
        for attempt in range(3):
            try:
                response = http.request('GET', url, headers=headers)
                
                if response.status != 200:
                    return f"❌ Error HTTP {response.status} al descargar el PDF"
                    
                # Verificar tipo de contenido
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower():
                    return f"❌ El contenido no es PDF (Content-Type: {content_type})"
                    
                # Procesar PDF
                with pdfplumber.open(io.BytesIO(response.data)) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages[:50]])  # Limitar a 50 páginas
                    return text[:15000] + "..." if len(text) > 15000 else text  # Limitar tamaño
                    
            except Exception as e:
                if attempt == 2:
                    return f"❌ Error al procesar PDF: {str(e)}"
                time.sleep(1)
                
    except Exception as e:
        return f"🚨 Error crítico en download_paper: {str(e)}"

@tool("ask-human-feedback")
def ask_human_feedback(question: str) -> str:
    """Solicita feedback humano. Úsalo cuando encuentres problemas ambiguos o necesites confirmación."""
    ctx = get_script_run_ctx()
    
    # Modo Streamlit
    if ctx:
        key = f"human_feedback_{hash(question)}"
        if key not in st.session_state:
            st.session_state[key] = st.text_input(question, key=key)
            st.rerun()
        return st.session_state.get(key, "")
        
    # Modo CLI
    return input(f"[HUMANO] {question}: ")

tools = [search_papers, download_paper, ask_human_feedback]