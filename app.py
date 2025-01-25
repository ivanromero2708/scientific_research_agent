import os
import asyncio
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from astream_events_handler import invoke_our_graph
from contextlib import contextmanager

load_dotenv()

# Configuración inicial de la página
st.set_page_config(
    page_title="Investigador Científico IA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@contextmanager
def handle_async_errors():
    """Maneja errores asíncronos y muestra mensajes en la UI"""
    try:
        yield
    except Exception as e:
        st.error(f"🚨 Error crítico: {str(e)}")
        st.session_state.processing = False
        st.rerun()

def setup_api_key():
    """Configuración segura de la API Key"""
    st.sidebar.header("Configuración de API Keys")
    
    if 'api_keys_set' not in st.session_state:
        st.session_state.api_keys_set = False
        
    with st.sidebar.expander("🔑 Configurar claves API", expanded=not st.session_state.api_keys_set):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Obtén tu clave en https://platform.openai.com/api-keys"
        )
        
        core_key = st.text_input(
            "CORE API Key", 
            type="password",
            help="Regístrate en https://core.ac.uk/services/api/"
        )
        
        if st.button("💾 Guardar Claves"):
            if openai_key and core_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["CORE_API_KEY"] = core_key
                st.session_state.api_keys_set = True
                st.rerun()
            else:
                st.warning("¡Ambas claves son requeridas!")

def initialize_chat():
    """Inicializa el estado del chat"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="¡Hola! Soy tu asistente de investigación. ¿En qué tema deseas profundizar hoy?")
        ]
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

def render_chat_history():
    """Renderiza el historial del chat con formato mejorado"""
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg.content)

def show_welcome_expander():
    """Muestra el panel de bienvenida inicial"""
    if "expander_open" not in st.session_state:
        st.session_state.expander_open = True

    if st.session_state.expander_open:
        with st.expander("🚀 Bienvenido al Investigador Científico IA", expanded=True):
            st.markdown("""
            ### Características Principales:
            - **Búsqueda Inteligente**: Acceso a millones de artículos científicos mediante la API de CORE
            - **Análisis de PDF**: Extracción y análisis de contenido de documentos científicos
            - **Colaboración Humano-AI**: Integración fluida para feedback y validaciones

            ### 📚 Flujo de Trabajo:
            1. Ingresa tu pregunta de investigación
            2. El agente planifica y ejecuta búsquedas
            3. Revisa los resultados y provee feedback
            4. Genera reportes ejecutivos automáticamente

            *¡Comienza escribiendo tu pregunta en el chat!*
            """)

def clear_conversation():
    """Reinicia la conversación"""
    st.session_state.messages = [
        AIMessage(content="¡Conversación reiniciada! ¿En qué tema deseas profundizar ahora?")
    ]
    st.session_state.processing = False
    st.rerun()

# UI Principal
def main():
    st.title("🔍 Investigador Científico Asistido por IA")
    
    setup_api_key()
    
    if not st.session_state.get("api_keys_set", False):
        st.info("⚠️ Por favor configura tus API Keys en la barra lateral para comenzar")
        return

    initialize_chat()
    
    # Barra de herramientas lateral
    st.sidebar.button("🧹 Limpiar Conversación", on_click=clear_conversation)
    
    # Panel de bienvenida
    show_welcome_expander()
    
    # Historial del chat
    render_chat_history()

    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta de investigación..."):
        if st.session_state.processing:
            st.warning("Por favor espera a que termine la operación actual")
            return

        st.session_state.processing = True
        st.session_state.expander_open = False
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            
            with handle_async_errors(), st.spinner("🔍 Analizando tu consulta..."):
                try:
                    response = asyncio.run(invoke_our_graph(
                        st.session_state.messages,
                        placeholder
                    ))
                    
                    if response:
                        st.session_state.messages.append(AIMessage(content=response))
                        st.session_state.processing = False
                except Exception as e:
                    st.error(f"Error en el flujo de trabajo: {str(e)}")
                finally:
                    st.session_state.processing = False

if __name__ == "__main__":
    main()