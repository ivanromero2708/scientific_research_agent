# app.py

import os
import asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from astream_events_handler import execute_research_flow
from contextlib import contextmanager
from datetime import datetime
import re
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Configuración inicial de la página
st.set_page_config(
    page_title="Investigador Científico IA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@contextmanager
def handle_async_errors():
    """Maneja errores asíncronos y muestra mensajes en la UI."""
    try:
        yield
    except RuntimeError as e:
        st.error(f"🔁 Límite de iteraciones alcanzado: {str(e)}")
        st.session_state.processing = False
        st.rerun()
    except asyncio.CancelledError:
        st.warning("⏹️ Investigación detenida por el usuario")
        st.session_state.processing = False
        st.rerun()
    except Exception as e:
        st.error(f"🚨 Error crítico: {str(e)}")
        st.session_state.processing = False
        st.rerun()

def setup_api_key():
    """Configuración segura de API Keys."""
    st.sidebar.header("🔑 Configuración de API Keys")
    
    with st.sidebar.expander("⚙️ Configurar claves", expanded=True):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Obtén tu clave en https://platform.openai.com/account/api-keys"
        )
        
        core_key = st.text_input(
            "CORE API Key",
            type="password",
            value=os.getenv("CORE_API_KEY", ""),
            help="Regístrate en https://core.ac.uk/services/api/"
        )
        
        if st.button("💾 Guardar Claves", type="primary"):
            if not openai_key:
                st.error("¡Clave OpenAI requerida!")
                return
            if not core_key:
                st.error("¡Clave CORE requerida!")
                return
                
            # Guardar en variables de entorno y sesión
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["CORE_API_KEY"] = core_key
            st.session_state.api_keys_set = True
            st.success("✅ Claves configuradas correctamente")
            st.rerun()
        
        # Botón para detener la investigación
        if st.sidebar.button("⏹️ Detener Investigación"):
            # Opcional: si guardas un supervisor en session_state
            if 'research_supervisor' in st.session_state:
                st.session_state.research_supervisor.cancel_research()
            st.rerun()

def initialize_chat():
    """Inicializa el estado del chat con al menos un mensaje del asistente."""
    defaults = {
        "processing": False,
        "requires_research": False,
        "num_feedback_requests": 0,
        "is_good_answer": True,
        "research_cycles": 0,
        "created_at": datetime.now(),
        "last_updated": datetime.now()
    }
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="¡Hola! Soy tu asistente de investigación. ¿En qué tema deseas profundizar hoy?")
        ]
        st.session_state.update(defaults)
    elif not st.session_state.messages:
        st.session_state.messages = [
            AIMessage(content="¡La conversación se reinició! ¿Sobre qué deseas investigar?")
        ]
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

def render_chat_history():
    """Renderiza el historial del chat con formato básico."""
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg.content)

def show_welcome_expander():
    """Muestra el panel de bienvenida inicial."""
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
    """Reinicia la conversación con al menos un mensaje del asistente."""
    st.session_state.messages = [
        AIMessage(content="¡Conversación reiniciada! ¿En qué tema deseas profundizar ahora?")
    ]
    st.session_state.processing = False
    st.rerun()

def show_tool_monitoring():
    """Muestra el panel de herramientas ejecutadas."""
    if "tool_executions" not in st.session_state:
        st.session_state.tool_executions = []
    
    with st.expander("🔍 **Supervisión de Herramientas Ejecutadas**", expanded=True):
        if not st.session_state.tool_executions:
            st.info("No se han ejecutado herramientas aún")
            return
            
        for idx, tool in enumerate(st.session_state.tool_executions, 1):
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown(f"**Herramienta #{idx}**")
                    st.caption(f"🕒 {tool['execution_time']:.1f}s")
                    st.write(f"🔧 **Tipo:** {tool['name']}")
                    status_emoji = "✅" if tool['status'] == "success" else "❌"
                    st.write(f"{status_emoji} **Estado:** {tool['status']}")
                
                with cols[1]:
                    with st.expander("📥 **Input**", expanded=False):
                        st.json(tool["input"], expanded=False)
                    
                    if tool["output"]:
                        with st.expander("📤 **Output**", expanded=False):
                            if isinstance(tool["output"], dict):
                                st.json(tool["output"])
                            else:
                                st.code(str(tool["output"]), language="text")
                    else:
                        st.warning("Sin output generado")

def main():
    st.title("🔍 Investigador Científico Asistido por IA")
    
    # Configura la sección lateral para ingresar las claves
    setup_api_key()
    
    # Verifica que las claves estén configuradas
    if not st.session_state.get("api_keys_set", False):
        st.info("⚠️ Configura tus API Keys en la barra lateral")
        return

    # Muestra panel de herramientas ejecutadas
    show_tool_monitoring()
    # Inicializa la sesión de chat
    initialize_chat()
    # Muestra la sección de bienvenida
    show_welcome_expander()
    # Renderiza el historial del chat
    render_chat_history()
    
    # Entrada de texto del usuario
    if prompt := st.chat_input("Escribe tu pregunta de investigación..."):
        if st.session_state.processing:
            st.warning("Espera a que termine la operación actual")
            return

        st.session_state.processing = True
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            
            with handle_async_errors(), st.spinner("🔍 Analizando consulta..."):
                try:
                    # Llamada asíncrona para ejecutar el flujo de investigación
                    response = asyncio.run(
                        execute_research_flow(
                            st.session_state.messages,
                            placeholder
                        )
                    )
                    
                    if response:
                        # Limpieza de formato (quitamos ```markdown si viene en la respuesta)
                        clean_response = re.sub(r"```markdown\n|\n```", "", response)
                        final_message = AIMessage(content=clean_response)
                        st.session_state.messages.append(final_message)
                finally:
                    st.session_state.processing = False

if __name__ == "__main__":
    main()
