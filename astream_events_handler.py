import time
import logging
from typing import Optional, Dict, Any
import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from graph import app_runnable

logger = logging.getLogger(__name__)

class ToolExecutionTracker:
    """Clase para manejar el estado de ejecución de herramientas"""
    def __init__(self):
        self.tools = {}
        self.start_time = time.time()
        self.progress = 0.0

    def add_tool(self, tool_id: str, container: st.delta_generator.DeltaGenerator):
        """Registra una nueva herramienta en ejecución"""
        self.tools[tool_id] = {
            'container': container,
            'start_time': time.time(),
            'status': None,
            'output': None
        }
    
    def complete_tool(self, tool_id: str, output: str):
        """Marca una herramienta como completada"""
        if tool_id in self.tools:
            self.tools[tool_id]['execution_time'] = time.time() - self.tools[tool_id]['start_time']
            self.tools[tool_id]['output'] = output
            self.progress += 1.0 / len(self.tools)

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Trunca texto largo con puntos suspensivos"""
    return text[:max_length] + "..." if len(text) > max_length else text

async def invoke_our_graph(st_messages: list, st_placeholder: st.delta_generator.DeltaGenerator) -> str:
    """
    Maneja el flujo de eventos asíncrono con seguimiento detallado de herramientas
    
    Args:
        st_messages: Historial de mensajes del chat
        st_placeholder: Contenedor de Streamlit para mostrar actualizaciones
    
    Returns:
        Respuesta final del asistente
    """
    tracker = ToolExecutionTracker()
    final_text = ""
    progress_bar = st_placeholder.progress(0.0, text="Iniciando proceso...")
    
    try:
        async for event in app_runnable.astream_events({"messages": st_messages}, version="v2"):
            event_type = event["event"]
            
            # Actualizar barra de progreso
            tracker.progress = min(tracker.progress + 0.05, 0.95)
            progress_bar.progress(tracker.progress, text="Procesando eventos...")
            
            if event_type == "on_chat_model_stream":
                # Manejar streaming de tokens del modelo
                chunk = event["data"]["chunk"].content
                final_text += chunk
                with st_placeholder.container():
                    st.markdown(f"```markdown\n{final_text}\n```", unsafe_allow_html=True)
            
            elif event_type == "on_tool_start":
                # Configurar UI para nueva herramienta
                tool_id = event["run_id"]
                with st_placeholder.container():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        st.subheader(f"🛠️ {event['name']}")
                    with cols[1]:
                        with st.status(f"⚙️ Ejecutando {event['name']}...", expanded=True) as status:
                            st.write(f"**Input:**\n`{truncate_text(str(event['data'].get('input')), 300)}`")
                            output_placeholder = st.empty()
                    
                    tracker.add_tool(tool_id, output_placeholder)
            
            elif event_type == "on_tool_end":
                # Mostrar resultados de herramienta
                tool_id = event["run_id"]
                output = event["data"].get("output", "")
                error = event["data"].get("error")
                
                exec_time = time.time() - tracker.tools[tool_id]['start_time']
                output_text = f"⏱️ Tiempo ejecución: {exec_time:.1f}s\n\n"
                
                if error:
                    output_text += f"❌ **Error:** \n```\n{truncate_text(str(error), 500)}\n```"
                else:
                    output_text += f"📋 **Resultado:** \n```\n{truncate_text(str(output), 800)}\n```"
                
                tracker.tools[tool_id]['container'].markdown(output_text)
                tracker.complete_tool(tool_id, output)
                
                # Actualizar progreso final
                progress_bar.progress(
                    min(tracker.progress + 0.05, 1.0),
                    text=f"Herramienta {event['name']} completada"
                )
    
    except Exception as e:
        logger.error(f"Error en el flujo de eventos: {str(e)}", exc_info=True)
        with st_placeholder:
            st.error(f"🚨 Error crítico en el flujo: {str(e)}")
        return "Lo siento, hubo un error procesando tu solicitud"
    
    finally:
        progress_bar.empty()
    
    return final_text