from typing import List, Optional, Literal, Annotated
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    ConfigDict,
    field_validator
)
from datetime import datetime
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage
)

class AgentState(BaseModel):
    """
    Estado principal del agente durante el proceso de investigación científica.
    
    Attributes:
        requires_research: Indica si la consulta requiere investigación adicional
        num_feedback_requests: Número de veces que se ha solicitado feedback
        is_good_answer: Indica si la respuesta actual es satisfactoria
        messages: Historial completo de mensajes de la conversación
        created_at: Timestamp de creación del estado
        last_updated: Timestamp de última actualización
        research_cycles: Número de ciclos de investigación completados
    """
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "requires_research": True,
                    "num_feedback_requests": 0,
                    "is_good_answer": False,
                    "messages": [AIMessage(content="¿En qué tema deseas investigar?")],
                    "research_cycles": 1
                }
            ]
        }
    )

    requires_research: bool = Field(
        default=False,
        description="Indica si la consulta requiere investigación con herramientas externas"
    )
    
    num_feedback_requests: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Número de veces que se ha solicitado feedback humano",
        json_schema_extra={"example": 0}
    )
    
    is_good_answer: bool = Field(
        default=False,
        description="Indica si la respuesta actual cumple con los criterios de calidad"
    )
    
    messages: Annotated[
        List[BaseMessage],
        Field(
            min_length=1,
            description="Historial de mensajes de la conversación",
            json_schema_extra={"format": "message-list"}
        )
    ]
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp de creación del estado"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp de última actualización"
    )
    
    research_cycles: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Número de ciclos completados de búsqueda e investigación"
    )

    # Validadores
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Valida que la lista de mensajes tenga una estructura válida"""
        if not messages:
            raise ValueError("El estado debe contener al menos un mensaje")
        
        if not isinstance(messages[-1], (AIMessage, HumanMessage)):
            raise ValueError("El último mensaje debe ser del usuario o asistente")
            
        return messages

    @model_validator(mode='after')
    def validate_state_consistency(self) -> 'AgentState':
        """Valida consistencia lógica del estado"""
        if self.is_good_answer and self.requires_research:
            raise ValueError("Estado inválido: respuesta marcada como buena pero requiere investigación")
            
        if self.num_feedback_requests > 0 and not self.requires_research:
            raise ValueError("Estado inválido: feedback solicitado sin investigación requerida")
            
        return self

    # Métodos de utilidad
    def add_message(self, message: BaseMessage) -> None:
        """Añade un nuevo mensaje al estado y actualiza timestamps"""
        self.messages.append(message)
        self.last_updated = datetime.utcnow()
        
        if isinstance(message, ToolMessage):
            self.research_cycles += 1

    def reset_research_state(self) -> None:
        """Reinicia el estado relacionado con la investigación"""
        self.requires_research = False
        self.num_feedback_requests = 0
        self.research_cycles = 0
        self.last_updated = datetime.utcnow()

class SearchPapersInput(BaseModel):
    """Input validado para búsquedas en CORE API"""
    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Consulta de búsqueda para papers científicos",
        json_schema_extra={"example": "machine learning in drug discovery"}
    )
    
    max_papers: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número máximo de papers a retornar",
        json_schema_extra={"example": 3}
    )

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, value: str) -> str:
        """Sanitiza la consulta de búsqueda"""
        cleaned = value.strip().replace('"', '')
        if len(cleaned) < 3:
            raise ValueError("La consulta debe tener al menos 3 caracteres")
        return cleaned

class DecisionMakingOutput(BaseModel):
    """Output estructurado del nodo de toma de decisiones inicial"""
    requires_research: bool = Field(
        ...,
        description="Indica si la consulta requiere investigación adicional"
    )
    
    answer: Optional[str] = Field(
        default=None,
        min_length=10,
        max_length=2000,
        description="Respuesta directa si no se requiere investigación",
        json_schema_extra={"example": "La teoría de la relatividad fue desarrollada por Albert Einstein en 1905."}
    )

    @model_validator(mode='after')
    def validate_answer_presence(self) -> 'DecisionMakingOutput':
        if not self.requires_research and not self.answer:
            raise ValueError("Se requiere una respuesta cuando no hay necesidad de investigación")
        return self

class JudgeOutput(BaseModel):
    """Output estructurado del nodo de evaluación de calidad"""
    is_good_answer: bool = Field(
        ...,
        description="Indica si la respuesta cumple con los criterios de calidad"
    )
    
    feedback: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=1000,
        description="Feedback detallado para mejorar la respuesta",
        json_schema_extra={"example": "Faltan citar fuentes recientes (últimos 5 años)"}
    )

    @model_validator(mode='after')
    def validate_feedback_presence(self) -> 'JudgeOutput':
        if not self.is_good_answer and not self.feedback:
            raise ValueError("Se requiere feedback cuando la respuesta no es satisfactoria")
        return self

# Tipos especializados para el grafo
AgentState.update_forward_refs()
SearchStep = Literal["decision_making", "planning", "tools", "agent", "judge"]
ValidationResult = Annotated[dict, Field(description="Resultado de validación del estado")]