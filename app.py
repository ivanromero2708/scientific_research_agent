import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

from astream_events_handler import invoke_our_graph   # Utility function to handle events from astream_events from graph

load_dotenv()

st.title("Agent Researcher 🤝 LangGraph")
st.markdown("#### Chat Streaming and Tool Calling using Astream Events for Researching with the CORE API")

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

# Check if the OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    st.sidebar.header("OPENAI_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

# st write magic
with st.expander(label="Asynchronous Research Assistant with LangGraph's Astream Events", expanded=st.session_state.expander_open):
    """
    This example demonstrates how to build an interactive research assistant powered by [_LangGraph_](https://langchain-ai.github.io/langgraph/)
    and its asynchronous [`astream_events (v2)`](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/).
    The implementation avoids using callbacks or external libraries, making it fully compatible with Streamlit.

    Key features include:
    - `on_llm_new_token`: Streams tokens from the ChatLLM model as they are generated.
    - `on_tool_start`: Triggers during the initialization of any tool call, supporting multiple invocations.
    - `on_tool_end`: Captures and displays the final result of tool calls for seamless interaction.
    """

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="What is going to be your research question?")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a placeholder container for streaming and any other events to visually render here
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))
