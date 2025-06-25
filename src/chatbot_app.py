import streamlit as st
import importlib.util
import sys
import os
import ast

# Dynamically import lookup_rag_information
rag_tool_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Tools/RAG agents/rag_tool.py'))
spec_rag = importlib.util.spec_from_file_location("rag_tool", rag_tool_path)
if spec_rag is None or spec_rag.loader is None:
    raise ImportError(f"Could not load spec for rag_tool from {rag_tool_path}")
rag_module = importlib.util.module_from_spec(spec_rag)
sys.modules["rag_tool"] = rag_module
spec_rag.loader.exec_module(rag_module)
lookup_rag_information = rag_module.lookup_rag_information

# Dynamically import query_single_table_db
sql_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Tools/Sql agents/sql_agent.py'))
spec_sql = importlib.util.spec_from_file_location("sql_agent", sql_agent_path)
if spec_sql is None or spec_sql.loader is None:
    raise ImportError(f"Could not load spec for sql_agent from {sql_agent_path}")
sql_module = importlib.util.module_from_spec(spec_sql)
sys.modules["sql_agent"] = sql_module
spec_sql.loader.exec_module(sql_module)
query_single_table_db = sql_module.query_single_table_db



def run_rag_tool(query):
    # Dummy implementation, replace with your actual function
    return f"RAG Tool response to: {query}"

def run_sql_tool(query):
    # Dummy implementation, replace with your actual function
    return f"SQL Tool response to: {query}"

st.title("Data Analysis Chatbot")

# Sidebar for tool selection and clear chat
st.sidebar.header("Settings")
tool_choice = st.sidebar.radio(
    "Select Tool:",
    ("SQL Agent", "RAG Tool"),
    index=0
)
if st.sidebar.button("Clear Chat"):
    st.session_state.history = []
    st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

st.write("Ask me anything about your data!")

user_input = st.text_input("You:", key="input")

if user_input:
    with st.spinner("Generating response..."):
        if tool_choice == "RAG Tool":
            response = lookup_rag_information(user_input)
        else:
            response = query_single_table_db(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", response))

for speaker, message in st.session_state.history:
    if speaker == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(message)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Try to format SQL agent output as a table if possible
            formatted = False
            if tool_choice == "SQL Agent":
                try:
                    # Try to parse the message as a Python literal (list, tuple, etc.)
                    parsed = ast.literal_eval(str(message))
                    import pandas as pd
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # If it's a list of tuples, convert to DataFrame
                        if isinstance(parsed[0], (tuple, list)):
                            df = pd.DataFrame(parsed)
                            st.dataframe(df)
                            formatted = True
                        # If it's a list of dicts, convert to DataFrame
                        elif isinstance(parsed[0], dict):
                            df = pd.DataFrame(parsed)
                            st.dataframe(df)
                            formatted = True
                except Exception:
                    pass
            if not formatted:
                st.markdown(message) 