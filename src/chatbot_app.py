import streamlit as st
import importlib.util
import sys
import os
import ast
import pandas as pd
from typing import Any, List, Tuple, Optional
import traceback
import logging
from pandas import Index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolLoader:
    """Handles dynamic loading of external tools"""
    
    @staticmethod
    def load_module(module_name: str, file_path: str) -> Any:
        """
        Dynamically load a module from file path
        
        Args:
            module_name: Name to assign to the module
            file_path: Path to the Python file
            
        Returns:
            Loaded module
            
        Raises:
            ImportError: If module cannot be loaded
        """
        try:
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                raise ImportError(f"File not found: {abs_path}")
                
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {module_name} from {abs_path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            logger.info(f"Successfully loaded module: {module_name}")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {str(e)}")
            raise ImportError(f"Failed to load {module_name}: {str(e)}")

class DataAnalysisChatbot:
    """Main chatbot application class"""
    
    def __init__(self):
        self.setup_tools()
        self.setup_session_state()
    
    def setup_tools(self):
        """Initialize and load external tools"""
        try:
            # Load RAG tool
            rag_path = os.path.join(os.path.dirname(__file__), '../Tools/RAG agents/rag_tool.py')
            rag_module = ToolLoader.load_module("rag_tool", rag_path)
            lookup_rag = getattr(rag_module, 'lookup_rag_information', None)
            self.lookup_rag_information = lookup_rag if callable(lookup_rag) else self._fallback_rag_tool
            
            # Load SQL tool
            sql_path = os.path.join(os.path.dirname(__file__), '../Tools/Sql agents/sql_agent.py')
            sql_module = ToolLoader.load_module("sql_agent", sql_path)
            query_sql = getattr(sql_module, 'query_single_table_db', None)
            self.query_single_table_db = query_sql if callable(query_sql) else self._fallback_sql_tool
            
        except ImportError as e:
            logger.error(f"Tool loading failed: {str(e)}")
            st.error(f"Failed to load tools: {str(e)}")
            # Use fallback implementations
            self.lookup_rag_information = self._fallback_rag_tool
            self.query_single_table_db = self._fallback_sql_tool
    
    def _fallback_rag_tool(self, query: str) -> str:
        """Fallback implementation for RAG tool"""
        return f"RAG Tool (Fallback): {query}\n\nNote: External RAG tool unavailable."
    
    def _fallback_sql_tool(self, query: str) -> str:
        """Fallback implementation for SQL tool"""
        return f"SQL Tool (Fallback): {query}\n\nNote: External SQL tool unavailable."
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if "history" not in st.session_state:
            st.session_state.history = []
        if "tool_choice" not in st.session_state:
            st.session_state.tool_choice = "SQL Agent"
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("Settings")
        
        # Tool selection
        tool_choice = st.sidebar.radio(
            "Select Analysis Tool:",
            ("SQL Agent", "RAG Tool"),
            index=0 if st.session_state.tool_choice == "SQL Agent" else 1,
            help="Choose between SQL database queries or RAG document search"
        )
        st.session_state.tool_choice = tool_choice
        
        # Clear chat button
        if st.sidebar.button("Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        # Show chat history count
        if st.session_state.history:
            msg_count = len([msg for msg in st.session_state.history if msg[0] == "user"])
            st.sidebar.info(f"Messages: {msg_count}")
        
        # Tool info
        with st.sidebar.expander("Tool Information"):
            if tool_choice == "SQL Agent":
                st.write("**SQL Agent**: Query structured databases using natural language")
            else:
                st.write("**RAG Tool**: Search and retrieve information from documents")
        # Made by Devansh footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("<div style='text-align: center; color: gray;'>Conceptualized and developed by Devansh</div>", unsafe_allow_html=True)
    
    def process_query(self, query: str) -> Any:
        """
        Process user query with selected tool
        Args:
            query: User input query
        Returns:
            Tool response (str or (rows, columns))
        """
        try:
            if st.session_state.tool_choice == "RAG Tool":
                return self.lookup_rag_information(query)
            else:
                result = self.query_single_table_db(query)
                # If result is a tuple (rows, columns), return as is
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                # Otherwise, return as string (error)
                return str(result)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return f"Error processing query: {str(e)}\n\nPlease try a different query or check your tool configuration."
    
    def format_sql_response(self, response: Any) -> bool:
        """
        Try to format SQL response as a table with column names if available
        Args:
            response: SQL tool response (should be (rows, columns))
        Returns:
            True if formatted successfully, False otherwise
        """
        try:
            # If response is a tuple of (rows, columns)
            if isinstance(response, tuple) and len(response) == 2:
                rows, columns = response
                if isinstance(rows, list) and len(rows) > 0 and isinstance(columns, list):
                    # Ensure columns are strings and convert to Index
                    columns = Index([str(col) for col in columns])
                    df = pd.DataFrame(rows, columns=columns)
                    st.dataframe(df, use_container_width=True)
                    return True
            # Fallback: try to parse as Python literal
            if isinstance(response, str):
                parsed = ast.literal_eval(response)
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], (tuple, list)):
                        df = pd.DataFrame(parsed)
                        st.dataframe(df, use_container_width=True)
                        return True
                    elif isinstance(parsed[0], dict):
                        df = pd.DataFrame(parsed)
                        st.dataframe(df, use_container_width=True)
                        return True
                elif isinstance(parsed, pd.DataFrame):
                    st.dataframe(parsed, use_container_width=True)
                    return True
        except Exception as e:
            logger.debug(f"Table formatting failed: {str(e)}")
        return False
    
    def render_chat_message(self, speaker: str, message: str):
        """
        Render a single chat message
        
        Args:
            speaker: 'user' or 'assistant'
            message: Message content
        """
        if speaker == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**You:** {message}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Try to format SQL responses as tables
                formatted = False
                if st.session_state.tool_choice == "SQL Agent":
                    formatted = self.format_sql_response(message)
                
                if not formatted:
                    st.markdown(message)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("Data Analysis Chatbot")
        st.markdown("---")
        
        # Instructions
        with st.expander("How to use", expanded=False):
            st.markdown("""
            1. **Select a tool** from the sidebar (SQL Agent or RAG Tool)
            2. **Type your question** in the input field below
            3. **Press Enter** to get your answer
            
            **Tips:**
            - For SQL Agent: Ask about database queries, data analysis, statistics
            - For RAG Tool: Ask about document content, information retrieval
            """)
        
        # Display chat history
        for speaker, message in st.session_state.history:
            self.render_chat_message(speaker, message)
        
        # User input
        user_input = st.chat_input(
            f"Ask me anything using {st.session_state.tool_choice}...",
            key="user_input"
        )
        
        if user_input and user_input.strip():
            # Add user message to history
            st.session_state.history.append(("user", user_input))
            
            # Process query
            with st.spinner(f"Processing with {st.session_state.tool_choice}..."):
                response = self.process_query(user_input)
            
            # Add response to history
            st.session_state.history.append(("assistant", response))
            
            # Rerun to show new messages
            st.rerun()
    
    def run(self):
        """Main application entry point"""
        # Configure page
        st.set_page_config(
            page_title="Data Analysis Chatbot",
            # page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render components
        self.render_sidebar()
        self.render_chat_interface()

# Application entry point
if __name__ == "__main__":
    try:
        app = DataAnalysisChatbot()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        st.error("Please check your tool configurations and file paths.")
        logger.error(f"Application startup failed: {traceback.format_exc()}")