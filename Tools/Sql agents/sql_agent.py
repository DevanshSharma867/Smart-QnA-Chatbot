import os
from dotenv import load_dotenv
from pyprojroot import here
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from sqlalchemy import text
import logging
import re

load_dotenv()

class SingleTableSQLAgent:
    def __init__(self, sqldb_directory: str, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0):
        self.sql_agent_llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        self.query_chain = create_sql_query_chain(self.sql_agent_llm, self.db)
        self.table_info = self.db.get_table_info()

    def query(self, question: str):
        """
        Execute a natural language query against the database and return (rows, columns).
        Returns:
            tuple: (rows, columns) where rows is a list of tuples and columns is a list of column names
        """
        try:
            raw_sql_query = self.query_chain.invoke({"question": question})
            sql_query = self._clean_sql_query(raw_sql_query)
            # Use SQLAlchemy engine to get both rows and column names
            with self.db._engine.connect() as conn:
                result_proxy = conn.execute(text(sql_query))
                rows = result_proxy.fetchall()
                columns = list(result_proxy.keys())
            return rows, columns
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _clean_sql_query(self, raw_query: str) -> str:
        prefixes_pattern = r'^(?:[`\n\s]*)?(SQLQuery:|SQL Query:|Query:|SQL:|```sql|```|sql)([\s\n]*)'
        cleaned_query = raw_query.strip()
        logging.info(f"Raw SQL query before cleaning: {repr(raw_query)}")
        # Remove code block markers and all known prefixes iteratively
        while True:
            new_query = re.sub(prefixes_pattern, '', cleaned_query, flags=re.IGNORECASE)
            if new_query == cleaned_query:
                break
            cleaned_query = new_query.strip()
        # Remove trailing triple backticks if present
        if cleaned_query.endswith("```"):
            cleaned_query = cleaned_query[:-3].strip()
        logging.info(f"Cleaned SQL query: {repr(cleaned_query)}")
        return cleaned_query

    def get_sample_data(self, limit: int = 20):
        table_name = list(self.db.get_usable_table_names())[0]
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        with self.db._engine.connect() as conn:
            result_proxy = conn.execute(text(query))
            rows = result_proxy.fetchall()
            columns = list(result_proxy.keys())
        return rows, columns

    def get_column_info(self):
        table_name = list(self.db.get_usable_table_names())[0]
        query = f"PRAGMA table_info({table_name});"
        with self.db._engine.connect() as conn:
            result_proxy = conn.execute(text(query))
            rows = result_proxy.fetchall()
            columns = list(result_proxy.keys())
        return rows, columns

    def get_table_stats(self) -> dict:
        table_name = list(self.db.get_usable_table_names())[0]
        count_query = f"SELECT COUNT(*) as total_rows FROM {table_name};"
        with self.db._engine.connect() as conn:
            result_proxy = conn.execute(text(count_query))
            row_count = result_proxy.fetchall()
            columns = list(result_proxy.keys())
        return {
            "table_name": table_name,
            "total_rows": row_count,
            "column_info": self.get_column_info()
        }

def query_single_table_db(query: str):
    """
    Query the single table database and return (rows, columns).
    Returns:
        tuple: (rows, columns) where rows is a list of tuples and columns is a list of column names
    """
    sqldb_directory = str(here("data/csv_sql.db"))
    agent = SingleTableSQLAgent(
        sqldb_directory=sqldb_directory,
        llm_model="gpt-4o-mini",
        llm_temperature=0
    )
    return agent.query(query) 