import os
from dotenv import load_dotenv
from pyprojroot import here
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

load_dotenv()

class SingleTableSQLAgent:
    def __init__(self, sqldb_directory: str, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0):
        self.sql_agent_llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        self.query_chain = create_sql_query_chain(self.sql_agent_llm, self.db)
        self.table_info = self.db.get_table_info()

    def query(self, question: str) -> str:
        try:
            raw_sql_query = self.query_chain.invoke({"question": question})
            sql_query = self._clean_sql_query(raw_sql_query)
            result = self.db.run(sql_query)
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def _clean_sql_query(self, raw_query: str) -> str:
        prefixes_to_remove = [
            "SQLQuery: ", "SQL Query: ", "Query: ", "SQL: ", "```sql\n", "```\n", "sql\n"
        ]
        cleaned_query = raw_query.strip()
        for prefix in prefixes_to_remove:
            if cleaned_query.startswith(prefix):
                cleaned_query = cleaned_query[len(prefix):].strip()
        if cleaned_query.endswith("```"):
            cleaned_query = cleaned_query[:-3].strip()
        return cleaned_query

    def get_sample_data(self, limit: int = 5) -> str:
        table_name = self.db.get_usable_table_names()[0]
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        return self.db.run(query)

    def get_column_info(self) -> str:
        table_name = self.db.get_usable_table_names()[0]
        query = f"PRAGMA table_info({table_name});"
        return self.db.run(query)

    def get_table_stats(self) -> dict:
        table_name = self.db.get_usable_table_names()[0]
        count_query = f"SELECT COUNT(*) as total_rows FROM {table_name};"
        row_count = self.db.run(count_query)
        return {
            "table_name": table_name,
            "total_rows": row_count,
            "column_info": self.get_column_info()
        }

def query_single_table_db(query: str) -> str:
    sqldb_directory = here("data/csv_sql.db")
    agent = SingleTableSQLAgent(
        sqldb_directory=sqldb_directory,
        llm_model="gpt-4o-mini",
        llm_temperature=0
    )
    return agent.query(query) 