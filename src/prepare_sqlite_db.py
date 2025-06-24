import os
import sqlite3
import pandas as pd
from ruamel.yaml import YAML
from pyprojroot import here
import re

CONFIG_PATH = here("configs/tools_config.yml")

# Load config
def load_config():
    yaml = YAML(typ="safe")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f)
    return config

def clean_table_name(filename):
    """Clean filename to create valid SQLite table name"""
    # Remove file extension
    name = os.path.splitext(filename)[0]
    # Replace spaces, parentheses, and other special chars with underscores
    name = re.sub(r'[^\w]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = 'table_' + name
    return name or 'unnamed_table'

def make_unique_columns(columns):
    """Make column names unique by appending numbers to duplicates"""
    seen = {}
    unique_columns = []
    
    for col in columns:
        # Clean column name for SQLite compatibility
        clean_col = str(col).strip()
        if not clean_col:
            clean_col = 'unnamed_column'
        
        # Replace problematic characters
        clean_col = re.sub(r'[^\w]', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        
        if not clean_col:
            clean_col = 'unnamed_column'
        
        # Handle duplicates
        if clean_col.lower() not in seen:
            seen[clean_col.lower()] = 0
            unique_columns.append(clean_col)
        else:
            seen[clean_col.lower()] += 1
            unique_columns.append(f"{clean_col}_{seen[clean_col.lower()]}")
    
    return unique_columns

def csvs_to_sqlite(csv_dir, db_path):
    csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    # Remove existing database to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    created_tables = []
    
    for csv_file in csv_files:
        table_name = clean_table_name(csv_file)
        csv_path = os.path.join(csv_dir, csv_file)
        print(f"Processing {csv_file} -> table '{table_name}'")
        
        try:
            # Read CSV with mixed types handling
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Get original column names
            orig_cols = list(df.columns)
            print(f"  Original columns: {len(orig_cols)} columns")
            
            # Check for duplicates
            if len(orig_cols) != len(set(str(col).lower() for col in orig_cols)):
                print(f"  Warning: Duplicate columns detected in {csv_file}")
                # Make columns unique
                unique_cols = make_unique_columns(orig_cols)
                df.columns = unique_cols
                print(f"  Renamed columns to ensure uniqueness")
            else:
                # Still clean column names even if no duplicates
                clean_cols = make_unique_columns(orig_cols)
                df.columns = clean_cols
            
            print(f"  Final columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"  Final columns: {list(df.columns)}")
            
            # Convert DataFrame to SQL
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            created_tables.append(table_name)
            print(f"  Successfully created table '{table_name}' with {len(df)} rows")
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {str(e)}")
            continue
    
    conn.close()
    print(f"\nSummary: Created {len(created_tables)} tables in {db_path}:")
    for t in created_tables:
        print(f"  - {t}")

def main():
    config = load_config()
    csv_dir = here(config["csv_rag"]["unstructured_docs"])
    # Use a .db file in the same directory as the vectordb, or default
    vectordb_dir = here(config["csv_rag"]["vectordb"])
    db_path = os.path.join(os.path.dirname(vectordb_dir), "csv_sql.db")
    
    print(f"Converting CSVs in {csv_dir} to SQLite DB at {db_path}")
    csvs_to_sqlite(csv_dir, db_path)

if __name__ == "__main__":
    main()