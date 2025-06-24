import os
import yaml
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import tiktoken


class PrepareVectorDB:
    """
    An optimized class to prepare and manage a Vector Database (VectorDB) using CSV documents.
    
    Features:
    - Parallel file processing with ThreadPoolExecutor
    - Efficient token-based text splitting
    - Batch processing for embeddings with proper token limit handling
    - File change detection using content hashing
    - Progress tracking and performance profiling
    - Memory-efficient batch persistence
    
    Attributes:
        doc_dir (str): Path to the directory containing CSV documents to be processed.
        chunk_size (int): The maximum size of each chunk (in tokens) for text splitting.
        chunk_overlap (int): The number of overlapping tokens between consecutive chunks.
        embedding_model (str): The name of the embedding model for generating vector representations.
        vectordb_dir (str): Directory where the resulting vector database will be stored.
        collection_name (str): The name of the collection within the vector database.
        batch_size (int): Number of documents to process in each embedding batch.
        max_workers (int): Maximum number of threads for parallel processing.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int = 800,
                 chunk_overlap: int = 80,
                 embedding_model: str = "text-embedding-3-small",
                 vectordb_dir: str = "vectordb",
                 collection_name: str = "default",
                 batch_size: int = 256,
                 max_workers: int = 4
                 ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Cache file for tracking processed files
        self.cache_file = os.path.join(here(self.vectordb_dir), "processed_files.cache")

    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content for change detection."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")
            return ""
        return hash_sha256.hexdigest()

    def load_file_cache(self) -> dict:
        """Load cache of previously processed files."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return eval(f.read())  # Simple dict storage
            except:
                pass
        return {}

    def save_file_cache(self, cache: dict):
        """Save cache of processed files."""
        try:
            with open(self.cache_file, 'w') as f:
                f.write(str(cache))
        except Exception as e:
            print(f"Error saving cache: {e}")

    def path_maker(self, file_name: str, doc_dir: str) -> str:
        """Creates a full file path by joining the given directory and file name."""
        return os.path.join(here(doc_dir), file_name)

    def load_single_csv(self, file_path: str) -> Tuple[str, List[Document]]:
        """Load a single CSV file and return filename with documents."""
        try:
            csv_loader = CSVLoader(file_path=file_path)
            docs = csv_loader.load()
            filename = os.path.basename(file_path)
            print(f"Loaded {len(docs)} documents from {filename}")
            return filename, docs
        except Exception as e:
            filename = os.path.basename(file_path)
            print(f"Error loading {filename}: {e}")
            return filename, []

    def load_csvs_parallel(self, file_paths: List[str]) -> List[Document]:
        """Load multiple CSV files in parallel using ThreadPoolExecutor."""
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file loading tasks
            future_to_file = {
                executor.submit(self.load_single_csv, fp): fp 
                for fp in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                filename, docs = future.result()
                all_docs.extend(docs)
        
        return all_docs

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def validate_chunk_size(self, chunk: str) -> bool:
        """Validate that chunk doesn't exceed embedding model token limits."""
        token_count = self.count_tokens(chunk)
        # OpenAI embedding models typically have 8192 token limit
        max_tokens = 8000  # Leave some buffer
        
        if token_count > max_tokens:
            print(f"Warning: Chunk has {token_count} tokens (max: {max_tokens})")
            return False
        return True

    def split_documents_efficiently(self, docs: List[Document]) -> List[Document]:
        """Split documents using token-based splitting with validation."""
        print(f"Splitting {len(docs)} documents with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Use TokenTextSplitter for more accurate token-based splitting
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name="cl100k_base"  # Same encoding as OpenAI models
        )
        
        start_time = time.time()
        doc_splits = text_splitter.split_documents(docs)
        
        # Validate chunks don't exceed token limits
        valid_splits = []
        for chunk in doc_splits:
            if self.validate_chunk_size(chunk.page_content):
                valid_splits.append(chunk)
        
        split_time = time.time() - start_time
        print(f"Text splitting completed in {split_time:.2f}s")
        print(f"Created {len(valid_splits)} valid chunks (filtered {len(doc_splits) - len(valid_splits)} oversized)")
        
        return valid_splits

    def create_embeddings_batch(self, doc_splits: List[Document]) -> Chroma:
        """Create embeddings with optimized batch processing."""
        print(f"Creating embeddings with batch_size={self.batch_size}")
        
        # Initialize embedding model with optimized batch size
        embedding_model = OpenAIEmbeddings(
            model=self.embedding_model,
            chunk_size=self.batch_size,  # Process in batches
            max_retries=3  # Add retry logic
        )
        
        start_time = time.time()
        
        # Create vector database with batch processing
        vectordb = Chroma.from_documents(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding=embedding_model,
            persist_directory=str(here(self.vectordb_dir))
        )
        
        embedding_time = time.time() - start_time
        print(f"Embedding creation completed in {embedding_time:.2f}s")
        
        return vectordb

    def process_in_memory_batches(self, doc_splits: List[Document], batch_size: int = 500) -> Chroma:
        """Process documents in memory-efficient batches to avoid overwhelming the system."""
        print(f"Processing {len(doc_splits)} documents in batches of {batch_size}")
        
        embedding_model = OpenAIEmbeddings(
            model=self.embedding_model,
            chunk_size=self.batch_size
        )
        
        vectordb = None
        total_batches = (len(doc_splits) + batch_size - 1) // batch_size
        
        for i in range(0, len(doc_splits), batch_size):
            batch_num = (i // batch_size) + 1
            batch = doc_splits[i:i + batch_size]
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            start_time = time.time()
            
            if vectordb is None:
                # Create initial vector database
                vectordb = Chroma.from_documents(
                    documents=batch,
                    collection_name=self.collection_name,
                    embedding=embedding_model,
                    persist_directory=str(here(self.vectordb_dir))
                )
            else:
                # Add to existing vector database
                vectordb.add_documents(batch)
            
            batch_time = time.time() - start_time
            print(f"Batch {batch_num} completed in {batch_time:.2f}s")
        
        if vectordb is None:
            raise RuntimeError("No documents were processed; vectordb is None.")
        return vectordb

    def cleanup_old_collections(self):
        """Clean up old collections to free up space."""
        try:
            # This is a placeholder - implement based on your Chroma setup
            print("Cleaning up old collections...")
        except Exception as e:
            print(f"Warning: Could not clean up old collections: {e}")

    def run(self):
        """
        Execute the optimized vector database preparation process.
        
        Features:
        - Parallel file loading
        - File change detection with caching
        - Efficient token-based text splitting
        - Batch processing for embeddings
        - Performance profiling
        """
        total_start_time = time.time()
        
        if not os.path.exists(here(self.vectordb_dir)):
            os.makedirs(here(self.vectordb_dir))
            print(f"Directory '{self.vectordb_dir}' was created.")
        
        # Load file cache
        file_cache = self.load_file_cache()
        
        # Get CSV files
        csv_files = [f for f in os.listdir(here(self.doc_dir)) if f.lower().endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files to process.")
        
        if not csv_files:
            print("No CSV files found in the specified directory.")
            return
        
        # Check which files need processing
        files_to_process = []
        for csv_file in csv_files:
            file_path = self.path_maker(csv_file, self.doc_dir)
            current_hash = self.get_file_hash(file_path)
            
            if csv_file not in file_cache or file_cache[csv_file] != current_hash:
                files_to_process.append(file_path)
                file_cache[csv_file] = current_hash
            else:
                print(f"Skipping {csv_file} (unchanged)")
        
        if not files_to_process:
            print("All files are up to date. No processing needed.")
            return
        
        print(f"Processing {len(files_to_process)} new/changed files...")
        
        # Clean up old collections
        self.cleanup_old_collections()
        
        # Step 1: Load CSV files in parallel
        load_start = time.time()
        docs = self.load_csvs_parallel(files_to_process)
        load_time = time.time() - load_start
        print(f"File loading completed in {load_time:.2f}s")
        print(f"Total documents loaded: {len(docs)}")
        
        if not docs:
            print("No documents were loaded successfully.")
            return
        
        # Step 2: Split documents efficiently
        doc_splits = self.split_documents_efficiently(docs)
        
        if not doc_splits:
            print("No valid document chunks were created.")
            return
        
        # Step 3: Create embeddings with batch processing
        if len(doc_splits) > 1000:
            # Use memory-efficient batch processing for large datasets
            vectordb = self.process_in_memory_batches(doc_splits, batch_size=500)
        else:
            # Use standard batch processing for smaller datasets
            vectordb = self.create_embeddings_batch(doc_splits)
        
        # Save updated file cache
        self.save_file_cache(file_cache)
        
        # Final statistics
        total_time = time.time() - total_start_time
        print(f"\n VectorDB creation completed!")
        print(f"Total vectors in database: {vectordb._collection.count()}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per document: {total_time/len(docs):.3f}s")


if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPEN_AI_API_KEY environment variable is not set.")
    os.environ['OPENAI_API_KEY'] = openai_api_key

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    # # Uncomment the following configs to run for swiss airline policy document
    chunk_size = app_config["csv_rag"]["chunk_size"]
    chunk_overlap = app_config["csv_rag"]["chunk_overlap"]
    embedding_model = app_config["csv_rag"]["embedding_model"]
    vectordb_dir = app_config["csv_rag"]["vectordb"]
    collection_name = app_config["csv_rag"]["collection_name"]
    doc_dir = app_config["csv_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name,
        batch_size=256,  # Optimized batch size
        max_workers=4    # Parallel processing
    )

    prepare_db_instance.run()

    # Uncomment the following configs to run for stories document
    chunk_size = app_config["json_rag"]["chunk_size"]
    chunk_overlap = app_config["json_rag"]["chunk_overlap"]
    embedding_model = app_config["json_rag"]["embedding_model"]
    vectordb_dir = app_config["json_rag"]["vectordb"]
    collection_name = app_config["json_rag"]["collection_name"]
    doc_dir = app_config["json_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name,
        batch_size=512,  # Optimized batch size
        max_workers=4    # Parallel processing
    )

    prepare_db_instance.run()