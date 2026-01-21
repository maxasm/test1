"""
ChromaDB Interaction Logger

A wrapper that prints all ChromaDB interactions for debugging and monitoring.
Usage: Replace chromadb.HttpClient with LoggingChromaClient in your code.
"""

import functools
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb.config import Settings


def _log_interaction(method_name: str, args: tuple, kwargs: dict, result: Any = None, error: Exception = None):
    """Print a formatted log of the ChromaDB interaction."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    print(f"\n{'='*60}")
    print(f"CHROMA DB LOG [{timestamp}] - {method_name}")
    print(f"{'='*60}")
    
    if args:
        print(f"  Args: {args}")
    if kwargs:
        print(f"  Kwargs: {kwargs}")
    
    if error:
        print(f"  ERROR: {type(error).__name__}: {error}")
    elif result is not None:
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "... (truncated)"
        print(f"  Result: {result_str}")
    
    print(f"{'='*60}\n")


class LoggingCollection:
    """Wrapper around ChromaDB Collection that logs all interactions."""
    
    def __init__(self, collection):
        self._collection = collection
    
    def _wrap_method(self, method_name: str):
        """Create a wrapper for a collection method that logs calls."""
        original_method = getattr(self._collection, method_name)
        
        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            try:
                result = original_method(*args, **kwargs)
                _log_interaction(f"Collection.{method_name}", args, kwargs, result=result)
                return result
            except Exception as e:
                _log_interaction(f"Collection.{method_name}", args, kwargs, error=e)
                raise
        
        return wrapper
    
    def __getattr__(self, name: str):
        attr = getattr(self._collection, name)
        if callable(attr):
            return self._wrap_method(name)
        return attr
    
    # Explicitly wrap common methods for better IDE support
    def add(self, *args, **kwargs):
        return self._wrap_method("add")(*args, **kwargs)
    
    def query(self, *args, **kwargs):
        return self._wrap_method("query")(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return self._wrap_method("get")(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        return self._wrap_method("update")(*args, **kwargs)
    
    def upsert(self, *args, **kwargs):
        return self._wrap_method("upsert")(*args, **kwargs)
    
    def delete(self, *args, **kwargs):
        return self._wrap_method("delete")(*args, **kwargs)
    
    def count(self, *args, **kwargs):
        return self._wrap_method("count")(*args, **kwargs)
    
    def peek(self, *args, **kwargs):
        return self._wrap_method("peek")(*args, **kwargs)


class LoggingChromaClient:
    """Wrapper around ChromaDB HttpClient that logs all interactions."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        settings: Optional[Settings] = None,
    ):
        _log_interaction("HttpClient.__init__", (), {"host": host, "port": port})
        self._client = chromadb.HttpClient(host=host, port=port, settings=settings)
    
    def heartbeat(self) -> int:
        result = self._client.heartbeat()
        _log_interaction("Client.heartbeat", (), {}, result=result)
        return result
    
    def list_collections(self):
        result = self._client.list_collections()
        _log_interaction("Client.list_collections", (), {}, result=result)
        return result
    
    def get_collection(self, name: str, **kwargs) -> LoggingCollection:
        result = self._client.get_collection(name, **kwargs)
        _log_interaction("Client.get_collection", (name,), kwargs, result=f"<Collection: {name}>")
        return LoggingCollection(result)
    
    def create_collection(self, name: str, **kwargs) -> LoggingCollection:
        result = self._client.create_collection(name, **kwargs)
        _log_interaction("Client.create_collection", (name,), kwargs, result=f"<Collection: {name}>")
        return LoggingCollection(result)
    
    def get_or_create_collection(self, name: str, **kwargs) -> LoggingCollection:
        result = self._client.get_or_create_collection(name, **kwargs)
        _log_interaction("Client.get_or_create_collection", (name,), kwargs, result=f"<Collection: {name}>")
        return LoggingCollection(result)
    
    def delete_collection(self, name: str):
        result = self._client.delete_collection(name)
        _log_interaction("Client.delete_collection", (name,), {}, result=result)
        return result
    
    def reset(self):
        result = self._client.reset()
        _log_interaction("Client.reset", (), {}, result=result)
        return result


# Standalone demo/test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    
    print(f"Connecting to ChromaDB at {host}:{port}")
    print("All interactions will be logged below:\n")
    
    try:
        client = LoggingChromaClient(host=host, port=port)
        
        # Test heartbeat
        client.heartbeat()
        
        # Test collection operations
        collection = client.get_or_create_collection("test_logging")
        
        # Test add
        collection.add(
            ids=["test1"],
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
        )
        
        # Test query
        collection.query(
            query_texts=["test"],
            n_results=1,
        )
        
        # Test get
        collection.get(ids=["test1"])
        
        # Test count
        collection.count()
        
        # Cleanup
        collection.delete(ids=["test1"])
        
        print("\n✓ All ChromaDB interactions logged successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure ChromaDB is running and accessible.")
