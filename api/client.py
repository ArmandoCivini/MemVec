"""
Simple client example for testing MemVec API.
"""

import requests
import json


class MemVecClient:
    """Simple client for interacting with MemVec API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def upload_file(self, file_path, metadata=None):
        """Upload a file via API."""
        url = f"{self.base_url}/upload"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if metadata:
                data['metadata'] = json.dumps(metadata)
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def list_files(self):
        """List all processed files."""
        url = f"{self.base_url}/files"
        response = requests.get(url)
        return response.json()
    
    def query_vectors(self, query_text, k=5, threshold=None):
        """Query vectors via API."""
        url = f"{self.base_url}/query"
        payload = {
            "query_text": query_text,
            "k": k,
            "threshold": threshold
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def get_stats(self):
        """Get system statistics."""
        url = f"{self.base_url}/stats"
        response = requests.get(url)
        return response.json()
    
    def health_check(self):
        """Check if API is running."""
        url = f"{self.base_url}/"
        response = requests.get(url)
        return response.json()


def main():
    """Example usage of MemVec API client."""
    client = MemVecClient()
    
    # Health check
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Create a sample text file for upload
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Machine learning is a subset of artificial intelligence.\n")
        f.write("Vector databases are useful for similarity search.\n")
        temp_file_path = f.name
    
    try:
        # Upload file
        print("\n=== Uploading File ===")
        upload_result = client.upload_file(
            temp_file_path, 
            metadata={"source": "example", "type": "text"}
        )
        print(json.dumps(upload_result, indent=2))
        
        # List files
        print("\n=== Listing Files ===")
        files = client.list_files()
        print(json.dumps(files, indent=2))
        
        # Query vectors
        print("\n=== Querying Vectors ===")
        query_result = client.query_vectors("artificial intelligence", k=2)
        print(json.dumps(query_result, indent=2))
        
        # Get stats
        print("\n=== System Stats ===")
        stats = client.get_stats()
        print(json.dumps(stats, indent=2))
    
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    main()
