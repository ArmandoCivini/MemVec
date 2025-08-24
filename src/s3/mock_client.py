"""
Mock S3 client for testing purposes.

This module provides a mock S3 client that can be used for testing
when real S3 is not available or desired.
"""


class MockS3Client:
    """Mock S3 client for testing purposes when S3 is not accessible."""
    
    def __init__(self):
        self.objects = {}
    
    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = None):
        """Mock put_object method."""
        full_key = f"{Bucket}/{Key}"
        self.objects[full_key] = {
            'Body': Body,
            'ContentType': ContentType
        }
    
    def get_object(self, Bucket: str, Key: str):
        """Mock get_object method."""
        full_key = f"{Bucket}/{Key}"
        if full_key not in self.objects:
            raise Exception(f"Object {full_key} not found")
        
        class MockBody:
            def __init__(self, data):
                self.data = data
            
            def read(self):
                return self.data
        
        return {
            'Body': MockBody(self.objects[full_key]['Body'])
        }
    
    def delete_object(self, Bucket: str, Key: str):
        """Mock delete_object method."""
        full_key = f"{Bucket}/{Key}"
        if full_key in self.objects:
            del self.objects[full_key]
    
    def head_bucket(self, Bucket: str):
        """Mock head_bucket method."""
        # For testing purposes, we'll assume the bucket always exists
        pass
