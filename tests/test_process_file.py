"""
Simple test for process_file functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.process_file import process_file


def test_process_file(file_path: str):
    """Test processing a file and creating Vector objects."""
    
    print(f"Processing file: {file_path}")
    print("=" * 50)
    
    try:
        # Process the file
        vectors = process_file(file_path)
        
        print(f"✓ Successfully processed file")
        print(f"✓ Generated {len(vectors)} vectors")
        
        # Show details of first few vectors
        for i, vector in enumerate(vectors[:3]):  # Show first 3 vectors
            print(f"\nVector {i+1}:")
            print(f"  Index: {vector.index}")
            print(f"  Embedding dimension: {len(vector.values)}")
            print(f"  Source file: {vector.metadata.get('source_file', 'N/A')}")
            print(f"  Chunk index: {vector.metadata.get('chunk_index', 'N/A')}")
            print(f"  Text preview: {vector.metadata.get('text', 'N/A')[:100]}...")
        
        if len(vectors) > 3:
            print(f"\n... and {len(vectors) - 3} more vectors")
        
        print(f"\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error processing file: {str(e)}")
        return False


if __name__ == "__main__":
    # Test file path - can be changed to test different files
    TEST_FILE_PATH = "datasets/attention.pdf"
    
    print("Process File Test")
    print("=" * 30)
    
    success = test_process_file(TEST_FILE_PATH)
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Test failed!")
