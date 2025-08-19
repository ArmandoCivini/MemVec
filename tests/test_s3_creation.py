"""
S3 Vector dataset upload tests for existing buckets.

This test works with pre-existing S3 Vector buckets (created via AWS Console).
Set VECTOR_BUCKET_NAME environment variable to specify which bucket to use.

Environment variables:
- VECTOR_BUCKET_NAME: Specify bucket to use (required if multiple buckets exist)
- MEMVEC_FORCE_RECREATE=true: Delete and recreate index 
- MEMVEC_SKIP_IF_EXISTS=false: Upload even if data exists  
- MAX_ARTICLES=N: Limit number of articles to process

Examples:
  # List available buckets
  python tests/test_s3_creation.py
  
  # Use specific bucket
  VECTOR_BUCKET_NAME=my-vector-bucket pytest tests/test_s3_creation.py -v
  
  # Force recreate index
  VECTOR_BUCKET_NAME=my-vector-bucket MEMVEC_FORCE_RECREATE=true pytest tests/test_s3_creation.py -v
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Iterator
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from s3.creation import S3VectorManager, ensure_bucket_exists, create_index_simple
from config.env import S3_BUCKET_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "simplewiki-latest-pages-articles.xml"
TEMP_JSON_PATH = Path(__file__).parent / "temp_simplewiki_chunks.json"
TEST_INDEX = "simplewiki-index"
CHUNK_SIZE = 512
MAX_ARTICLES = 100

# Test control flags
FORCE_RECREATE = os.getenv("MEMVEC_FORCE_RECREATE", "false").lower() in ["true", "1", "yes"]
SKIP_UPLOAD_IF_EXISTS = os.getenv("MEMVEC_SKIP_IF_EXISTS", "true").lower() in ["true", "1", "yes"]

# Allow user to specify bucket via environment variable
USER_BUCKET = os.getenv("VECTOR_BUCKET_NAME")


class WikipediaXMLParser:
    """Parser for Wikipedia XML dumps."""
    
    def __init__(self, xml_file: str):
        self.xml_file = xml_file
        self.namespace = {"": "http://www.mediawiki.org/xml/export-0.11/"}
    
    def parse_articles(self, max_articles: int = None) -> Iterator[Dict[str, str]]:
        """Parse Wikipedia XML and yield article data."""
        try:
            context = ET.iterparse(self.xml_file, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            article_count = 0
            current_page = {}
            
            for event, elem in context:
                tag = elem.tag.replace("{http://www.mediawiki.org/xml/export-0.11/}", "")
                
                if event == 'end':
                    if tag == 'title':
                        current_page['title'] = elem.text or ""
                    elif tag == 'id' and 'id' not in current_page:
                        current_page['id'] = elem.text or ""
                    elif tag == 'text':
                        current_page['text'] = elem.text or ""
                    elif tag == 'page':
                        if all(key in current_page for key in ['title', 'id', 'text']):
                            if (not current_page['text'].strip().lower().startswith('#redirect') and
                                len(current_page['text'].strip()) > 100):
                                
                                yield {
                                    'title': current_page['title'],
                                    'id': current_page['id'],
                                    'text': self._clean_wikitext(current_page['text'])
                                }
                                
                                article_count += 1
                                if max_articles and article_count >= max_articles:
                                    break
                        
                        current_page = {}
                        
                    elem.clear()
                    root.clear()
                    
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            raise
    
    def _clean_wikitext(self, text: str) -> str:
        """Basic wikitext cleaning."""
        if not text:
            return ""
        
        import re
        
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        # Remove file/image references
        text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text)
        # Convert wiki links [[target|text]] to text, [[target]] to target
        text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)
        text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
        # Remove HTML tags and references
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)
        # Clean whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()


class TextChunker:
    """Split text into chunks for vector embedding."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_article(self, article: Dict[str, str]) -> List[Dict[str, str]]:
        """Split article into chunks."""
        text = article['text']
        title = article['title']
        article_id = article['id']
        
        if len(text) <= self.chunk_size:
            return [{
                'key': f"{article_id}_chunk_0",
                'text': text,
                'metadata': {
                    'title': title,
                    'article_id': article_id,
                    'chunk_index': 0,
                    'chunk_count': 1
                }
            }]
        
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_ends = ['.', '!', '?', '\n']
                for i in range(min(100, end - start)):
                    if text[end - i - 1] in sentence_ends:
                        end = end - i
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'key': f"{article_id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'metadata': {
                        'title': title,
                        'article_id': article_id,
                        'chunk_index': chunk_index,
                        'chunk_count': -1
                    }
                })
                chunk_index += 1
            
            start = end - self.overlap if end < len(text) else len(text)
        
        # Update chunk_count for all chunks
        for chunk in chunks:
            chunk['metadata']['chunk_count'] = len(chunks)
        
        return chunks


def get_target_bucket() -> str:
    """Get target bucket name from user input or environment."""
    manager = S3VectorManager()
    available_buckets = manager.list_buckets()
    
    if not available_buckets:
        logger.error("No vector buckets found. Please create one via AWS Console first.")
        return None
    
    # If user specified a bucket, use it
    if USER_BUCKET:
        if USER_BUCKET in available_buckets:
            logger.info(f"Using user-specified bucket: {USER_BUCKET}")
            return USER_BUCKET
        else:
            logger.error(f"Specified bucket '{USER_BUCKET}' not found in available buckets: {available_buckets}")
            return None
    
    # If only one bucket, use it
    if len(available_buckets) == 1:
        bucket = available_buckets[0]
        logger.info(f"Using single available bucket: {bucket}")
        return bucket
    
    # Multiple buckets - need user to specify
    logger.error(f"Multiple buckets available: {available_buckets}")
    logger.error("Please specify which bucket to use with: VECTOR_BUCKET_NAME=bucket-name")
    return None


def process_simplewiki_dataset(xml_path: str, max_articles: int = None, 
                             chunk_size: int = 512) -> List[Dict[str, str]]:
    """Process SimplWiki XML dump into text chunks ready for embedding."""
    logger.info(f"Processing SimplWiki dataset: {xml_path}")
    logger.info(f"Max articles: {max_articles}, Chunk size: {chunk_size}")
    
    parser = WikipediaXMLParser(xml_path)
    chunker = TextChunker(chunk_size=chunk_size)
    
    all_chunks = []
    article_count = 0
    
    try:
        for article in tqdm(parser.parse_articles(max_articles), desc="Processing articles"):
            chunks = chunker.chunk_article(article)
            all_chunks.extend(chunks)
            article_count += 1
            
            if article_count % 10 == 0:
                logger.info(f"Processed {article_count} articles, {len(all_chunks)} chunks")
    
    except Exception as e:
        logger.error(f"Error processing articles: {e}")
        raise
    
    logger.info(f"Finished: {article_count} articles, {len(all_chunks)} total chunks")
    return all_chunks


@pytest.fixture
def target_bucket():
    """Get target bucket for testing."""
    bucket = get_target_bucket()
    if not bucket:
        pytest.skip("No valid target bucket available")
    return bucket


@pytest.fixture
def s3_manager():
    """Create S3VectorManager instance."""
    return S3VectorManager()


@pytest.fixture
def processed_chunks():
    """Process SimplWiki dataset and return chunks."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found: {DATASET_PATH}")
    
    # Check if we already have processed chunks
    if TEMP_JSON_PATH.exists():
        logger.info(f"Loading existing processed chunks from {TEMP_JSON_PATH}")
        with open(TEMP_JSON_PATH, 'r') as f:
            return json.load(f)
    
    # Process the dataset
    chunks = process_simplewiki_dataset(
        str(DATASET_PATH), 
        max_articles=MAX_ARTICLES, 
        chunk_size=CHUNK_SIZE
    )
    
    # Save processed chunks for reuse
    logger.info(f"Saving processed chunks to {TEMP_JSON_PATH}")
    with open(TEMP_JSON_PATH, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return chunks


def test_bucket_discovery(s3_manager):
    """Test bucket discovery and validation."""
    buckets = s3_manager.list_buckets()
    assert len(buckets) > 0, "No vector buckets found. Create one via AWS Console first."
    
    target = get_target_bucket()
    assert target is not None, "No valid target bucket determined"
    assert target in buckets, f"Target bucket {target} not in available buckets"


def test_index_management(target_bucket, s3_manager):
    """Test index creation and management."""
    logger.info(f"Testing index management for bucket: {target_bucket}")
    
    # Check existing indexes
    existing_indexes = s3_manager.list_indexes(target_bucket)
    logger.info(f"Existing indexes: {existing_indexes}")
    
    index_exists = s3_manager.index_exists(target_bucket, TEST_INDEX)
    logger.info(f"Index '{TEST_INDEX}' exists: {index_exists}")
    
    if index_exists and FORCE_RECREATE:
        logger.info("Force recreate flag set - deleting existing index")
        success = s3_manager.delete_index(target_bucket, TEST_INDEX)
        assert success, f"Failed to delete index {TEST_INDEX}"
        index_exists = False
    
    if not index_exists:
        success = s3_manager.create_index(
            bucket_name=target_bucket,
            index_name=TEST_INDEX,
            dimension=1536,  # Bedrock embedding dimension
            distance_metric="cosine"
        )
        assert success, f"Failed to create index {TEST_INDEX}"
        
        # Verify index was created
        assert s3_manager.index_exists(target_bucket, TEST_INDEX), "Index not found after creation"
    else:
        logger.info(f"Index {TEST_INDEX} already exists - skipping creation")


def test_simplewiki_upload(target_bucket, s3_manager, processed_chunks):
    """Test uploading SimplWiki chunks to S3 Vectors."""
    if not processed_chunks:
        pytest.skip("No processed chunks available")
    
    logger.info(f"Testing SimplWiki upload to {target_bucket}/{TEST_INDEX}")
    
    # Ensure index exists
    if not s3_manager.index_exists(target_bucket, TEST_INDEX):
        success = s3_manager.create_index(
            bucket_name=target_bucket, 
            index_name=TEST_INDEX, 
            dimension=1536
        )
        assert success, f"Failed to create index: {TEST_INDEX}"
    
    logger.info(f"Uploading {len(processed_chunks)} chunks with Bedrock embedding")
    
    # Upload in batches
    batch_size = 5  # Smaller batches for testing
    successful_batches = 0
    total_uploaded = 0
    
    try:
        for i in tqdm(range(0, len(processed_chunks), batch_size), desc="Uploading batches"):
            batch = processed_chunks[i:i + batch_size]
            
            try:
                success = s3_manager.upload_texts_with_bedrock(
                    bucket_name=target_bucket,
                    index_name=TEST_INDEX,
                    texts_data=batch
                )
                
                if success:
                    successful_batches += 1
                    total_uploaded += len(batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1} ({len(batch)} chunks)")
                else:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}")
                    
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {e}")
    
    except KeyboardInterrupt:
        logger.warning("Upload interrupted by user")
    
    total_batches = (len(processed_chunks) + batch_size - 1) // batch_size
    logger.info(f"Upload complete: {successful_batches}/{total_batches} batches successful")
    logger.info(f"Total chunks uploaded: {total_uploaded}")
    
    assert successful_batches > 0, "No batches uploaded successfully"


def test_cleanup():
    """Clean up temporary files."""
    if TEMP_JSON_PATH.exists():
        logger.info(f"Cleaning up temporary file: {TEMP_JSON_PATH}")
        TEMP_JSON_PATH.unlink()


if __name__ == "__main__":
    # Direct execution for quick testing
    logging.basicConfig(level=logging.INFO)
    
    # Check AWS credentials
    from config.env import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    if not AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID == "xxx":
        logger.error("AWS credentials not configured in .env file")
        sys.exit(1)
    
    logger.info(f"S3 Vector Testing - Region: {AWS_REGION}")
    logger.info(f"Force recreate: {FORCE_RECREATE}, Skip if exists: {SKIP_UPLOAD_IF_EXISTS}")
    
    # Test bucket discovery
    try:
        manager = S3VectorManager()
        buckets = manager.list_buckets()
        
        if not buckets:
            logger.error("No vector buckets found!")
            logger.error("Please create a vector bucket via AWS Console first:")
            logger.error("1. Go to S3 Console")
            logger.error("2. Create a new bucket with 'Vector' type")
            logger.error("3. Note the bucket name and set VECTOR_BUCKET_NAME=bucket-name")
            sys.exit(1)
        
        target = get_target_bucket()
        if target:
            logger.info(f"Target bucket: {target}")
            
            # Test dataset processing if available
            if DATASET_PATH.exists():
                max_articles = int(os.getenv("MAX_ARTICLES", "3"))
                chunks = process_simplewiki_dataset(str(DATASET_PATH), max_articles, CHUNK_SIZE)
                logger.info(f"Generated {len(chunks)} chunks from {max_articles} articles")
                
                # Save sample
                sample_path = Path(__file__).parent / "sample_chunks.json"
                with open(sample_path, 'w') as f:
                    json.dump(chunks[:2], f, indent=2)
                logger.info(f"Sample saved to {sample_path}")
            else:
                logger.warning(f"Dataset not found: {DATASET_PATH}")
                logger.info("Run: ./datasets/wikipedia-dataset.sh")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        sys.exit(1)