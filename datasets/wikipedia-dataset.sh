#!/bin/bash
set -e

# URL for Simple English Wikipedia dump (smaller and good for POC)
DUMP_URL="https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
OUTPUT_FILE="simplewiki-latest-pages-articles.xml.bz2"
EXTRACTED_FILE="simplewiki-latest-pages-articles.xml"

echo "Downloading Simple English Wikipedia dump..."
curl -L "$DUMP_URL" -o "$OUTPUT_FILE"

echo "Extracting dump..."
bunzip2 "$OUTPUT_FILE"

echo "Extraction complete: $EXTRACTED_FILE"
echo "Cleaning up compressed file..."
rm -f "$OUTPUT_FILE" 2>/dev/null || true

echo "Done! You now have $EXTRACTED_FILE ready for parsing."
