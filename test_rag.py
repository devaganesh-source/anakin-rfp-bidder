import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from crawler import RFPCrawler
from reasoner.vector_store import build_index, load_and_chunk_docs
from reasoner.groq_reasoner import process_all_sections

async def main():
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    
    rfp_url = os.environ.get("RFP_SOURCE_URL", "http://localhost:8080/mock_rfp.html")
    
    print("1. Crawling RFP from Web...")
    crawler = RFPCrawler(rfp_url)
    scraped_sections = await crawler.crawl()
    print(f"-> Scraped {len(scraped_sections)} requirements from the messy DOM.")
    
    print("\n2. Loading Vector Knowledge Base (FAISS)...")
    chunks = load_and_chunk_docs(project_root / "sample-data")
    faiss_index = build_index(chunks)
    print(f"-> Loaded {len(chunks)} document chunks.")
    
    print("\n3. AI Reasoning Phase (Groq - Throttled Batches)...")
    answers = []
    items = list(scraped_sections.items())
    batch_size = 3  # Process 3 sections at a time to respect the 8000 TPM limit
    
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        print(f"-> Processing batch {i // batch_size + 1} ({len(batch)} sections)...")
        batch_answers = await process_all_sections(batch, faiss_index, chunks)
        answers.extend(batch_answers)
        if i + batch_size < len(items):
            await asyncio.sleep(3)  # Cooldown delay for Groq rate limits
    
    print("\n" + "="*70)
    print("FINAL AI GENERATED PROPOSAL")
    print("="*70)
    for ans in answers:
        print(f"\n[SECTION]: {ans['section']}")
        print(f"[REQUIREMENT]: {scraped_sections[ans['section']]}")
        print(f"[ANSWER]: {ans['answer']}")
        print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())