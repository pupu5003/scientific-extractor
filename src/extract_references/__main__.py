"""
__main__.py
CLI entry point for the extraction module.
"""
import asyncio
import argparse
import sys
import os
import json
from dotenv import load_dotenv
from .clients import AsyncLLMClient
from .pipeline import ExtractionPipeline

def main():
    load_dotenv() # Automatically loads .env file
    
    parser = argparse.ArgumentParser(description="Production-grade Scientific Reference Extractor")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--grobid_url", type=str, default="http://localhost:8070", help="GROBID server URL")
    parser.add_argument("--llm_backend", type=str, default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Initialize LLM Client
    api_key = os.environ.get("OPENAI_API_KEY", "ollama_placeholder")
    base_url = "http://localhost:11434/v1" if args.llm_backend == "ollama" else None
    
    llm_client = AsyncLLMClient(api_key=api_key, base_url=base_url, model=args.model)
    pipeline = ExtractionPipeline(
        grobid_url=args.grobid_url, 
        llm_client=llm_client, 
        max_concurrency=args.concurrency
    )

    # Execute async pipeline
    try:
        results = asyncio.run(pipeline.run(args.pdf_path))
        
        # Output strictly matching the Pydantic schema
        output_data = [res.model_dump(exclude_none=True) for res in results]
        
        output_file = f"{args.pdf_path}_extracted.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n[Success] Processed {len(results)} citations.")
        print(f"[Success] Saved structured output to: {output_file}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()