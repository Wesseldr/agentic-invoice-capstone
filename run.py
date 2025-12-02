#!/usr/bin/env python3
# ============================================================================
# File: run.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     CLI Entry Point for Phase 1 (Pre-processing). Orchestrates the ingestion
#     of raw PDFs, executes the Universal Invoice Processor, and prepares
#     text artifacts and metadata for the Multi-Agent pipeline.
# ============================================================================

"""
Pipeline Entry Point: Phase 1 (Pre-processing)
==============================================

This script acts as the CLI entry point for the **Universal Invoice Processor**.
It handles the setup and execution of the first phase of the pipeline:
ingesting raw PDF invoices and converting them into structured text artifacts for the Agents.

Key Responsibilities:
1. **Path Management:** Dynamically resolves project paths relative to the script location, ensuring portability.
2. **Environment Setup:** Adds the `src` directory to the Python path to expose core modules.
3. **Orchestration:** Initializes the `UniversalInvoiceExtractor` with the correct input/output directories.
4. **User Feedback:** Provides clear console output regarding the processing status and next steps.

Usage:
    Run this script directly from the command line to process all invoices in `data/invoices/`:
    $ python run.py
"""

import sys
from pathlib import Path

# Dynamisch het pad bepalen (waar dit script staat)
PROJECT_ROOT = Path(__file__).resolve().parent

# Add src directory to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from universal_invoice_processor import UniversalInvoiceExtractor

def main():
    """
    Executes the batch processing of invoices.

    Flow:
    1. verifies that the input directory (`data/invoices`) exists.
    2. Initializes the `UniversalInvoiceExtractor`.
    3. Triggers the batch processing loop.
    4. Saves artifacts (`invoice_metadata.json`, `manifest.json`) if successful.
    5. Reports final status to the console.

    Returns:
        Optional[pl.DataFrame]: A dataframe containing processing metrics if successful, else None.
    """
    print("="*60)
    print("UNIVERSAL INVOICE PROCESSOR")
    print("Extracting raw text for LLM processing")
    print("="*60)
    
    invoice_dir = PROJECT_ROOT / "data" / "invoices"
    output_dir = PROJECT_ROOT / "data" / "llm_ready"
    
    # Initialize extractor
    #print(f"\n📂 Input directory: {invoice_dir}")
    #print(f"📂 Output directory: {output_dir}\n")
    
    # Check if input dir exists
    if not invoice_dir.exists():
        print(f"❌ Error: Input directory not found at {invoice_dir}")
        print("   Did you run the generator script?")
        return None

    extractor = UniversalInvoiceExtractor(str(invoice_dir), str(output_dir))
    
    # Process all invoices
    results_df = extractor.process_all_invoices()
    
    if results_df is not None and not results_df.is_empty():
        # Save results
        extractor.save_results(results_df)
        
        print("\n" + "="*60)
        print("✅ SUCCESS - All invoices processed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check raw_texts/ folder for complete text extractions")
        print("2. Use llm_prompts/ folder for LLM API calls")
        
        return results_df
    else:
        print("\n❌ No invoices were processed successfully.")
        return None

if __name__ == "__main__":
    results = main()
