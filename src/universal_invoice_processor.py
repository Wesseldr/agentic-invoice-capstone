# ============================================================================
# File: universal_invoice_processor.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Batch Pre-processor / Gatekeeper. Handles PDF ingestion, raw text
#     extraction, and initial validation of client case numbers.
# ============================================================================

"""
Universal Invoice Processor Module
==================================

This module serves as the **Batch Pre-processor** for the Agentic Invoice Pipeline.
It is responsible for the initial ingestion of PDF files, converting them into raw text,
and performing a first-pass analysis using deterministic tools (Regex).

Key Responsibilities:
1. **Text Extraction:** extracting text from PDFs using `pdfplumber` (with `PyPDF2` fallback).
2. **Pattern Recognition:** Using `InvoiceRegexExtractor` to find potential client cases, dates, and amounts.
3. **Validation Gatekeeping:** Checking found client codes against the `ClientCaseMatcher` database to filter out
   non-coaching invoices (e.g., utility bills) before they reach the expensive LLM stage.
4. **Metadata Generation:** Producing `invoice_metadata.json` and `manifest.json` to guide the Multi-Agent system.

Dependencies:
    - `polars`: For efficient data handling and reporting.
    - `pdfplumber` / `PyPDF2`: For PDF parsing.
    - `regex_tools`: For shared regex patterns.
    - `clientcase_matcher`: For validation against the allowed-list.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import polars as pl
import pdfplumber
import PyPDF2

from clientcase_matcher import ClientCaseMatcher, MatchResult
from regex_tools import InvoiceRegexExtractor  # <--- NEW IMPORT

class ExtractionMethod(Enum):
    """Enum for tracking which extraction method was used"""
    PDF_TEXT = "pdf_text_extraction"
    TESSERACT_OCR = "tesseract_ocr"
    GOOGLE_VISION = "google_vision_api"
    FAILED = "extraction_failed"


@dataclass
class UniversalInvoiceData:
    """
    Data Transfer Object (DTO) holding all extracted information for a single invoice.
    This object acts as the state container during the pre-processing phase.
    """    
    filename: str
    extracted_text: str
    extraction_method: ExtractionMethod
    
    # Pattern-based extractions (may be None if not found)
    client_case_numbers: List[str] = field(default_factory=list)
    invoice_numbers: List[str] = field(default_factory=list)
    dates_found: List[str] = field(default_factory=list)
    amounts_found: List[float] = field(default_factory=list)
    is_coaching_invoice: bool = False
    
    # Basic metadata
    text_length: int = 0
    confidence_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    # Structured data for LLM prompting
    llm_prompt: Optional[str] = None
    patterns_found: Dict[str, Any] = field(default_factory=dict)


class UniversalInvoiceExtractor:
    """
    The main engine for batch-processing PDF invoices.

    This class iterates through a directory of PDFs, extracts their text,
    and determines if they are suitable for further processing by the AI Agent.
    It acts as a **Cost-Saving Gatekeeper**: only invoices with valid or correctable
    client case numbers are marked as 'ready_for_llm'.

    Attributes:
        invoice_dir (Path): Directory containing source PDF files.
        output_dir (Path): Directory where artifacts (raw text, metadata) will be saved.
        clientcase_matcher (ClientCaseMatcher): Instance of the validator logic.
    """
    
    def __init__(self, invoice_dir: str, output_dir: str):
        self.invoice_dir = Path(invoice_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_text_dir = self.output_dir / "raw_texts"
        self.raw_text_dir.mkdir(exist_ok=True)
        
        # Reference data for client cases
        # Assuming output_dir = data/llm_ready/, so project_root = data/
        project_root = self.output_dir.parent
        self.reference_dir = project_root / "reference"
        self.clientcase_index_path = self.reference_dir / "valid_clientcases.csv"

        self.clientcase_matcher: Optional[ClientCaseMatcher] = None
        if self.clientcase_index_path.exists():
            try:
                self.clientcase_matcher = ClientCaseMatcher(self.clientcase_index_path)
            except Exception as e:
                print(f"[WARN] Failed to initialize ClientCaseMatcher: {e}")
        else:
            print(f"[WARN] my valid ClientCaseID DB is not found at {self.clientcase_index_path}, "
                  f"skipping client case matching.")
        
        self.results = []
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> Tuple[str, bool]:
        """
        Primary extraction method using `pdfplumber`.
        Ideally suited for digital-born PDFs (selectable text).
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page markers for LLM context
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
            
            if len(text.strip()) > 50:
                return text, True
            return text, False
            
        except Exception as e:
            print(f"PDFPlumber extraction failed for {pdf_path.name}: {e}")
            return "", False
    
    def extract_text_pypdf2(self, pdf_path: Path) -> Tuple[str, bool]:
        """
        Secondary/Fallback extraction method using `PyPDF2`.
        Used if `pdfplumber` fails or returns empty text.
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
            
            if len(text.strip()) > 50:
                return text, True
            return text, False
            
        except Exception as e:
            print(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return "", False
    
    def find_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extracts all patterns found in text using the shared `InvoiceRegexExtractor`.
        This ensures consistency between Pre-processing and Agent Runtime.
        """
        return InvoiceRegexExtractor.extract_all(text)

    def annotate_client_cases(self, invoice_data: UniversalInvoiceData) -> None:
        """
        Enriches the raw extracted client codes with validation status from the Database.
        It determines if a code is 'exact', 'fuzzy' (typo corrected), or 'unknown'.
        """
        if not self.clientcase_matcher:
            return

        cases = invoice_data.client_case_numbers or []
        if not cases:
            return

        match_info = {}
        for code in cases:
            result: MatchResult = self.clientcase_matcher.match(code)
            match_info[code] = {
                "matchedCode": result.matched_code,
                "matchStatus": result.match_status,
                "matchConfidence": result.match_confidence,
                "contamination": result.contamination,
                "candidates": result.candidates,
                "notes": result.notes,
            }

        invoice_data.patterns_found["client_case_matches"] = match_info

    def _evaluate_client_case_verdict(self, invoice_data: UniversalInvoiceData) -> dict:
        """
        Evaluates the overall quality of client cases for a specific invoice.
        
        Returns a dictionary with a 'verdict' (ACCEPT/REJECT/NEEDS_REVIEW) and a 'reason'.
        This logic enforces the business rule that invoices without valid client codes
        should be rejected early.
        """
        matches = invoice_data.patterns_found.get("client_case_matches", {})
        if not matches:
            return {
                "verdict": "reject",
                "reason": "No clientCaseNumbers found; likely not a coaching invoice",
                "counts": {
                    "exact": 0,
                    "fuzzy_io_swap": 0,
                    "unknown": 0,
                    "ambiguous_io_swap": 0,
                },
            }

        counts = {
            "exact": 0,
            "fuzzy_io_swap": 0,
            "unknown": 0,
            "ambiguous_io_swap": 0,
        }
        for info in matches.values():
            status = (info.get("matchStatus") or "").lower()
            if status in counts:
                counts[status] += 1

        verdict = "accept"
        reason = "All clientCaseNumbers are known and match exactly"

        if counts["unknown"] > 0 or counts["ambiguous_io_swap"] > 0:
            verdict = "reject"
            reason = "The invoice contains clientCaseNumbers that are either unknown or ambiguous. The sender must update the invoice with valid CCNs."
        elif counts["fuzzy_io_swap"] > 0:
            verdict = "needs_review"
            reason="Invoice contains clientCaseNumbers with historical character errors (e.g., 1/I or 0/O swaps caused by the 2024–2025 bug). Characters “1” and “0” have now been auto-corrected to “I” and “O.”"
        return {
            "verdict": verdict,
            "reason": reason,
            "counts": counts,
        }
    
    def save_raw_text(self, filename: str, text: str) -> Path:
        """Saves the extracted raw text to disk for the LLM Agent to consume."""
        text_filename = filename.replace('.pdf', '_raw.txt')
        text_path = self.raw_text_dir / text_filename
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"=== RAW TEXT EXTRACTION ===\n")
            f.write(f"Source: {filename}\n")
            f.write(f"Extraction Date: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n\n")
            f.write(text)
        
        return text_path
    
    def calculate_confidence(self, invoice_data: UniversalInvoiceData) -> float:
        """
        Calculates a heuristic confidence score (0.0 - 1.0) based on the density 
        of expected patterns (dates, amounts, invoice numbers) found in the text.
        """
        if not invoice_data.is_coaching_invoice:
            if invoice_data.text_length > 0:
                return 0.1
            return 0.0

        score = 0.4

        if invoice_data.dates_found:
            score += 0.2

        if invoice_data.amounts_found:
            score += 0.2

        if invoice_data.invoice_numbers:
            score += 0.1

        if invoice_data.text_length > 500:
            score += 0.1
        elif invoice_data.text_length > 100:
            score += 0.05

        return min(score, 1.0)
    
    def process_invoice(self, pdf_path: Path) -> UniversalInvoiceData:
        """
        Orchestrates the processing of a single PDF file.
        
        Steps:
        1. Text Extraction (PDFPlumber -> PyPDF2)
        2. Pattern Extraction (Regex)
        3. Data Enrichment (ClientCaseMatcher)
        4. Scoring & Classification
        """
        start_time = datetime.now()
        
        # Try primary extraction
        text, success = self.extract_text_pdfplumber(pdf_path)
        method = ExtractionMethod.PDF_TEXT
        
        # Fallback to PyPDF2 if needed
        if not success:
            text, success = self.extract_text_pypdf2(pdf_path)
        
        if not success:
            method = ExtractionMethod.FAILED
        
        # Find patterns using the shared Regex Tool
        patterns = self.find_patterns(text) if text else {}
        
        # Create invoice data
        invoice_data = UniversalInvoiceData(
            filename=pdf_path.name,
            extracted_text=text,
            extraction_method=method,
            client_case_numbers=patterns.get('client_cases', []),
            invoice_numbers=patterns.get('invoice_numbers', []),
            dates_found=patterns.get('dates', []),
            amounts_found=patterns.get('amounts', []),
            text_length=len(text),
            processing_time=(datetime.now() - start_time).total_seconds(),
            patterns_found=patterns,
            error_message=None if success else "Failed to extract text"
        )

        # Verrijk clientcases met index-matches
        self.annotate_client_cases(invoice_data)
        
        # Evaluate client case verdict
        client_case_summary = self._evaluate_client_case_verdict(invoice_data)
        invoice_data.patterns_found["client_case_summary"] = client_case_summary

        # Check if coaching invoice based on client cases present
        invoice_data.is_coaching_invoice = bool(invoice_data.client_case_numbers)
        
        # Calculate confidence
        invoice_data.confidence_score = self.calculate_confidence(invoice_data)

        return invoice_data
    
    def process_all_invoices(self) -> pl.DataFrame:
        """
        Main execution loop. Iterates over all PDFs in the input directory.
        
        Displays real-time status updates to the console, including:
        - Validation Verdicts (using colored emojis for clarity).
        - Status of Critical Header Data (indicating if OCR/Agent intervention will be needed).
        
        Returns:
            pl.DataFrame: A summary dataframe containing metrics for all processed invoices.
        """
        pdf_files = sorted(list(self.invoice_dir.glob("*.pdf")))
        
        if not pdf_files:
            print(f"No PDF files found in {self.invoice_dir}")
            return pl.DataFrame()
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("="*60)
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            invoice_data = self.process_invoice(pdf_file)
            self.results.append(invoice_data)
            
            # Save raw text
            text_path = self.save_raw_text(invoice_data.filename, invoice_data.extracted_text)
            
            kvk_list = invoice_data.patterns_found.get("kvk_numbers", [])
            vat_list = invoice_data.patterns_found.get("vat_numbers", [])
            
            kvk_single = kvk_list[0] if kvk_list else None
            vat_single = vat_list[0] if vat_list else None
            
            summary = invoice_data.patterns_found.get("client_case_summary", {})
            verdict = summary.get("verdict", "N/A").upper()
            reason = summary.get("reason", "N/A")

            # Gebruik emojies voor visuele feedback
            if verdict == "ACCEPT":
                color_code = "🟢"
            elif verdict == "NEEDS_REVIEW":
                color_code = "🟡"
            else:
                color_code = "🔴"
            
            print(f"  {color_code} VALIDATION VERDICT: {verdict}")
            print(f"    Reason: {reason}")
            
            print("\n  🔑 HEADER DATA STATUS (Initial Extract):")
            
            # KVK Status
            if kvk_single:
                print(f"    🟢 KVK: Found ({kvk_single})")
            else:
                print("    🟡 KVK: Missing. Will trigger OCR fallback.")
                
            # VAT Status
            if vat_single:
                print(f"    🟢 VAT: Found ({vat_single})")
            else:
                print("    🟡 VAT: Missing. Will trigger OCR fallback.")
            
            # Print summary
            print(f"  Text Length: {invoice_data.text_length} characters")
            print(f"  Extraction: {invoice_data.extraction_method.value}")
            print(f"  Client Cases Found: {len(invoice_data.client_case_numbers)}")
            # kvk_list = invoice_data.patterns_found.get("kvk_numbers", [])
            # vat_list = invoice_data.patterns_found.get("vat_numbers", [])
            
            inv_list = invoice_data.invoice_numbers
            date_list = invoice_data.dates_found

            # kvk_single = kvk_list[0] if kvk_list else None
            # vat_single = vat_list[0] if vat_list else None
            inv_single = inv_list[0] if inv_list else None
            date_single = date_list[0] if date_list else None

            # print(f"  KVK: {kvk_single if kvk_single else 'None'}")
            # print(f"  VAT: {vat_single if vat_single else 'None'}")
            print(f"  Invoice Number: {inv_single if inv_single else 'None'}")
            print(f"  Invoice Date: {date_single if date_single else 'None'}")
            if invoice_data.client_case_numbers:
                print(f"    Cases: {', '.join(invoice_data.client_case_numbers[:5])}")
                if len(invoice_data.client_case_numbers) > 5:
                    print(f"    ... and {len(invoice_data.client_case_numbers)-5} more")
            print(f"  Amounts Found: {len(invoice_data.amounts_found)}")
            print(f"  Confidence: {invoice_data.confidence_score:.2f}")
            print(f"  Raw Text: {text_path.name}")
        
        # Create summary DataFrame
        df_data = []
        for inv in self.results:
            df_data.append({
                'filename': inv.filename,
                'text_length': inv.text_length,
                'extraction_method': inv.extraction_method.value,
                'client_cases_found': len(inv.client_case_numbers),
                'all_case_numbers': '|'.join(inv.client_case_numbers),
                'invoice_numbers': '|'.join(inv.invoice_numbers),
                'amounts_found': len(inv.amounts_found),
                'max_amount': max(inv.amounts_found) if inv.amounts_found else None,
                'confidence_score': inv.confidence_score,
                'processing_time': inv.processing_time,
                'has_error': inv.error_message is not None
            })
        
        df = pl.DataFrame(df_data)
        return df

    def save_results(self, df: pl.DataFrame):
            """
            Persists the processing results to disk.
        
            Creates:
            1. `invoice_metadata.json`: Detailed metrics and regex matches per invoice.
            2. `manifest.json`: A high-level index file used by the `capstone_agents.py` loader.
            """
            # Save complete JSON for programmatic access
            json_data = {}
            for inv in self.results:
                # Haal summary op (zit nu nog in patterns_found)
                client_case_summary = inv.patterns_found.get("client_case_summary", None)

                # Safety cleanup
                kvk_list = inv.patterns_found.get("kvk_numbers", [])
                vat_list = inv.patterns_found.get("vat_numbers", [])
                
                kvk_val = kvk_list[0] if kvk_list else None
                vat_val = vat_list[0] if vat_list else None
                inv_num = inv.invoice_numbers[0] if inv.invoice_numbers else None
                inv_date = inv.dates_found[0] if inv.dates_found else None

                # --- FIX DUPLICATE DATA ---
                # We maken een schone kopie van patterns_found ZONDER de summary
                # (want die zetten we zo meteen als top-level key neer)
                patterns_clean = inv.patterns_found.copy()
                patterns_clean.pop("client_case_summary", None)
                # --------------------------

                json_data[inv.filename] = {
                    "text_length": inv.text_length,
                    "extraction_method": inv.extraction_method.value,
                    "confidence_score": inv.confidence_score,
                    "kvk": kvk_val,
                    "vat": vat_val,
                    "invoice_number": inv_num,
                    "invoice_date": inv_date,
                    "patterns_found": patterns_clean,  # Gebruik de opgeschoonde dict
                    "client_case_summary": client_case_summary,
                    "raw_text_file": f"raw_texts/{inv.filename.replace('.pdf', '_raw.txt')}",
                    "text_preview": inv.extracted_text[:500] if inv.extracted_text else None,
                }
            
            json_path = self.output_dir / "invoice_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to: {json_path}")
            
            # Create a manifest file for LLM batch processing
            manifest = {
                "total_invoices": len(self.results),
                "processing_date": datetime.now().isoformat(),
                "output_structure": {
                    "raw_texts": "Full extracted text for each invoice",
                    "extraction_summary.csv": "Overview of all invoices",
                    "invoice_metadata.json": "Detailed metadata and patterns",
                },
                "invoices": [],
            }

            for inv in self.results:
                # We moeten hier wel de originele patterns_found gebruiken (of de variabele die we eerder pakten)
                # Omdat we inv.patterns_found niet in-place hebben aangepast, werkt dit nog steeds.
                client_case_summary = inv.patterns_found.get("client_case_summary", {})
                verdict = client_case_summary.get("verdict", "accept")
                reason = client_case_summary.get("reason", None)

                ready_for_llm = (
                    inv.is_coaching_invoice
                    and inv.text_length > 0
                    and verdict != "reject"
                )

                manifest["invoices"].append(
                    {
                        "filename": inv.filename,
                        "confidence": inv.confidence_score,
                        "is_coaching_invoice": inv.is_coaching_invoice,
                        "client_case_verdict": verdict,
                        "client_case_reason": reason,
                        "ready_for_llm": ready_for_llm,
                    }
                )
            
            manifest_path = self.output_dir / "manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest saved to: {manifest_path}")
            
            return df

def main():
    # """Main execution function"""
   # Use dynamic paths based on script location (src/universal_invoice_processor.py -> parent=src -> parent=root)
    project_root = Path(__file__).resolve().parent.parent
    
    invoice_dir = project_root / "data" / "invoices"
    output_dir = project_root / "data" / "llm_ready"
    
    # Initialize extractor
    extractor = UniversalInvoiceExtractor(invoice_dir, output_dir)
    
    # Process all invoices
    results_df = extractor.process_all_invoices()
    
    if not results_df.is_empty():
        # Display summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY FOR LLM PROCESSING")
        print("="*60)
        
        total = len(results_df)
        successful = len(results_df.filter(pl.col('text_length') > 0))
        
        print(f"Total invoices processed: {total}")
        print(f"Successfully extracted: {successful}/{total}")
        print(f"Average text length: {results_df['text_length'].mean():.0f} characters")
        print(f"Average confidence: {results_df['confidence_score'].mean():.2f}")
        
        # Pattern statistics
        with_cases = len(results_df.filter(pl.col('client_cases_found') > 0))
        with_amounts = len(results_df.filter(pl.col('amounts_found') > 0))
        
        print(f"\nPattern Detection:")
        print(f"  Invoices with client cases: {with_cases}/{total}")
        print(f"  Invoices with amounts: {with_amounts}/{total}")
        
        # Save all results
        extractor.save_results(results_df)
        
        print("\n" + "="*60)
        print("OUTPUT STRUCTURE")
        print("="*60)
        print(f"📁 {output_dir}")
        print("  ├── 📄 invoice_metadata.json       - Detailed patterns")
        print("  ├── 📄 manifest.json               - Processing manifest")
        print("  └── 📁 raw_texts/                  - Full text for each invoice")
        
        print("\n✅ All invoices ready for LLM processing!")
        print("The raw text preserves ALL information for LLM interpretation.")
        
        return results_df
    else:
        print("No invoices were processed successfully.")
        return None


if __name__ == "__main__":
    main()
