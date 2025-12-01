# ============================================================================
# File: utils.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Shared utility functions for data loading, text cleaning, and business
#     rule enforcement (e.g. applying correction maps).
# ============================================================================

"""
Utilities Module
================

This module provides shared helper functions for the Agentic Invoice Processor pipeline.
It handles critical data processing tasks that sit between the raw storage and the AI Agents.

Key Responsibilities:
1. **Data Loading:** Reading manifest files, metadata, and raw text for processing.
2. **Context Preparation:** assembling the correct "hints" and "allowed lists" for the Agent prompts.
3. **Output Cleaning:** stripping Markdown formatting from LLM responses to ensure valid JSON.
4. **Post-Processing:** Applying corrections (OCR fixes) and enforcing strict business rules on the extraction results.

Dependencies:
    - `llm_models`: For type-safe manipulation of the invoice data structures.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from llm_models import InvoiceLLMResult

def strip_markdown_json_fences(s: str) -> str:
    """
    Robustly extracts a JSON string from raw LLM output.

    LLMs often wrap their JSON output in Markdown code blocks (```json ... ```) or include
    conversational filler ("Here is the JSON you asked for..."). This function strips
    all of that away to return a clean, parseable JSON string.

    Args:
        s (str): The raw string response from the LLM.

    Returns:
        str: A clean string containing only the JSON object (from the first '{' to the last '}').
             Returns the original string if no brackets are found.
    """
    if not s:
        return s
        
    text = s.strip()
    
    # Zoek naar het begin en einde van het JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx != -1 and end_idx != -1:
        # Pak alles tussen de eerste { en laatste }
        return text[start_idx : end_idx + 1]
        
    # Fallback: als we geen brackets vinden, proberen we standaard markdown stripping
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        text = "\n".join(lines).strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[0].strip()
            
    return text

def apply_client_case_corrections(result: InvoiceLLMResult, correction_map: Dict[str, str]) -> InvoiceLLMResult:
    """
    Applies canonical corrections to the Agent's output based on pre-calculated matches.

    This function bridges the gap between OCR errors (e.g., '125') and database reality (e.g., 'I25').
    It checks the code returned by the Agent against a correction map. If a correction exists,
    it swaps the values and preserves the original 'typo' for audit purposes.

    Args:
        result (InvoiceLLMResult): The initial result object from the Agent.
        correction_map (Dict[str, str]): A mapping of {Raw_Code -> Corrected_Code} derived from 
                                         the Universal Processor phase.

    Returns:
        InvoiceLLMResult: The updated result object with corrected case IDs.
    """
    for case in result.clientCases:
        # De agent heeft in eerste instantie gevuld wat hij zag in 'validatedClientCaseNumber'
        # (omdat de prompt dat vroeg). We checken nu of die waarde gecorrigeerd moet worden.
        raw_code = case.validatedClientCaseNumber
        
        if raw_code in correction_map:
            corrected_code = correction_map[raw_code]
            
            # Only swap if they are different
            if raw_code != corrected_code:
                print(f"   ✨ Auto-Correcting: {raw_code} -> {corrected_code}")
                
                # Bewaar de originele (foute) OCR tekst in het 'raw' veld
                case.rawClientCaseNumber = raw_code
                
                # Update het hoofdveld naar de correcte database code
                case.validatedClientCaseNumber = corrected_code
    
    # Also fix NoActivity list (dit zijn strings, dus direct vervangen)
    fixed_no_activity = []
    for code in result.clientCasesNoActivity:
        if code in correction_map:
            fixed_no_activity.append(correction_map[code])
        else:
            fixed_no_activity.append(code)
    
    result.clientCasesNoActivity = sorted(list(set(fixed_no_activity)))
    
    return result

def enforce_allowed_client_cases(result: InvoiceLLMResult, allowed: List[str]) -> InvoiceLLMResult:
    """
    Acts as the final Gatekeeper for data quality.

    Ensures that every client case number remaining in the result is present in the 
    strict 'Allowed List' for this specific invoice. Any code not in the list is dropped.
    This prevents hallucinations or non-existent codes from entering the downstream system.

    Args:
        result (InvoiceLLMResult): The extracted invoice data.
        allowed (List[str]): The strict list of valid client case IDs for this context.

    Returns:
        InvoiceLLMResult: The filtered result object containing only valid cases.
    """
    allowed_set = set(allowed)

    filtered_active = []
    for case in result.clientCases:
        # UPDATE: We checken nu het gevalideerde veld
        if case.validatedClientCaseNumber in allowed_set:
            filtered_active.append(case)
        else:
            print(f"   [WARN] Dropping invalid clientCaseNumber in clientCases: {case.validatedClientCaseNumber!r}")

    filtered_no_activity = []
    for code in result.clientCasesNoActivity or []:
        if code in allowed_set:
            filtered_no_activity.append(code)
        else:
            print(f"   [WARN] Dropping invalid clientCaseNumber in clientCasesNoActivity: {code!r}")

    result.clientCases = filtered_active
    result.clientCasesNoActivity = sorted(list(set(filtered_no_activity)))
    return result

def load_all_coaching_invoices(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads and prepares the invoice tasks for the Orchestrator.

    This function reads the pre-processed metadata and raw text files. It constructs a
    comprehensive context object for each invoice, including:
    - The raw text content.
    - Metadata hints (KvK, VAT).
    - A 'Prompt List' (containing raw codes/typos) to help the Agent 'see' the data.
    - A 'Valid List' (containing correct codes) for final validation.
    - A 'Correction Map' to translate between the two.

    Args:
        base_dir (Path): The root directory containing 'manifest.json' and 'invoice_metadata.json'.

    Returns:
        List[Dict[str, Any]]: A list of invoice context dictionaries, sorted by filename.
    """
    manifest_path = base_dir / "manifest.json"
    metadata_path = base_dir / "invoice_metadata.json"
    raw_texts_dir = base_dir / "raw_texts"

    if not manifest_path.exists() or not metadata_path.exists():
        print("❌ manifest.json or invoice_metadata.json not found.")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    invoices = manifest.get("invoices", [])
    examples: List[Dict[str, Any]] = []

    for inv in invoices:
        if not (inv.get("is_coaching_invoice") and inv.get("ready_for_llm")):
            continue

        filename = inv.get("filename")
        if not filename: continue

        meta = metadata.get(filename)
        if meta is None: continue

        raw_text_file = raw_texts_dir / filename.replace(".pdf", "_raw.txt")
        if not raw_text_file.exists(): continue

        with open(raw_text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        patterns_found = meta.get("patterns_found", {}) or {}
        
        # --- Build Correction Map ---
        client_case_matches = patterns_found.get("client_case_matches", {})
        
        allowed_client_cases_raw = [] # Voor de Agent prompt (zodat hij de tekst vindt)
        allowed_client_cases_valid = [] # Voor de validator (zodat we schone data krijgen)
        correction_map = {} # De vertaalslag: Fout -> Goed

        for raw_code, info in client_case_matches.items():
            matched_code = info.get("matchedCode")
            status = info.get("matchStatus")
            
            if matched_code and status in ["exact", "fuzzy_io_swap"]:
                allowed_client_cases_raw.append(raw_code)
                allowed_client_cases_valid.append(matched_code)
                correction_map[raw_code] = matched_code

        # ----------------------------------------
        
        examples.append({
            "filename": filename,
            "raw_text": raw_text,
            "kvk_hint": meta.get("kvk"),
            "vat_hint": meta.get("vat"),
            "invoice_number_hint": meta.get("invoice_number"),
            "invoice_date_hint": meta.get("invoice_date"),
            
            # Agent krijgt RAW codes
            "allowed_client_cases_prompt": allowed_client_cases_raw, 
            
            # Validator krijgt VALID codes
            "allowed_client_cases_valid": allowed_client_cases_valid,
            
            # Utils krijgt de map
            "correction_map": correction_map
        })
        
    examples.sort(key=lambda x: x['filename'])
    return examples