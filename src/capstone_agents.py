# ============================================================================
# File: capstone_agents.py
# Author: J.W. de Roode 
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Core Orchestrator and Agent definitions. Implements the multi-agent
#     workflow and tiered self-correction strategy (Regex -> OCR -> LLM).
# ============================================================================

"""
Capstone Multi-Agent Invoice Processor
======================================

This module implements the core **Orchestrator Logic** for the Agentic Invoice Processor.
It manages the lifecycle of processing a single invoice by coordinating multiple
specialized AI Agents and deterministic tools.

Key Features:
- **Multi-Agent Architecture:** Splits the task between a `HeaderAgent` (metadata) and a `LineItemAgent` (tables).
- **Tiered Self-Correction:** Automatically detects missing data and attempts to recover it using a cost-efficient
  fallback strategy: regex on OCR text first (Tier 1), followed by LLM extraction (Tier 2) if necessary.
- **Resilience:** Implements retry logic with exponential backoff for API stability.

Classes:
    InvoiceOrchestrator: The central controller for the extraction pipeline.

Dependencies:
    - Google ADK (Agent Development Kit)
    - Google Cloud Vision (via ocr_tools)
    - Local utility modules (regex_tools, utils, llm_models)
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional, List
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError, BaseModel, Field

# ADK Imports
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

# Local imports (Clean Architecture)
from ocr_tools import run_pdf_ocr_google
from regex_tools import InvoiceRegexExtractor
# UPDATE: Added apply_client_case_corrections to imports
from utils import load_all_coaching_invoices, strip_markdown_json_fences, enforce_allowed_client_cases, apply_client_case_corrections
from llm_models import InvoiceLLMResult, InvoiceHeader, ClientCase

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash-lite"  # Fast, efficient model for extraction

# --- Partial Models for Individual Agents ---

class HeaderResult(BaseModel):
    """
    Intermediate data model for the HeaderAgent's output.
    Captures administrative metadata and the classification of the invoice type.
    """
    invoiceHeader: InvoiceHeader
    isCoachingInvoice: bool

class LineItemsResult(BaseModel):
    """
    Intermediate data model for the LineItemAgent's output.
    Captures the list of active client cases and those explicitly mentioned with no activity.
    """
    clientCases: List[ClientCase] = Field(default_factory=list)
    clientCasesNoActivity: List[str] = Field(default_factory=list)

# --- Prompts ---

def get_header_prompt(raw_text: str, hints: Dict[str, Any]) -> str:
    """
    Constructs the prompt for the HeaderAgent.

    This prompt instructs the LLM to extract administrative metadata (Supplier, Date, Totals)
    and specifically warns against common hallucinations (e.g., confusing Client with Supplier).

    Args:
        raw_text (str): The full text content extracted from the invoice PDF.
        hints (Dict[str, Any]): A dictionary of 'best guess' values (kvk, vat, etc.) 
                                derived from the pre-processing Regex step.

    Returns:
        str: A fully formatted instruction string for the LLM.
    """
    return f"""
You are the 'HeaderAgent', a specialist in extracting administrative metadata from Dutch invoices.

YOUR GOAL: Extract ONLY the header information into a JSON object. Ignore line items and hours.

INPUT DATA:
- KVK Hint: {hints.get('kvk_hint')}
- VAT Hint: {hints.get('vat_hint')}
- Date Hint: {hints.get('invoice_date_hint')}
- Invoice # Hint: {hints.get('invoice_number_hint')}

RAW TEXT:
{raw_text}

INSTRUCTIONS:
1. Extract the supplier name carefully. It is the party receiving payment.
   - Look for 't.n.v.', 'IBAN Name', 'KvK' or 'BTW' indicators.
   - CRITICAL: Do NOT select the name following 'Aan:', 'To:', 'Factuur aan:', or the address block of the invoice recipient. The entity being addressed is the Client, NOT the Supplier.

2. Extract Invoice Number, Date (YYYY-MM-DD), KVK, and VAT.
   - CRITICAL: Do NOT extract the Invoice Number from the "Source:" filename line at the top of the text.
   - CRITICAL: Do NOT guess or invent a number (like '0008') from the file name.
   - Only extract a number if it is clearly part of the invoice content (e.g. next to 'Factuurnummer', 'Ref', 'Kenmerk' or inside the text body).
   - If no clear number is found, return null.

3. Determine if this looks like a coaching invoice (isCoachingInvoice).

OUTPUT SCHEMA (JSON ONLY):
{{
  "invoiceHeader": {{
    "supplierName": "string | null",
    "invoiceNumber": "string | null",
    "invoiceDate": "YYYY-MM-DD | null",
    "kvkNumber": "string | null",
    "vatNumber": "string | null"
  }},
  "isCoachingInvoice": boolean
}}
"""

def get_line_item_prompt(raw_text: str, allowed_cases: List[str]) -> str:
    """
    Constructs the prompt for the LineItemAgent.

    This prompt instructs the LLM to extract tabular data regarding coaching sessions.
    It enforces a strict 'Allowed List' policy to prevent hallucination of non-existent client codes.

    Args:
        raw_text (str): The full text content extracted from the invoice PDF.
        allowed_cases (List[str]): A list of valid 'raw' client case strings known to exist 
                                   in the database (or close matches found by Regex).

    Returns:
        str: A fully formatted instruction string for the LLM.
    """
    allowed_list_str = "\n".join(allowed_cases) if allowed_cases else "(No allowed cases provided)"
    
    return f"""
You are the 'LineItemAgent', a specialist in extracting coaching sessions and hours from invoices.

YOUR GOAL: Extract ONLY the client cases, dates, and hours. Ignore the header info (address, VAT, etc).

CRITICAL RULE: ALLOWED CLIENT CASES
You may ONLY use client case numbers from this list. If a code on the invoice is not in this list (or is a typo), do NOT include it as a valid case, or try to correct it to the nearest match in this list.
--- START ALLOWED LIST ---
{allowed_list_str}
--- END ALLOWED LIST ---

RAW TEXT:
{raw_text}

INSTRUCTIONS:
1. Find all rows with hours/sessions.
2. Map them to the 'validatedClientCaseNumber' from the ALLOWED LIST.
3. If a date is mentioned for a line, extract it (YYYY-MM-DD).
4. Sum hours if multiple lines exist for the same case (unless separate dates are needed).
5. 'clientCasesNoActivity' are valid codes from the allowed list that appear on the invoice but have 0 hours or no cost.

OUTPUT SCHEMA (JSON ONLY):
{{
  "clientCases": [
    {{
      "validatedClientCaseNumber": "string (must be in allowed list)",
      "rawClientCaseNumber": "string (text as found on invoice)",
      "date": "YYYY-MM-DD | null",
      "durationHours": number | null
    }}
  ],
  "clientCasesNoActivity": ["string", ...]
}}
"""

# --- The Orchestrator Class ---

class InvoiceOrchestrator:
    """
    The central controller for the invoice processing pipeline.
    
    Responsibilities:
    1. Initializes Agents with specific roles and configurations.
    2. Dispatches tasks to agents in parallel (AsyncIO).
    3. Monitors outputs for missing critical data (KvK/VAT).
    4. Executes the Tiered Self-Correction strategy (OCR -> Regex -> LLM) if needed.
    5. Aggregates and validates the final result against business rules.
    """
    def __init__(self):
        self._setup_auth()
        
    def _setup_auth(self):
        """Loads API keys from the environment."""
        load_dotenv(os.path.expanduser("~/.env"))
        if not os.getenv("GOOGLE_API_KEY"):
            print("⚠️ Warning: GOOGLE_API_KEY not found in env.")

    def create_header_agent(self) -> Agent:
            """Creates the HeaderAgent with deterministic settings (temperature=0)."""  
            return Agent(
                name="HeaderAgent",
                model=MODEL_NAME,
                instruction="You are a strict JSON extractor for invoice headers.",
                tools=[], 
                output_key="json_result",
                # --- FIX: JUISTE PARAMETER NAAM ---
                generate_content_config={"temperature": 0.0}
        )

    def create_line_item_agent(self) -> Agent:
        """Creates the LineItemAgent with deterministic settings (temperature=0)."""
        return Agent(
            name="LineItemAgent",
            model=MODEL_NAME,
            instruction="You are a strict JSON extractor for invoice line items.",
            tools=[], 
            output_key="json_result",
            # --- FIX: JUISTE PARAMETER NAAM ---
            generate_content_config={"temperature": 0.0}
        )


    async def _run_agent(self, agent: Agent, prompt: str, result_type: type, max_retries: int = 3) -> Optional[BaseModel]:
            """
            Executes an Agent with built-in resilience.
            
            Features:
            - JSON Parsing: Automatically strips markdown fences from LLM output.
            - Pydantic Validation: Ensures the output matches the expected schema (`result_type`).
            - Exponential Backoff: Retries on 429 (Rate Limit) or 503 (Service Unavailable) errors.

            Args:
                agent (Agent): The ADK agent instance to run.
                prompt (str): The user prompt string.
                result_type (type): The Pydantic class to validate the output against.
                max_retries (int): Number of retry attempts on failure.

            Returns:
                Optional[BaseModel]: An instance of `result_type` if successful, else None.
            """
            last_error = None
            
            for attempt in range(max_retries + 1):
                runner = InMemoryRunner(agent=agent)
                try:
                    response = await runner.run_debug(user_messages=prompt, quiet=True)
                    
                    json_str = ""
                    for event in response[::-1]:
                        if hasattr(event, "content") and hasattr(event.content, "parts"):
                            parts = [p.text for p in event.content.parts if p.text]
                            if parts:
                                json_str = "".join(parts)
                                break
                    
                    if not json_str:
                        raise ValueError("Empty response from model")
                        
                    json_clean = strip_markdown_json_fences(json_str)
                    return result_type.model_validate_json(json_clean)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_overload = "503" in error_msg or "overloaded" in error_msg or "429" in error_msg or "unavailable" in error_msg
                    
                    last_error = e
                    
                    if is_overload and attempt < max_retries:
                        wait_time = 2 * (2 ** attempt) 
                        print(f"   ⏳ {agent.name} hit API limit/overload. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        if not is_overload:
                            print(f"❌ Error in {agent.name}: {e}")
                            return None

            print(f"❌ {agent.name} failed after {max_retries} retries. Last error: {last_error}")
            return None
        
    async def process_invoice(self, invoice_data: Dict[str, Any]) -> Optional[InvoiceLLMResult]:
            """
            Main Orchestration Pipeline for a single invoice.

            Flow:
            1. **Prepare Prompts:** Constructs prompts using raw text and hints from the pre-processor.
            2. **Parallel Execution:** Runs HeaderAgent and LineItemAgent concurrently for speed.
            3. **Self-Correction:**
               - Checks if critical header data (KvK/VAT) is missing.
               - If missing, triggers the **OCR Tool** (Google Cloud Vision).
               - Applies **Tier 1 Extraction** (Regex on OCR text) for low cost/latency.
               - Applies **Tier 2 Extraction** (LLM on OCR text) only if Regex fails.
            4. **Normalization:** Maps OCR typos (e.g., '125') to valid DB codes (e.g., 'I25').
            5. **Validation:** Enforces strict allow-lists for client codes.

            Args:
                invoice_data (Dict[str, Any]): Dictionary containing 'raw_text', 'filename', and hints.

            Returns:
                Optional[InvoiceLLMResult]: The structured, validated invoice data.
            """
            filename = invoice_data.get("filename", "unknown")
            raw_text = invoice_data.get("raw_text", "")
            
            # 1. Prepare Prompts
            hints = {
                "kvk_hint": invoice_data.get("kvk_hint"),
                "vat_hint": invoice_data.get("vat_hint"),
                "invoice_number_hint": invoice_data.get("invoice_number_hint"),
                "invoice_date_hint": invoice_data.get("invoice_date_hint"),
            }
            
            
            # Fallback voor backward compatibility
            default_allowed = invoice_data.get("allowed_client_cases", [])
            
            prompt_allowed_cases = invoice_data.get("allowed_client_cases_prompt", default_allowed)
            valid_allowed_cases = invoice_data.get("allowed_client_cases_valid", default_allowed)
            correction_map = invoice_data.get("correction_map", {})
            # --------------------------------------

            header_prompt = get_header_prompt(raw_text, hints)
            # Gebruik de PROMPT lijst voor de agent (zodat hij typo's herkent die in de tekst staan)
            lines_prompt = get_line_item_prompt(raw_text, prompt_allowed_cases) 

            # 2. Run Agents in Parallel (Async)
            print(f"🤖 Orchestrator: Dispatching agents for {filename}...")
            
            header_agent = self.create_header_agent()
            lines_agent = self.create_line_item_agent()

            header_task = self._run_agent(header_agent, header_prompt, HeaderResult)
            lines_task = self._run_agent(lines_agent, lines_prompt, LineItemsResult)

            header_res, lines_res = await asyncio.gather(header_task, lines_task)

            # 3. Handling Failures & Self-Correction (OCR Tool)
            if not header_res:
                print(f"⚠️ Header Agent failed for {filename}")
                header_res = HeaderResult(invoiceHeader=InvoiceHeader(), isCoachingInvoice=False)

            # --- CAPSTONE TOOL USE: SELF-CORRECTION LOGIC ---
            # Check if critical data is missing (KvK or VAT)
            if header_res and (not header_res.invoiceHeader.kvkNumber or not header_res.invoiceHeader.vatNumber):
                print(f"⚠️ Missing KvK/VAT for {filename}. Orchestrator deciding to use OCR Tool...")
                
                # Construct path to the source PDF
                pdf_path = Path(f"data/invoices/{filename}") 
                
                if pdf_path.exists():
                    print(f"   🛠️ Tool: Running Google Vision OCR on {filename}...")
                    ocr_text = run_pdf_ocr_google(pdf_path)
                    
                    if ocr_text:
                        # --- OBSERVABILITY ---
                        ocr_out_dir = Path("data/llm_ready/ocr_texts")
                        ocr_out_dir.mkdir(exist_ok=True, parents=True)
                        ocr_file = ocr_out_dir / filename.replace(".pdf", "_ocr.txt")
                        
                        with open(ocr_file, "w", encoding="utf-8") as f:
                            f.write(ocr_text)
                        print(f"   💾 Saved OCR dump to: {ocr_file.name}")
                        # -----------------------------------------

                        print(f"   ✅ Tool Output: OCR extracted {len(ocr_text)} characters.")
                        
                        # =========================================================
                        # TIER 1: DETERMINISTIC REGEX EXTRACTION (FAST & CHEAP)
                        # =========================================================
                        print("   ⚡ Orchestrator: Attempting Regex extraction first...")
                        
                        regex_data = InvoiceRegexExtractor.extract_header_fields(ocr_text)
                        
                        if regex_data['kvkNumber'] and not header_res.invoiceHeader.kvkNumber:
                            header_res.invoiceHeader.kvkNumber = regex_data['kvkNumber']
                            print(f"   🎯 REGEX SUCCESS: Found KvK {header_res.invoiceHeader.kvkNumber}")
                        
                        if regex_data['vatNumber'] and not header_res.invoiceHeader.vatNumber:
                            header_res.invoiceHeader.vatNumber = regex_data['vatNumber']
                            print(f"   🎯 REGEX SUCCESS: Found VAT {header_res.invoiceHeader.vatNumber}")

                        # =========================================================
                        # TIER 2: LLM FALLBACK (SMART BUT SLOWER)
                        # =========================================================
                        missing_kvk = not header_res.invoiceHeader.kvkNumber
                        missing_vat = not header_res.invoiceHeader.vatNumber
                        
                        if missing_kvk or missing_vat:
                            print("   🤔 Regex couldn't find everything. Falling back to LLM extraction on OCR text...")
                            
                            ocr_prompt = f"""
                            TASK: RECOVER MISSING DATA (SELF-CORRECTION)
                            
                            Previous attempts (Text & Regex) failed to find:
                            {'- KvK Number' if missing_kvk else ''}
                            {'- VAT Number' if missing_vat else ''}
                            
                            --- BEGIN OCR TEXT ---
                            {ocr_text}
                            --- END OCR TEXT ---
                            
                            INSTRUCTIONS:
                            1. Search the OCR text specifically for 'KvK' or 'BTW' numbers.
                            2. Sometimes OCR adds spaces (e.g. "84 72 61 80"). Try to reconstruct it.
                            3. Return the COMPLETE invoiceHeader JSON. Use existing values where possible, but fill in the blanks using the OCR text.
                            
                            OUTPUT SCHEMA:
                            {{
                            "invoiceHeader": {{
                                "supplierName": "{header_res.invoiceHeader.supplierName}",
                                "invoiceNumber": "{header_res.invoiceHeader.invoiceNumber}",
                                "invoiceDate": "{header_res.invoiceHeader.invoiceDate}",
                                "kvkNumber": "string | null",
                                "vatNumber": "string | null"
                            }},
                            "isCoachingInvoice": {str(header_res.isCoachingInvoice).lower()}
                            }}
                            """
                            
                            new_header_res = await self._run_agent(header_agent, ocr_prompt, HeaderResult)
                            
                            if new_header_res:
                                if new_header_res.invoiceHeader.kvkNumber:
                                    header_res.invoiceHeader.kvkNumber = new_header_res.invoiceHeader.kvkNumber
                                    print(f"   ✨ LLM FIXED: KvK found: {new_header_res.invoiceHeader.kvkNumber}")
                                if new_header_res.invoiceHeader.vatNumber:
                                    header_res.invoiceHeader.vatNumber = new_header_res.invoiceHeader.vatNumber
                                    print(f"   ✨ LLM FIXED: VAT found: {new_header_res.invoiceHeader.vatNumber}")
                        else:
                            print("   🚀 Skipping LLM fallback (Regex found everything needed).")

                else:
                    print(f"   ❌ Could not find PDF at {pdf_path} to run OCR.")
            # --- END TOOL USE ---

            if not lines_res:
                print(f"⚠️ LineItem Agent failed for {filename}")
                lines_res = LineItemsResult()
            else:
                lines_res = self._post_process_line_items(lines_res)

            final_result = InvoiceLLMResult(
                invoiceHeader=header_res.invoiceHeader,
                isCoachingInvoice=header_res.isCoachingInvoice,
                clientCases=lines_res.clientCases,
                clientCasesNoActivity=lines_res.clientCasesNoActivity
            )

            # --- APPLY CORRECTIONS (125 -> I25) ---
            # Dit zet de typo's die de agent heeft gevonden om naar geldige database codes
            final_result = apply_client_case_corrections(final_result, correction_map)
            # --------------------------------------

            # Post-processing enforcement (Business Logic - Strict Validation op CORRECTE codes)
            final_result = enforce_allowed_client_cases(final_result, valid_allowed_cases)
            
            return final_result
 
    def _post_process_line_items(self, lines_result: LineItemsResult) -> LineItemsResult:
            """
            Cleans up line items.
            Moves cases with 0 hours or None duration to the 'NoActivity' list.
            """
            cleaned_active = []
            no_activity_set = set(lines_result.clientCasesNoActivity)

            for case in lines_result.clientCases:
                if case.durationHours is None or case.durationHours == 0:
                    # UPDATE: Check validatedClientCaseNumber instead of clientCaseNumber
                    if case.validatedClientCaseNumber:
                        no_activity_set.add(case.validatedClientCaseNumber)
                else:
                    cleaned_active.append(case)
            
            lines_result.clientCases = cleaned_active
            lines_result.clientCasesNoActivity = sorted(list(no_activity_set))
            return lines_result
    
# --- Main Execution ---

async def main():
    base_dir = Path("data/llm_ready")
    output_dir = base_dir / "json_out_multi_agent"
    output_dir.mkdir(parents=True, exist_ok=True)    

    from utils import load_all_coaching_invoices
    
    print("📂 Loading invoices...")
    examples = load_all_coaching_invoices(base_dir)
    
    if not examples:
        print("No invoices found to process.")
        return

    orchestrator = InvoiceOrchestrator()
    
    success_count = 0
    
    total_invoices = len(examples)
    
    for index, example in enumerate(examples):
        filename = example['filename']
        print(f"\nProcessing: {filename}")
        
        result = await orchestrator.process_invoice(example)
        
        if result:
            out_path = output_dir / filename.replace(".pdf", "_parsed.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            print(f"✅ Success! Saved to {out_path.name}")
            print(f"   Supplier: {result.invoiceHeader.supplierName}")
            print(f"   Cases Found: {len(result.clientCases)}")
            success_count += 1
        else:
            print("❌ Failed to process.")

        # Smart Rate Limit: Only sleep if there are more items to process
        if index < total_invoices - 1:
            print("   💤 Cooling down for 10s to avoid Rate Limit...")
            await asyncio.sleep(10)
        else:
            print("   🚀 Batch complete. Skipping final cooldown.")
            

    await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
