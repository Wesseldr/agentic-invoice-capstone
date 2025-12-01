# ============================================================================
# File: regex_tools.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Shared library of compiled regular expressions for deterministic
#     extraction of patterns like dates, amounts, and client codes.
# ============================================================================

"""
Regex Utilities Module
======================

This module serves as a centralized library for **Deterministic Pattern Extraction**.
It provides compiled regular expressions and normalization logic to extract structured
data (dates, amounts, identifiers) from raw invoice text *before* AI processing.

Key Features:
- **Robust Client Case Detection:** Uses a permissive pattern (`A-Z0-9`) to capture codes even if they contain OCR errors (like 1/I swaps).
- **Multi-Format Support:** Handles various date formats (NL/ISO) and number formats (EU/US).
- **VAT/KvK Validation:** Detects Dutch administrative numbers in various layouts.

Classes:
    InvoiceRegexExtractor: Static utility class containing patterns and helper methods.
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class InvoiceRegexExtractor:
    """
    Shared logic for extracting patterns from invoice text.
    Used by both the Universal Processor (Batch) and the Agent Orchestrator (Runtime).

    This class uses a 'Tier 1' extraction strategy: fast, free, and deterministic.
    """
    # --- PATTERNS ---
    PATTERNS = {
        'client_cases': [
            re.compile(r'(?i)\b([A-Z0-9]{2}\d{1,2}-[A-Z0-9]\d{2}-\d{3})\b'),
            #re.compile(r'(?i)(?:client|case|dossier|cliënt)[:\s#-]*([A-Z0-9-]{6,})', re.IGNORECASE),
        ],
        'invoice_numbers': [
            re.compile(r'(?i)\b(?:factuurnummer|factuur ?nr\.?|invoice.?number)\b[:\s]+([A-Z0-9./-]+)'),
            re.compile(r'(?i)Factuurnummer:\s*\n(?:.*\n)?\s*([0-9]{3,})'),
            re.compile(r'(?i)\bBetreft:\s*facturen?\s+(\d{3,})'),
            re.compile(r'(?i)\b(?:ref|reference)\b[:\s#-]*([A-Z0-9./-]+)'),
            re.compile(r'(?i)\bFact\.:\s*([A-Z0-9._-]+)')
        ],
        'dates': [
            re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'),
            re.compile(r'\b(\d{1,2}\s+\w+\s+\d{4})\b'),
            re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'),
        ],
        'amounts': [
            re.compile(r'€\s*([\d.,]+)'),
            re.compile(r'EUR\s*([\d.,]+)'),
            re.compile(r'\$\s*([\d.,]+)'),
            re.compile(r'(?:total|totaal|amount|bedrag)[:\s]*(?:€|\$|EUR)?\s*([\d.,]+)', re.IGNORECASE),
        ],
        'emails': [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ],
        'vat_numbers': [
            re.compile(r'\b(NL\s*(?:\d[.\s]?){9}B\.?\s*\d{2})\b', re.IGNORECASE),
            re.compile(r'(?i)\b(?:btw|omzetbelastingnummer|ob-nummer|ob nummer)\b.*?(\d{9}\s*B\s*\d{2})'),
            re.compile(r'(?i)(?:btw(?:-id)?|btw ?nr\.?|vat(?: ?id)?|tax(?: ?id)?)[:\s#-]*([A-Z0-9.\-]+)'),
        ],
        'kvk_numbers': [
            re.compile(r'(?i)\bkvk(?:[-\s]?nummer|nr\.?)?(?:\s+\w+)?[:\s#-]*([0-9]{7,8})'),
            re.compile(r'(?i)kvk.*?(\d{8})'), 
        ],
    }

    VAT_NL_CANON = re.compile(r'^NL\d{9}B\d{2}$', re.IGNORECASE)
    VAT_EU_GENERIC = re.compile(r'^[A-Z]{2}\d{8,12}$')
    
    MONTHS_NL = {
        "januari": 1, "februari": 2, "maart": 3, "april": 4, "mei": 5, "juni": 6,
        "juli": 7, "augustus": 8, "september": 9, "oktober": 10, "november": 11, "december": 12,
    }

    @classmethod
    def extract_all(cls, text: str) -> Dict[str, Any]:
        """
        Runs all configured regex patterns against the provided text.

        Args:
            text (str): The raw input string (e.g., from PDF or OCR).

        Returns:
            Dict[str, Any]: A dictionary where keys match the PATTERNS keys (e.g., 'client_cases', 'amounts')
                            and values are lists of extracted, normalized data.
        """
        found = {}
        for pattern_name, pattern_list in cls.PATTERNS.items():
            matches = []
            for pattern in pattern_list:
                matches.extend(pattern.findall(text))
            
            if pattern_name == 'amounts':
                found[pattern_name] = cls._parse_amounts(matches)
            elif pattern_name == 'dates':
                raw_dates = cls._dedupe(matches)
                found['dates_raw'] = raw_dates
                found[pattern_name] = cls._normalize_dates(raw_dates)
            elif pattern_name == 'vat_numbers':
                found[pattern_name] = cls._parse_vat(matches)
            else:
                found[pattern_name] = cls._dedupe(matches)
        
        return found

    @classmethod
    def extract_header_fields(cls, text: str) -> Dict[str, Optional[str]]:
        """
        Targeted extraction for the Orchestrator (Tier 1 Fallback).
        Returns the 'best guess' for critical header fields only.
        """
        all_data = cls.extract_all(text)
        return {
            "kvkNumber": all_data['kvk_numbers'][0] if all_data.get('kvk_numbers') else None,
            "vatNumber": all_data['vat_numbers'][0] if all_data.get('vat_numbers') else None,
            "invoiceNumber": all_data['invoice_numbers'][0] if all_data.get('invoice_numbers') else None,
        }

    # --- INTERNAL HELPERS ---

    @staticmethod
    def _dedupe(matches: List[Any]) -> List[str]:
        """Removes duplicates while preserving order."""     
        seen = set()
        unique = []
        for item in matches:
            val = item[0] if isinstance(item, tuple) else item
            val = val.strip() if val else ""
            if val and val not in seen:
                seen.add(val)
                unique.append(val)
        return unique

    @classmethod
    def _parse_amounts(cls, matches: List[Any]) -> List[float]:
        """Parses currency strings (EU/US formats) into floats."""
        parsed = []
        for amount in matches:
            if isinstance(amount, tuple): amount = amount[0]
            if not amount: continue
            try:
                clean = amount.replace(' ', '')
                if ',' in clean and '.' in clean:
                    if clean.rindex(',') > clean.rindex('.'): 
                        clean = clean.replace('.', '').replace(',', '.')
                    else: 
                        clean = clean.replace(',', '')
                elif ',' in clean:
                    if len(clean.split(',')[1]) == 2: clean = clean.replace(',', '.')
                    else: clean = clean.replace(',', '')
                parsed.append(float(clean))
            except: continue
        return list(dict.fromkeys(parsed))

    @classmethod
    def _parse_vat(cls, matches: List[Any]) -> List[str]:
        """Normalizes extracted VAT numbers."""
        candidates = []
        for item in matches:
            raw = item[0] if isinstance(item, tuple) else item
            norm = cls.normalize_vat_number(raw)
            if norm: candidates.append(norm[0])
        return list(dict.fromkeys(candidates))

    @classmethod
    def normalize_vat_number(cls, raw_value: str) -> Optional[Tuple[str, str]]:
        """Cleans VAT strings to standard formats (e.g., removing separators)."""
        if not raw_value: return None
        cleaned = re.sub(r'[\s.\-]', '', raw_value.upper())
        cleaned = re.sub(r'^(BTWID|BTWNR|BTW|VATID|VATNR|VAT|TAXID|TAXNR|TAX)', '', cleaned)
        if cls.VAT_NL_CANON.match(cleaned):
            return (f"NL{cleaned[2:11]}B{cleaned[-2:]}", 'nl')
        if cls.VAT_EU_GENERIC.match(cleaned):
            return (cleaned, 'eu')
        return None

    @classmethod
    def _normalize_dates(cls, raw_dates: List[str]) -> List[str]:
        normalized = []
        for d in raw_dates:
            iso = cls.normalize_date(d)
            if iso: normalized.append(iso)
        return list(dict.fromkeys(normalized))

    @classmethod
    def normalize_date(cls, raw: str) -> Optional[str]:
        """Converts various date string formats to ISO 8601 (YYYY-MM-DD)."""
        if not raw: return None
        s = raw.strip()
        numeric_formats = ["%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y", "%Y-%m-%d", "%Y/%m/%d"]
        for fmt in numeric_formats:
            try: return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except ValueError: pass
        
        m = re.match(r"^\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*$", s)
        if m:
            day, month, year = m.groups()
            month_num = cls.MONTHS_NL.get(month.lower())
            if month_num:
                try: return datetime(int(year), month_num, int(day)).strftime("%Y-%m-%d")
                except ValueError: pass
        return None