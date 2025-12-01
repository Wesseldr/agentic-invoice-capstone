# ============================================================================
# File: clientcase_matcher.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Logic for validating and normalizing client case numbers against a
#     reference database. Handles fuzzy matching for OCR error correction.
# ============================================================================

"""
Client Case Matcher Module
==========================

This module provides the logic to validate and normalize client case numbers against a reference database.
It is the core component responsible for the **Data Quality Gatekeeper** pattern in the pipeline.

Key Features:
- **Exact Matching:** Verifies if a code exists directly in the database.
- **Fuzzy Correction:** Automatically detects and fixes common OCR errors (e.g., swapping '1' for 'I' or '0' for 'O') based on the canonical format.
- **Ambiguity Detection:** Flags codes that could match multiple valid entries as ambiguous to prevent incorrect assignments.

Classes:
    MatchResult: Data model for the result of a matching operation.
    ClientCaseMatcher: The logic engine that performs the matching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Literal
import csv

MatchStatus = Literal["exact", "fuzzy_io_swap", "ambiguous_io_swap", "unknown"]
Contamination = Literal["none", "I_only", "O_only", "I_and_O"]


@dataclass
class MatchResult:
    """
    Data model representing the outcome of a client case number lookup.

    Attributes:
        original_code (str): The raw string found on the invoice.
        matched_code (Optional[str]): The official code from the database if a match was found, else None.
        match_status (MatchStatus): The verdict of the matching process (e.g., 'exact', 'fuzzy_io_swap').
        match_confidence (float): A score (0.0 - 1.0) indicating certainty.
        contamination (Contamination): Flags if the original code contained 'I' or 'O' characters (often OCR noise).
        candidates (List[str]): List of potential valid codes if multiple matches were found (for debugging).
        notes (Optional[str]): Debugging information or reason for rejection.
    """
    original_code: str
    matched_code: Optional[str]          # official code from the index, or None
    match_status: MatchStatus
    match_confidence: float
    contamination: Contamination         # based on matched_code (or original if no match)
    candidates: List[str]                # all possible matches for fuzzy/ambiguous cases
    notes: Optional[str] = None


class ClientCaseMatcher:
    """
    Logic for validating and correcting Client Case Numbers against an official registry.

    This class handles:
    1. Loading the allow-list from a CSV file (`valid_clientcasenumbers.csv`).
    2. Normalizing input codes (handling common OCR errors like 1/I and 0/O swaps).
    3. Matching raw inputs to official codes using Exact and Fuzzy logic.

    Attributes:
        index_csv_path (Path): Path to the CSV file containing valid codes.
        valid_codes (Set[str]): A set of all valid codes for O(1) fast lookup.
        index_by_canon (Dict[str, List[str]]): A lookup map where keys are 'canonical' (normalized) codes
                                              and values are lists of actual valid codes.
    """

    def __init__(self, index_csv_path: Path):
        """
        Initializes the matcher by loading the reference index into memory.

        Args:
            index_csv_path (Path): Path to the 'valid_clientcasenumbers.csv' file.
        """
        self.index_csv_path = index_csv_path
        self.valid_codes: Set[str] = set()
        self.index_by_canon: Dict[str, List[str]] = {}
        self._load_index()

    # ---------- loading & indexing ----------

    def _load_index(self) -> None:
        """
        Loads valid codes from the CSV file into memory structures.
        
        It populates both the exact set (`valid_codes`) and the fuzzy index (`index_by_canon`).

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is empty or missing headers.
        """
        if not self.index_csv_path.exists():
            raise FileNotFoundError(f"Clientcase index not found: {self.index_csv_path}")

        with self.index_csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # We expect a 'clientCaseNumber' column
            col = "clientCaseNumber"
            if reader.fieldnames is None or col not in reader.fieldnames:
                # fallback: take the first column
                header = reader.fieldnames or []
                if not header:
                    raise ValueError(f"No columns found in clientcase index: {self.index_csv_path}")
                col = header[0]

            for row in reader:
                code = (row.get(col) or "").strip()
                if not code:
                    continue
                self.valid_codes.add(code)
                canon = self.canonical_case_number(code)
                self.index_by_canon.setdefault(canon, []).append(code)

    # ---------- canonical helpers ----------

    @staticmethod
    def canonical_case_number(code: str) -> str:
        """
        Converts a code to its 'canonical' form by normalizing ambiguous characters.
        
        Logic:
        - Based on the known format LLDD-LDD-DDD (Letter/Digit structure).
        - Checks specific positions (0, 1, 5) where letters are expected.
        - Swaps 'I' -> '1' and 'O' -> '0' at these positions to handle OCR misinterpretations.
        - Other positions are left untouched.

        Args:
            code (str): The input code string.

        Returns:
            str: The normalized (canonical) code string used for fuzzy matching.
        """
        c = (code or "").strip().upper()
        if len(c) < 11:
            # Leave it to later validation to decide this is wrong
            return c

        chars = list(c)
        letter_positions = [0, 1, 5]

        for pos in letter_positions:
            if pos < len(chars):
                if chars[pos] == "I":
                    chars[pos] = "1"
                elif chars[pos] == "O":
                    chars[pos] = "0"

        return "".join(chars)

    @staticmethod
    def contamination_flag(code: str) -> Contamination:
        """
        Detects if the code contains potentially problematic characters ('I' or 'O').
        This is used for reporting/debugging OCR quality.

        Args:
            code (str): The input code.

        Returns:
            Contamination: A literal indicating if I, O, both, or neither are present.
        """
        c = (code or "").upper()
        has_i = "I" in c
        has_o = "O" in c

        if not has_i and not has_o:
            return "none"
        if has_i and has_o:
            return "I_and_O"
        if has_i:
            return "I_only"
        return "O_only"

    # ---------- matching API ----------

    def match(self, code: str) -> MatchResult:
        """
        The core logic: Attempts to match a raw input string to a valid client case number.

        Matching Strategy:
        1. **Exact Match:** If the code exists in the DB as-is -> 'exact'.
        2. **Fuzzy Match:** If the canonical form matches exactly one valid code -> 'fuzzy_io_swap'.
        3. **Ambiguous:** If the canonical form matches multiple valid codes -> 'ambiguous_io_swap'.
        4. **Unknown:** If no match is found -> 'unknown'.

        Args:
            code (str): The raw string found on the invoice (e.g. "JN16-121-284").

        Returns:
            MatchResult: Object containing the best match, status, and metadata.
        """
        original = (code or "").strip()
        if not original:
            return MatchResult(
                original_code=original,
                matched_code=None,
                match_status="unknown",
                match_confidence=0.0,
                contamination="none",
                candidates=[],
                notes="Empty or whitespace-only clientCaseNumber",
            )

        # 1) Exact match
        if original in self.valid_codes:
            return MatchResult(
                original_code=original,
                matched_code=original,
                match_status="exact",
                match_confidence=1.0,
                contamination=self.contamination_flag(original),
                candidates=[original],
                notes=None,
            )

        # 2) Canonical match
        canon = self.canonical_case_number(original)
        candidates = self.index_by_canon.get(canon, [])

        if not candidates:
            # No matches
            return MatchResult(
                original_code=original,
                matched_code=None,
                match_status="unknown",
                match_confidence=0.0,
                contamination=self.contamination_flag(original),
                candidates=[],
                notes="clientCaseNumber is not registered",
            )

        if len(candidates) == 1:
            # One clear candidate -> suspected I/0 swap
            matched = candidates[0]
            return MatchResult(
                original_code=original,
                matched_code=matched,
                match_status="fuzzy_io_swap",
                match_confidence=0.75,
                contamination=self.contamination_flag(matched),
                candidates=candidates,
                notes="Matched via I/1 or O/0 canonical mapping",
            )

        # Multiple candidates → ambiguous
        return MatchResult(
            original_code=original,
            matched_code=None,
            match_status="ambiguous_io_swap",
            match_confidence=0.3,
            contamination="none",
            candidates=candidates,
            notes="Multiple possible matches for canonical form",
        )
