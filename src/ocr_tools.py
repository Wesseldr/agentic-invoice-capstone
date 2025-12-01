# ============================================================================
# File: ocr_tools.py
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-11-30
# Description:
#     Wrapper for the Google Cloud Vision API. Handles PDF-to-image conversion
#     and OCR text extraction for the self-correction pipeline.
# ============================================================================

"""
OCR Utilities Module
====================

This module provides a wrapper around the **Google Cloud Vision API** to extract text from 
PDF documents that contain scanned images or complex layouts (where standard text extraction fails).

It handles the full pipeline:
1. Converting a PDF page to an image (using `pdf2image`).
2. Sending the image to the Google Cloud Vision API.
3. Returning the raw text detected in the image.

Dependencies:
    - `google-cloud-vision`: For the OCR API.
    - `pdf2image` & `poppler`: For converting PDF pages to bytes.
    - If dependencies are missing, it gracefully falls back to a MOCK response for testing purposes.
"""
import io
import os
from pathlib import Path
from typing import Optional

# Probeer imports, faal vriendelijk als libraries missen
try:
    from google.cloud import vision
    from pdf2image import convert_from_path
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("⚠️ Warning: 'google-cloud-vision' or 'pdf2image' not installed. OCR will be mocked.")

def run_pdf_ocr_google(pdf_path: Path) -> str:
    """
    Executes Optical Character Recognition (OCR) on the first page of a PDF.

    This function is triggered by the Orchestrator when critical metadata (like KvK) is missing
    from the raw text layer. It renders the PDF as an image and asks Google Vision to read it.

    Args:
        pdf_path (Path): The file path to the PDF document.

    Returns:
        str: The full raw text string extracted from the image. 
             Returns an empty string if the process fails.
             Returns a specific mock string if dependencies are not installed.
    """
    if not HAS_DEPENDENCIES:
        return "[MOCK OCR RESULT] KvK nummer: 84726180 BTW Nr: NL863334647B01"

    # 1. Convert PDF to Image (First page is usually enough for Header info)
    try:
        images = convert_from_path(str(pdf_path), first_page=1, last_page=1)
        if not images:
            return ""
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()
    except Exception as e:
        print(f"❌ PDF to Image failed: {e}")
        return ""

    # 2. Call Google Vision API
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        
        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            # texts[0].description contains the full text block
            return texts[0].description
        return ""
        
    except Exception as e:
        print(f"❌ Google Vision API failed: {e}")
        # Fallback for demo purposes if API fails (e.g. auth issues)
        return ""