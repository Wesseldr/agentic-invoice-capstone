#!/bin/bash
# ============================================================================
# File: demo.sh
# Author: J.W. de Roode
# Description:
#     One-click demonstration script for the Agentic Invoice Processor.
#     Executes the full pipeline: Processing -> Extraction -> Evaluation.
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

echo "============================================================"
echo "🎬 STARTING CAPSTONE DEMO: Agentic Invoice Processor"
echo "============================================================"

# 1. Clean up previous run data to prove reproducibility
echo ""
echo "🧹 Cleaning up workspace (removing old output)..."
rm -rf data/llm_ready/*
# Ensure the directory structure exists so python doesn't complain
mkdir -p data/llm_ready
echo "✅ Workspace clean."

# 2. Run Phase 1: The Processor
echo ""
echo "🚀 [PHASE 1] STARTING: Universal Processing & Validation..."
echo "------------------------------------------------------------"
python run.py
echo "------------------------------------------------------------"
echo "✅ [PHASE 1] COMPLETE."

# 3. Run Phase 2: The Agents
echo ""
echo "🤖 [PHASE 2] STARTING: Multi-Agent Extraction (Gemini + OCR)..."
echo "------------------------------------------------------------"
python src/capstone_agents.py
echo "------------------------------------------------------------"
echo "✅ [PHASE 2] COMPLETE."

# 4. Run Phase 3: Evaluation
echo ""
echo "📊 [PHASE 3] STARTING: Final Accuracy Evaluation..."
echo "------------------------------------------------------------"
python evaluate_capstone.py
echo "------------------------------------------------------------"
echo "✅ [PHASE 3] COMPLETE."

echo ""
echo "🎉 DEMO FINISHED SUCCESSFULLY!"
