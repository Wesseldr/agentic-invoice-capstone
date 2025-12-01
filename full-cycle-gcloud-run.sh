#!/bin/bash
# ============================================================================
# File: full-cycle-gcloud-run.sh
# Author: J.W. de Roode
# GitHub: https://github.com/Wesseldr
# Project: Agentic Invoice Processor – Google AI 5-Day Intensive
# Created: 2025-12-01
# Description:
#     Turn-key automation script for Cloud Deployment. Handles environment
#     checks, Docker builds, Cloud Run deployment, and live log streaming.
# ============================================================================

# ==============================================================================
# AGENTIC INVOICE PROCESSOR - FULL CYCLE CLOUD DEPLOYMENT
# ------------------------------------------------------------------------------
# This script automates the entire lifecycle:
# 1. Environment Checks & Configuration
# 2. Enabling required Google Cloud APIs
# 3. Building the Docker Container (Cloud Build)
# 4. Deploying the Cloud Run Job
# 5. Executing the Job & Streaming Logs
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.
JOB_NAME="invoice-processor-job"
REGION="europe-west4" # You can change this default or make it interactive

echo "============================================================"
echo "☁️  STARTING FULL-CYCLE CLOUD DEPLOYMENT"
echo "============================================================"

# --- 1. CONFIGURATION CHECK ---

# Get Project ID from gcloud config
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [[ -z "$PROJECT_ID" ]]; then
    echo "❌ Error: No active Google Cloud Project detected."
    echo "   Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "   🎯 Target Project: $PROJECT_ID"
echo "   🌍 Target Region:  $REGION"

# Check for API Key
API_KEY=""

# Option 1: Check local .env
if [ -f ".env" ]; then
    API_KEY=$(grep GOOGLE_API_KEY .env | cut -d '=' -f2 | tr -d '"' | tr -d "'")
fi

# Option 2: Check home dir .env
if [[ -z "$API_KEY" ]] && [ -f "$HOME/.env" ]; then
    API_KEY=$(grep GOOGLE_API_KEY "$HOME/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'")
fi

# Validation
if [[ -z "$API_KEY" ]]; then
    echo "❌ Error: GOOGLE_API_KEY not found in .env or ~/.env"
    exit 1
fi

echo "   🔑 API Key:        Detected (will be securely injected)"

echo ""
echo "   ⚠️  NOTE: This process requires the following APIs to be enabled:"
echo "      - Cloud Build API"
echo "      - Cloud Run API"
echo "      - Artifact Registry API"
echo "   (The script will attempt to enable them for you...)"
echo ""

# --- 2. SETUP ENVIRONMENT ---

echo ""
echo "🛠️  Configuring Project Environment..."

# Set quota project to fix common "403 permission denied" errors on new projects
echo "   [1/3] Setting quota project..."
gcloud auth application-default set-quota-project $PROJECT_ID --quiet > /dev/null 2>&1

echo "   [2/3] Enabling required APIs (this may take a moment)..."
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    --quiet

# --- 3. BUILD CONTAINER ---

IMAGE_NAME="gcr.io/$PROJECT_ID/agentic-invoice-demo"

echo ""
echo "🏗️  Building Container Image..."
echo "   Destination: $IMAGE_NAME"
echo "   --------------------------------------------------------"
# Using --quiet to reduce noise, remove it if you want to see build logs
gcloud builds submit --tag $IMAGE_NAME --quiet

# --- 4. DEPLOY TO CLOUD RUN ---

echo ""
echo "🚀 Deploying Cloud Run Job..."

# Delete existing job to ensure a clean slate (updates can sometimes be tricky)
echo "   [1/2] Cleaning up old job (if any)..."
gcloud run jobs delete $JOB_NAME --region $REGION --quiet 2>/dev/null || true

echo "   [2/2] Creating new job '$JOB_NAME'..."
gcloud run jobs create $JOB_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --set-env-vars GOOGLE_API_KEY="$API_KEY" \
  --max-retries 0 \
  --task-timeout 10m \
  --memory 1Gi \
  --quiet

# --- 5. EXECUTE & STREAM LOGS ---

echo ""
echo "▶️  Executing Job..."
# Start execution async so we can grab the ID immediately
gcloud run jobs execute $JOB_NAME --region $REGION --async

echo "   ⏳ Waiting for job to initialize..."
sleep 5

# Get the execution ID of the job we just started
LATEST_EXEC=$(gcloud run jobs executions list --job $JOB_NAME --region $REGION --limit 1 --format="value(name)")

if [[ -z "$LATEST_EXEC" ]]; then
    echo "❌ Error: Could not determine execution ID. Please check Cloud Console."
    exit 1
fi

echo "   📡 Streaming logs from Execution: $LATEST_EXEC"
echo "   --------------------------------------------------------"

# Stream logs (FIXED COMMAND)
gcloud beta run jobs executions logs tail $LATEST_EXEC --region $REGION

echo ""
echo "============================================================"
echo "✅ FULL CYCLE COMPLETE"
echo "============================================================"