#!/bin/bash
# ============================
# LLMTailor Merge Script
# ============================
# Usage: bash start_merge.sh [OPTIONS]
# Run with --help for full options list

# ============================
# Default values
# ============================
OUTPUT_PATH=""
LORA_MERGE_CACHE="/tmp/llmtailor_cache"
CONFIG_YML=""
COPY_TOKENIZER=true
LAZY_UNPICKLE=false
LOW_CPU_MEMORY=false
NUM_GPU=8
BASE_STEP=0
FAILURE_STEP=160
TOTAL_LAYERS=28
STREAMCHECK_JSON=""
STREAMCHECK_PATH=""
DRY_RUN=false

# ============================
# Parse command line arguments
# ============================
show_help() {
    cat << EOF
Usage: bash start_merge.sh [OPTIONS]

Options:
  --output_path PATH        Output path for merged model (required for merge)
  --config_yml PATH         Path to YAML merge config
  --streamcheck_json PATH   Path to checkpoint_flags.jsonl from StreamCheck
  --streamcheck_path PATH   Path to checkpoints directory
  --base_step NUM           Base checkpoint step (layers with ckpt <= base use this)
  --failure_step NUM        Failure step (find each layer's nearest ckpt before this)
  --total_layers NUM        Number of transformer layers (default: 28)
  --num_gpu NUM             Number of GPUs to use (default: 8)
  --lora_merge_cache PATH   Cache directory (default: /tmp/llmtailor_cache)
  --copy_tokenizer BOOL     Copy tokenizer to output (default: true)
  --lazy_unpickle BOOL      Low-memory model loader (default: false)
  --low_cpu_memory BOOL     Use when VRAM > RAM (default: false)
  --dry_run                 Only generate YAML, skip merge
  --weight_tie BOOL         Weight tie (default: false)
  --help                    Show this help message

Examples:
  # Mode 1: Use existing YAML config
  bash start_merge.sh --config_yml config.yaml --output_path /path/to/output

  # Mode 2: Generate YAML only (dry run)
  bash start_merge.sh --streamcheck_json logs/flags.jsonl --streamcheck_path /ckpts --config_yml out.yaml --dry_run

  # Mode 3: One-stop service
  bash start_merge.sh --streamcheck_json logs/flags.jsonl --streamcheck_path /ckpts --config_yml out.yaml --output_path /out --failure_step 500
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --lora_merge_cache)
            LORA_MERGE_CACHE="$2"
            shift 2
            ;;
        --config_yml)
            CONFIG_YML="$2"
            shift 2
            ;;
        --copy_tokenizer)
            COPY_TOKENIZER="$2"
            shift 2
            ;;
        --lazy_unpickle)
            LAZY_UNPICKLE="$2"
            shift 2
            ;;
        --low_cpu_memory)
            LOW_CPU_MEMORY="$2"
            shift 2
            ;;
        --num_gpu)
            NUM_GPU="$2"
            shift 2
            ;;
        --base_step)
            BASE_STEP="$2"
            shift 2
            ;;
        --failure_step)
            FAILURE_STEP="$2"
            shift 2
            ;;
        --total_layers)
            TOTAL_LAYERS="$2"
            shift 2
            ;;
        --streamcheck_json)
            STREAMCHECK_JSON="$2"
            shift 2
            ;;
        --streamcheck_path)
            STREAMCHECK_PATH="$2"
            shift 2
            ;;
        --weight_tie)
            WEIGHT_TIE="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# ============================
# Validation
# ============================
if [ -z "$CONFIG_YML" ]; then
    echo "Error: --config_yml is required"
    exit 1
fi

if [ "$DRY_RUN" = false ] && [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output_path is required when not using --dry_run"
    exit 1
fi

# ============================
# Build and run command
# ============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="python ${SCRIPT_DIR}/start_merge.py \
    --config_yml \"$CONFIG_YML\" \
    --copy_tokenizer \"$COPY_TOKENIZER\" \
    --lazy_unpickle \"$LAZY_UNPICKLE\" \
    --low_cpu_memory \"$LOW_CPU_MEMORY\" \
    --num_gpu \"$NUM_GPU\" \
    --base_step \"$BASE_STEP\" \
    --failure_step \"$FAILURE_STEP\" \
    --total_layers \"$TOTAL_LAYERS\""

# Add optional arguments
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
fi

if [ -n "$LORA_MERGE_CACHE" ]; then
    CMD="$CMD --lora_merge_cache \"$LORA_MERGE_CACHE\""
fi

if [ -n "$STREAMCHECK_JSON" ]; then
    CMD="$CMD --streamcheck_json \"$STREAMCHECK_JSON\""
fi

if [ -n "$STREAMCHECK_PATH" ]; then
    CMD="$CMD --streamcheck_path \"$STREAMCHECK_PATH\""
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry_run"
fi

if [ -n "$WEIGHT_TIE" ]; then
    CMD="$CMD --weight_tie \"$WEIGHT_TIE\""
fi

echo "Running: $CMD"
eval $CMD
