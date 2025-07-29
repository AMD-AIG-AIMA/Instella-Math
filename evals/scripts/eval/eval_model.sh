set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"  # Add default output directory
DATASET_DIR=example
SEQ_LEN=28000
N_RESPOSES=16
# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --n-responses)
            N_RESPOSES="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --dataset-dir <dataset directory path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Dataset Directory: ${DATASET_DIR}"
echo "Sequence Len: ${SEQ_LEN}"
echo "Number of resposes: ${N_RESPOSES}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=./deepscaler/${DATASET_DIR}/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=${N_RESPOSES} \
        data.batch_size=512 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        +model.trust_remote_code=True \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.response_length=${SEQ_LEN} \
        rollout.gpu_memory_utilization=0.9
        # rollout.enable_chunked_prefill=True
done
