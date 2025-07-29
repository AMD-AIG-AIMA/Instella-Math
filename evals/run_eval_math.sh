export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_LOCAL_RANK=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export HF_HOME="<PATH_TOHF_CACHE>"
export HF_TOKEN="<HF_TOKEN>"
 
# Tasks: aime aime25 math amc olympiad_bench minerva gsm8k gpqa

bash ./scripts/eval/eval_model.sh \
    --model Qwen/Qwen2.5-Math-1.5B \
    --datasets aime aime25 math amc olympiad_bench minerva gsm8k \
    --dataset-dir data_processed \
    --output-dir ./results_math/Qwen2.5-Math-1.5B

bash ./scripts/eval/eval_model.sh \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets aime aime25 math amc olympiad_bench minerva gsm8k gpqa \
    --dataset-dir data_processed \
    --output-dir ./results_math/DeepSeek-R1-Distill-Qwen-1.5B

bash ./scripts/eval/eval_model.sh \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --datasets aime aime25 math amc olympiad_bench minerva gsm8k gpqa \
    --dataset-dir data_processed \
    --output-dir ./results_math/DeepScaleR-1.5B-Preview

bash ./scripts/eval/eval_model.sh \
    --model RUC-AIBOX/STILL-3-1.5B-preview \
    --datasets aime aime25 math amc olympiad_bench minerva gsm8k gpqa \
    --dataset-dir data_processed \
    --output-dir ./results_math/STILL-3-1.5B-preview

bash ./scripts/eval/eval_model.sh \
    --model amd/Instella-3B-Math \
    --datasets aime aime25 math amc olympiad_bench minerva gsm8k gpqa \
    --dataset-dir data_processed \
    --output-dir ./results_math/Instella-3B-Math
