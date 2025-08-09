<div align="center">
  <br>
  <br>
  <h1>Instella-Math✨: Fully Open Language Model with Reasoning Capability</h1>
<a href='https://huggingface.co/amd/Instella-3B-Math'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/instella-math-language/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>

## Getting Started

### Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "amd/Instella-3B-Math"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)

prompt = [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer within \\boxed{}."}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.8,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))
```

## Supervised Fine-tuning (SFT)
We conducted two-stage math SFT to enhance the math capabilities of the base [Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct) model.
### Stage 1
We use the [Instella](https://github.com/AMD-AIG-AIMA/Instella) codebase for the stage 1 math SFT, where the model is trained on the [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset with 4096 context length. Please follow the [installation guide](https://github.com/AMD-AIG-AIMA/Instella?tab=readme-ov-file#installation) to set up the environment.

Run the following commands to prepare the stage 1 math SFT data:
```bash
git clone https://github.com/AMD-AIG-AIMA/Instella.git
cd Instella
bash scripts/prepare_math_sft_data.sh
```

Launch the SFT job with the [SFT config file](https://github.com/AMD-AIG-AIMA/Instella/blob/main/configs/instella-3b-sft-math-stage1.yaml):

```
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-sft-math-stage1.yaml
```

Note: You need to convert the Huggingface [Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct) checkpoint to PyTorch format and then update `load_path` in the config file to the converted model checkpoint. Please see the instruction for checkpoint conversion [here](https://github.com/AMD-AIG-AIMA/Instella/tree/instella-long?tab=readme-ov-file#base-model-preparation).

### Stage 2

In the stage 2 math SFT, we continue to train the model on the English subset of the [AM-DeepSeek-R1-Distilled-1.4M](https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M) dataset with 1.3M samples, and increase the context length to 32K. We provide a [script](./sft/prepare_ammath_dataset.py) to prepare the dataset for training. The training is based on [open-instruct](https://github.com/allenai/open-instruct/tree/bcb991d4d9b297dc301e03ebaaa5d80dd76bb384/). To run the stage 2 math SFT training:

```
cd sft

bash scripts/finetune_with_accelerate_config_stage3.sh configs/train_configs/instella/instella-3b-sft-math-stage2.yaml
```
Note: Please update `model_name_or_path` in the [config](./sft/configs/train_configs/instella/instella-3b-sft-math-stage2.yaml) to your stage 1 math SFT model. You need to convert the checkpoint to the Huggingface format (see the instruction [here](https://github.com/AMD-AIG-AIMA/Instella/tree/instella-long?tab=readme-ov-file#direct-preference-optimization-dpo)). In addition, please update `dataset_name` to the Huggingface repo of your processed dataset.

## Reinforcement Learning (GRPO)
We conduct GRPO after SFT using [VERL](https://github.com/volcengine/verl). 

### Installation
Run the following command to setup the docker image `rocm6.3.4:vllm-0.8.5-numa-patch-ubuntu-22.04-cascade-fix`
```bash
docker build -f docker/dockerfile -t rocm6.3.4:vllm-0.8.5-numa-patch-ubuntu-22.04-cascade-fix .
```

### Data Preparation
Run the following commands to prepare the RL data:
```bash
cd verl
python examples/data_preprocess/big_math.py
python examples/data_preprocess/deep_math.py
python examples/data_preprocess/deepscaler.py
```

### Training
```bash
cd verl
sbatch ./scripts/run_grpo_multi_nodes.sh
```

## Evaluation
Our evaluations are based on the [DeepScaleR](https://github.com/rllm-org/rllm/tree/deepscaler) codebase. All our evaluations are done on a single node with 8 AMD Instinct™ MI300X GPUs.

Run the following commands to setup the evaluation environment:
```bash
# For AMD GPUs:
cd evals
# Start the docker container:
bash start_docker.sh
docker exec -it instella-math-eval bash
# Install dependencies
bash install_setup.sh
```
All the processed test datasets are avilable in `evals/deepscaler/data_processed`. Directly run the following command to reproduce our results:
```bash
bash run_eval_math.sh
```

## Acknowledgement
The RL training codebase is built from [VERL](https://github.com/volcengine/verl). 

The evaluation codebase is built from [DeepScaleR](https://github.com/rllm-org/rllm/tree/deepscaler).

## License

- The Instella-3B-Math models are licensed for academic and research purposes under a ResearchRAIL license. 
- Refer to the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files for more information.

## Citations
Feel free to cite our Instella-3B models and give us a star⭐ if you find our work helpful :)

```text
@misc{Instella,
    title = {Instella: Fully Open Language Models with Stellar Performance},
    url = {https://huggingface.co/amd/Instella-3B},
    author = {Jiang Liu and Jialian Wu and Xiaodong Yu and Prakamya Mishra and Sudhanshu Ranjan and Zicheng Liu and Chaitanya Manem and Yusheng Su and Pratik Prabhanjan Brahma and Gowtham Ramesh and Ximeng Sun and Ze Wang and Emad Barsoum},
    month = {March},
    year = {2025}
}
```
