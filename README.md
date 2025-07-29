<div align="center">
  <br>
  <br>
  <h1>Instella-Math✨: Fully Open Language Model with Reasoning Capability</h1>
<a href='https://huggingface.co/AIG-GenAI/Instella-3B-Math'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>



[^1]: Here even for instruct models, we compared against pre-training tokens as 1) exact open weigth instruct model training token numbers are unknown, and 2) adding instruct model training tokens (in billions) leads to marginally insignificant shift in trends.
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

### Installation

### Data Preparation
Run the following commands to prepare the SFT data:
```bash
bash scripts/prepare_sft_data.sh
```
### Training 
Launch the SFT job with the [SFT config file](./configs/instella-3b-sft.yaml):

```
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-sft.yaml
```

Note: please make sure to update `load_path` to your final pretrain checkpoint.

## Reinforcement Learning (GRPO)
We conduct GRPO after SFT using [VERL](https://github.com/volcengine/verl). 

### Installation
Please download the training docker from .

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


 
## Acknowledgement
This codebase is built from [VERL](https://github.com/volcengine/verl). 

## License

- The Instella-3B-MATH models are licensed for academic and research purposes under a ResearchRAIL license. 
- Refer to the [LICENSE](./LICENSE) and [NOTICES](./NOTICES) files for more information.

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
