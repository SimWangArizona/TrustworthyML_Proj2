# Trustworthy ML (ECE 696B) Project 2 Benchmark Contamination in LLMs

This project includes evaluation of [Deepseek-R1-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [LLaMA-2-7B](https://huggingface.co/meta-llama) models on [MMLU](https://huggingface.co/datasets/cais/mmlu) and [AIME-2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) datasets. This project also includes contamination detection for MMLU and LLaMA-2-7B.

---
# Requirements
- Torch
- cuda=12.1
- **Transformers >= 4.46.0**
# Clone and install the dependencies
```
git clone https://github.com/SimWangArizona/TrustworthyML_Proj2.git
cd TrustworthyML_Proj2-main
pip install -r requirements.txt
```
# How to use
## 1. Downloading [Deepseek-R1-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [LLaMA-2-7B](https://huggingface.co/meta-llama) models and set up the folder path. Downloading [MMLU](https://huggingface.co/datasets/cais/mmlu) and [AIME-2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) datasets and set up the folder path.


## 2. AIME evaluation
```
cd llm_evaluation_aime/
python evaluate.py -d <your_aime_path> -s <your_output_path> -m <your_model_path>
```

## 3. MMLU evaluation
```
cd llm_evaluation_4_mmlu/
python evaluate_hf.py -m <your_model_path> -d  <your_mmlu_path> -s <your_output_path>
```


## 4. MMLU contamination detection
```
cd llm-decontaminator/
Please refer to README.md in this directory
```

## 5. LLaMA-2-7B contamination detection
```
cd ConStat/
Please refer to README.md in this directory
```

## Acknowledgement

This code reuses components from several libraries including [llm_evaluation_mmlu](https://github.com/percent4/llm_evaluation_4_mmlu), [ConStat](https://github.com/eth-sri/ConStat) as well as [llm-decontaminator](https://github.com/lm-sys/llm-decontaminator).