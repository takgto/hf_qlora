# hf_qlora
Python scripts for finetuning by QLoRA in huggningcace library. 
The following two jupyter notebook files (*.ipynb) are utilized for QLORA finetuneing experiment.  
**1. train_medalpaca.ipynb** Finetune LLM using QLoRA technique.The datasets are alpaca medical datasets in huggingface datasets libraries. nf4bit, nf8bit and bf16 can be tested.  
**2. similarity_comp.ipynb** Evaluate finetuned LLM loading medical test datasets. Similarity comparison between test answer and correct answer is computed and pickup agreed number which is determined by thresholding(>0.9). 

## Enviroment variables setting
Environment variables should be set as follows.
```bash
export HF_TOKEN=<your huggingface token>
```
```bash
export OPENAI_API_KEY=<your openAI API key>
```
Need to set CUDA DEVICES to avoid memory error. 
```bash
export CUDA_VISIBLE_DEVICES=0 (or single digit)
```

## Dependencies  
- torch: tested on v2.3.0+cu11.8 (python3.10.12)  
- transformers: test on 4.41.2
- datasets: tested on 2.19.1
- huggingface_hub: tested on 0.23.2
- langchain: tested on 0.3.7
- Need to install JetMoE repo (https://github.com/myshell-ai/JetMoE)  
  
