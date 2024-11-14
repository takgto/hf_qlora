# hf_qlora
Python scripts for finetuning by QLoRA in huggningcace library. 
The following two jupyter notebook files (*.ipynb) are utilized for QLORA finetuneing experiment.  
**1. train_medalpaca.ipynb** Finetune LLM using QLoRA technique.The datasets are alpaca medical datasets in huggingface datasets libraries. nf4bit, nf8bit and bf16 can be tested.  
**2. similarity_comp.ipynb** Evaluate finetuned LLM loading medical test datasets. Similarity comparison between test answer and correct answer is computed and pickup agreed number which is determined by thresholding. 
