# LLM-Edge-Project

Hello Everyone
Here, we have files for doing pruning (ranking) on Llama 3.2 1b instruct model on the GMS8K dataset and file name is called pruning_GSM8K.py. 
modeling_fllama_soft.py is a modification of modeling_fllama.py where we can no attention edges are pruned but we can do attention edges ampliftcaion. 

Doing inference on the Llama 3.2 1b instruct unpruned model, we can use baseline_model_inference.py. 
Doing inference and training, we need to load functions from fllama_boolean_expressions_fs.py, I0_fllama.py, and modeling_fllama_soft.py

****
