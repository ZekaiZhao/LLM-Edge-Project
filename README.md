# LLM-Edge-Project

Hello Everyone

Here, we have files for doing pruning (ranking) on Llama 3.2 1b instruct model on the GMS8K dataset and file name is called pruning_GSM8K.py. 
Modeling_fllama_soft.py is a modification of modeling_fllama.py where we can not only keep all attention edges but also do attention edges amplification. 

Doing inference on the Llama 3.2 1b instruct unpruned model, we can use baseline_model_inference.py. 
Doing inference and training, we need to load functions from fllama_boolean_expressions_fs.py, I0_fllama.py, and modeling_fllama_soft.py

Amplify_edges.py can help you directly rank the attention edges.

****
