import torch

# load fp 16 model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# create pipeline
gen = pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)

# run prediction
res = gen("K-ON!’s story revolves around four (and later five) Japanese high school girls who join their school’s Light Music Club.")
print(res)