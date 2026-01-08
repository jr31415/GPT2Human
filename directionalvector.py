import sys
jacobfile = sys.argv[1]
gptfile = sys.argv[2]

from transformers import pipeline, BartTokenizer, BartModel
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartModel.from_pretrained("facebook/bart-large")
pipeline = pipeline(
    task="fill-mask",
    model="facebook/bart-large",
    dtype=torch.float16,
    device=0
)

def encode(sentences):
    tokens = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state, attention_mask
    
def mean_pool(outputs, attention_mask):
    extended_mask = torch.unsqueeze(attention_mask, dim=-1) #make the mask the same dimensions as the outputs
    padded_outputs = outputs * extended_mask #remove any masked elements
    real_tokens = torch.sum(extended_mask, dim=1)
    summed_tokens = torch.sum(padded_outputs, dim=1)
    averages = summed_tokens / real_tokens
    return averages
    
def batch_inputs(sentences, batch_size=64):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i: i+batch_size]

def get_model_mean_pool(unbatched_sentences, batch_size=64, print_on=True):
    batched_sentences = batch_inputs(unbatched_sentences, batch_size) #so that we don't kill memory and get a useful indication of progress
    batched_sentences = list(batched_sentences)
    total_len = len(batched_sentences)
    outs, attns, pooled_means = [],[],[]
    
    for sentencesid, sentences in enumerate(batched_sentences):
        if print_on == True:
            print(f"Calculating batch {sentencesid+1} of {total_len}")
        outputs, attention_mask = encode(sentences)
        pooled_mean = mean_pool(outputs, attention_mask)
        pooled_means.append(pooled_mean)
    
    print(torch.cat(pooled_means, dim=0).mean(dim=0))
    pooled_means = torch.cat(pooled_means, dim=0).mean(dim=0)
    
    return pooled_means

with open(gptfile) as infile:
    gpts = infile.read().split("\n")[:128]

with open(jacobfile) as infile:
    jacobs = infile.read().split("\n")[:128]

print("Computing GPTs...")
gptmeanpool = get_model_mean_pool(gpts)

print("Computing Jacobs...")
jacobmeanpool = get_model_mean_pool(jacobs)

    

jacobdirection = (jacobmeanpool - gptmeanpool) / 2

print(jacobdirection)

