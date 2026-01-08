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
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state, attention_mask
    
def mean_pool(outputs, attention_mask):
    extended_mask = torch.unsqueeze(attention_mask, dim=-1) #make the mask the same dimensions as the outputs
    padded_outputs = outputs * extended_mask #remove any masked elements
    real_tokens = torch.sum(extended_mask, dim=1)
    summed_tokens = torch.sum(padded_outputs, dim=1)
    averages = summed_tokens / real_tokens
    return averages
    
def batch_inputs(sentences, batch_size=32):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i: i+batch_size]

def get_model_encodings(unbatched_sentences, batch_size=32, print_on=True):
    batched_sentences = batch_inputs(unbatched_sentences, batch_size)
    batched_sentences = list(batched_sentences)
    total_len = len(batched_sentences)
    outs, attns = [],[]
    
    for sentencesid, sentences in enumerate(batched_sentences):
        if print_on == True:
            print(f"Calculating batch {sentencesid} of {total_len}")
        outputs, attention_mask = encode(sentences)
        outs.append(outputs.cpu())
        attns.append(attention_mask.cpu())
        
    concatenated_outs = torch.cat(outs, dim=0)
    concatenated_attns = torch.cat(attns, dim=0)
        
    return concatenated_outs, concatenated_attns

with open(gptfile) as infile:
    gpts = infile.read().split("\n")

with open(jacobfile) as infile:
    jacobs = infile.read().split("\n")

print("Computing GPTs...")
outputs, attention_mask = get_model_encodings(gpts)
gptmeanpool = mean_pool(outputs, attention_mask)

print("Computing Jacobs...")
outputs, attention_mask = get_model_encodings(jacobs)
jacobmeanpool = mean_pool(outputs, attention_mask)

    

jacobdirection = (jacobmeanpool - gptmeanpool).mean(dim=0)

print(jacobdirection)

