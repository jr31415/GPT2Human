import sys
runtype = sys.argv[1]
arg2 = sys.argv[2] #either str (a sentence) or str (path to jacob sentences)
if len(sys.argv) == 4:
    arg3 = sys.argv[3] #str (path to gpt sentences)

from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
pipeline = pipeline(
    task="fill-mask",
    model="facebook/bart-large",
    dtype=torch.float16,
    device=0
)

device = torch.device("mps")

class decode(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )
    
    def forward(self, x):
        out = self.net(x)
        out = out.view(x.size(0), 1024)
    
        return out

class DataSet(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, itemid):
        embedding, target = self.samples[itemid]
        return torch.tensor(embedding), torch.tensor(target)

def encode(sentences):
    tokens = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.encoder_last_hidden_state, attention_mask
    
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

if runtype == "-c":
    with open(arg2) as infile:
        jacobs = infile.read().split("\n")

    with open(arg3) as infile:
        gpts = infile.read().split("\n")

    print("Computing GPTs...")
    gptmeanpool = get_model_mean_pool(gpts)

    print("Computing Jacobs...")
    jacobmeanpool = get_model_mean_pool(jacobs)

    jacobdirection = jacobmeanpool - gptmeanpool

    print(f"Directional vector = {jacobdirection}")
    torch.save(jacobdirection, 'direction.pt')
    exit()

if runtype == "-t":
    decoder = decode()
    decoder.to(device)
    with open(arg2) as infile:
        jacobs = infile.read().split("\n")
        
    def train(loader, epochs=64):
        optimizer = torch.optim.SGD(decoder.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            print(f"Training Epoch #{epoch + 1}")
            mse_loss = nn.MSELoss()
            for samples, target in loader:
                samples, target = samples.to(device), target.to(device)
                optimizer.zero_grad()
                model_output = decoder(samples)
                loss = mse_loss(model_output, target)
                print(f"Loss: {loss}")
                loss.backward()
                optimizer.step()
        
    
    jacoboutputs, jacobmasks = encode(jacobs)
    unsqueezedmasks = torch.unsqueeze(jacobmasks, dim=-1)
    targets = jacobouputs * unsqueezedmasks
    
    samples = []
    for sentence, mask in zip(torch.split(jacoboutputs, dim=0), torch.split(jacobmasks, dim=0)):
        samples.append(mean_pool(sentence, mask))
    data = zip(samples, targets)
    dataset = DataSet(data)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    train(loader, epochs=64)
    torch.save(decoder, 'decode.pt')
    
    
if runtype == "-s":
    dataset = DataSet(samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    #TODO: FINISH THIS
