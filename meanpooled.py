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
        self.ll1 = nn.Linear(1024, 512) #linear layer 1
        self.tembed = nn.Embedding(50265, 512) #token embedding layer
        self.pembed = nn.Embedding(100, 512) #positional embedding layer
        self.dl = nn.TransformerDecoderLayer(nhead=16, dim_feedforward=2048, d_model=512) #decoder layer
        self.tl = nn.TransformerDecoder(self.dl, num_layers=4)
        self.ol = nn.Linear(512, 50265) #output layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, embedding, seq, padding=None):
        seq_len = seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask==1, float("-inf")).to(seq.device)
        proj = self.ll1(embedding)
        positions = torch.arange(seq_len, device=seq.device)
        token_embeddings = self.tembed(seq)
        position_embeddings = self.pembed(positions).unsqueeze(dim=0).expand(token_embeddings.size(dim=0), -1, -1)
        decoder_input = (position_embeddings + token_embeddings).transpose(0, 1) #mark word position
        memory = proj.unsqueeze(dim=0).expand(seq_len, -1, -1)
        output = self.ol(self.dropout(self.tl(decoder_input, memory, tgt_mask=mask, tgt_key_padding_mask=padding))).transpose(0, 1)
        
        return output
        
class DataSet(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, itemid):
        embedding, target = self.samples[itemid]
        return embedding, target

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
    pooled_means = []
    
    for sentencesid, sentences in enumerate(batched_sentences):
        if print_on == True:
            print(f"Calculating batch {sentencesid+1} of {total_len}")
        outputs, attention_mask = encode(sentences)
        pooled_mean = mean_pool(outputs, attention_mask)
        pooled_means.append(pooled_mean)
    
    print(torch.cat(pooled_means, dim=0).mean(dim=0))
    pooled_means = torch.cat(pooled_means, dim=0).mean(dim=0)
    
    return pooled_means
    
def encode_all_sentences(unbatched_sentences, batch_size=64, print_on=True):
    batched_sentences = batch_inputs(unbatched_sentences, batch_size) #so that we don't kill memory and get a useful indication of progress
    batched_sentences = list(batched_sentences)
    total_len = len(batched_sentences)
    outputs, attention_masks = [],[]
    max_len = 0
    
    for sentencesid, sentences in enumerate(batched_sentences):
        if print_on == True:
            print(f"Encoding batch {sentencesid+1} of {total_len}")
        output, attention_mask = encode(sentences)
        outputs.append(output)
        attention_masks.append(attention_mask)
        output_len = output.size(dim=1)
        if max_len < output_len:
            max_len = output_len
        
    
    padded_outputs, padded_masks = [],[]
    for output, mask in zip(outputs, attention_masks):
        batch_size, seqlen, dummy = output.shape
        pad_len = max_len - seqlen
        
        if pad_len > 0:
            output_pad = torch.zeros(batch_size, pad_len, 1024)
            mask_pad = torch.zeros(batch_size,pad_len)
            
            output = torch.cat([output, output_pad], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
     
        padded_outputs.append(output)
        padded_masks.append(mask)

    return torch.cat(padded_outputs, dim=0), torch.cat(padded_masks, dim=0)

if runtype == "-c": #create directional vector
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

if runtype == "-t": #train decoder model
    decoder = decode()
    decoder.to(device)
    with open(arg2) as infile:
        jacobs = infile.read().split("\n")
        
    def train(loader, epochs=64):
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
        lossfn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        for epoch in range(epochs):
            print(f"Training Epoch #{epoch + 1}")
            for samples, target in loader:
                samples, target = samples.to(device), target.to(device)
                padding = target == tokenizer.pad_token_id
                optimizer.zero_grad()
                model_output = decoder(samples, target, padding=padding)
                loss = lossfn(model_output.reshape(-1, 50265), target.reshape(-1))
                print(f"Loss: {loss}")
                loss.backward()
                optimizer.step()
        
    jacoboutputs, jacobmasks = encode_all_sentences(jacobs[:100])
    print("this is done king :)")
    unsqueezedmasks = torch.unsqueeze(jacobmasks, dim=-1)
    targets = tokenizer(jacobs[:100], padding=True, return_tensors="pt")["input_ids"]
    samples = []
    for sentence, mask in zip(torch.split(jacoboutputs, 1, dim=0), torch.split(jacobmasks, 1, dim=0)):
        samples.append(mean_pool(sentence, mask))
    samples = torch.cat(samples, dim=0)
    data = list(zip(samples, targets))
    dataset = DataSet(data)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    train(loader, epochs=64)
    torch.save(decoder, 'decode.pt')
    
    
if runtype == "-s":
    dataset = DataSet(samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    #TODO: FINISH THIS
