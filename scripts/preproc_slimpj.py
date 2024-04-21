import datasets
import random
from pathlib import Path

#path = '/mnt/datastore/bigdata/hf_home/lfs/SlimPajama-627B'
path = '/root/lfs/SlimPajama-627B'
#ds = datasets.load_from_disk(path)
#ds = datasets.load_dataset(path, data_files={'train':)
#ds = datasets.load_dataset('cerebras/SlimPajama-627B', 
datasets.config.HF_DATASETS_OFFLINE = True
##
train_files = list((Path(path)/'train').iterdir())
train_chunk = []

samplesize = 10

dtree = {}
for chunk in train_files:
    chunk_files = list(chunk.iterdir())
    chunk_sample = random.choices(chunk_files)
    for train_file in chunk_sample:
        train_chunk.append(str(train_file))

    dtree[str(chunk.name)] = train_chunk

#exit(1)
#with ('/root/data/subset.txt')


from tqdm import tqdm
import torch

import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel




bs=32
#bs=512
#bs=768
#bs=1024
#bs=1024
#bs=1280

# size of the text embeddings
emb_size = 384

device='cuda'

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2',
                                  load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.float16
                                  #bnb_4bit_compute_dtype=torch.float8_e5m2
                                  )#, device_map='auto')

model = model.eval()

prec = torch.float8_e4m3fn

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    ats = attention_mask.sum(dim=1)[..., None]
    lhs = last_hidden.sum(dim=1)
    '''
    if ats < 1e-10:
        return torch.zeros_like(lhs)
    else:
    '''
    return lhs / ats

#model = torch.compile(torch..autocast(model.to(device)))
#@torch.compile
#@torch.autocast('cuda', prec)
def modelfn(**batch_dict):
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings

def modelfn_dyn(**batch_dict):
    #split_ii = batch_dict['input_ids'].tensor_split(context_size, dim=-1)
    split_ii = batch_dict['input_ids'].reshape([bs,-1, context_size])
    split_am = batch_dict['attention_mask'].reshape([bs,-1, context_size])
    # context_multiple
    cmultiple = split_ii.shape[1]

    #vecs = torch.zeros([bs, emb_size], device=device)
    '''
    vecs = []
    for cl in range(cmultiple):
        outputs = model(input_ids=split_ii[:,cl], attention_mask=split_am[:,cl])
        vecs.append(outputs.last_hidden_state)
        
    outputs2 = torch.stack(vecs, dim=1).mean(dim=1).squeeze(1)
    embeddings = average_pool(outputs2, batch_dict['attention_mask'])
    '''

    vecs = []
    for cl in range(cmultiple):
        atsm = split_am[:,cl]
        ats = atsm.sum(dim=1)

        # mask out files with only padding
        empty_mask = torch.all(split_ii[:,cl]==0,dim=1)

        outputs = model(input_ids=split_ii[:,cl], attention_mask=split_am[:,cl])
        embeddings = average_pool(outputs.last_hidden_state, split_am[:,cl])
        embmasked = embeddings.masked_fill(empty_mask[...,None], 0)
        vecs.append(embmasked)
    return torch.stack(vecs, dim=1).mean(dim=1).squeeze(dim=1)
        



context_size = 512

def embed_chunk(dl):
    for batch in tqdm(dl):
        input_texts = batch['text']
        batch_dict = tokenizer(input_texts, max_length=context_size, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = modelfn(**batch_dict)


def embed(batch):
    input_texts = batch['text']
    batch_dict = tokenizer(input_texts, max_length=context_size, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = modelfn(**batch_dict)
    batch['embedings'] = outputs
    return batch

def embed_dyn(batch):
    input_texts = batch['text']
    batch_dict = tokenizer(input_texts, max_length=context_size, padding='longest', pad_to_multiple_of=context_size, truncation=False, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = modelfn_dyn(**batch_dict)
    batch['embedings'] = outputs
    return batch


'''
collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding='longest',
        pad_to_multiple_of=512
        )
'''
##
savepath = '/root/data/data/slimpajama/train'

for chunk, chunk_files in dtree.items():
    ds = datasets.load_dataset(path,
                               data_files={
                                   #"train":'train/chunk1/example_train_1??.jsonl.zst',
                                   "train":chunk_files,
                                   #"valid":'validation/chunk1/example_holdout_0.jsonl.zst'
                                },
                               )
    example = ds['train'][0:bs]
    dse = ds['train'].map(embed_dyn, batched=True, batch_size=bs, drop_last_batch=True)
    dse.save_to_disk(str(Path('/root/data/data/slimpajama/train/') / chunk))
    break
    #dl = DataLoader(ds['train'], batch_size=bs)
    #embed_chunk(chunk, dl)
