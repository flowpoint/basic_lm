from datetime import datetime
from time import time
import gc
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
import datasets
import transformers
from transformers import MambaForCausalLM
from aim import Run

import lmdb
import json

from safetensors.torch import save_model, load_model
from dataclasses import dataclass
from typing import Optional, Literal, Any
from uuid import uuid4

class DiskCache:
    def __init__(self):
        # 100MB max size
        map_size = 100 * 1000*2
        path = '/tmp/basic_lm_cache'
        self.env = lmdb.open(path, max_dbs=10, map_size=map_size)

    def put(self, k, v):
        with self.env.begin(write=True) as txn:
            txn.put(k.encode('utf-8'), v.encode('utf-8'))

    def __contains__(self, k):
        with self.env.begin(write=False) as txn:
            return txn.get(k.encode('utf-8')) is not None

    def get(self, k):
        with self.env.begin(write=False) as txn:
            return txn.get(k.encode('utf-8')).decode('utf-8')

cache = DiskCache()

def disk_cached(func):
    def wrapper(*args, **kwargs):
        k = json.dumps({'fname': func.__name__, "args":args, "kwargs":kwargs}) 
        if k in cache:
            print('cache hit')
            cr = json.loads(cache.get(k))['val']
            return cr
        else:
            print('cache fail')
            res = func(*args, **kwargs)
            val = json.dumps({"val": res})
            cache.put(k, val)
            return res
    return wrapper


from mamba_ssm import Mamba
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from utils import Checkpoint, save_checkpoint

run = Run()

stub = False

# hparams
lr = 1.5e-3
grad_accum = 256
context_size = 768
#batch_size = 2
#basemodel = 'state-spaces/mamba-2.8b-slimpj'
basemodel = 'state-spaces/mamba-130m-hf'
#precision = 'half'
precision = ['fp16', 'amp'][0]
num_epochs = 1

# run settings
device = 'cuda'


# its apparently the 20b tokenizer:
#tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = transformers.AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#model = MambaLMHeadModel.from_pretrained(basemodel)
model = MambaForCausalLM.from_pretrained(basemodel)
if precision == 'fp16':
    model = model.half()

model = model.to(device)
gc.collect()

optim = transformers.Adafactor(model.parameters(), lr=lr, relative_step=False)


def loss_fn(labels, logits):
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    #loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    return loss


#@torch.compile
@torch.autocast(device_type='cuda', dtype=torch.float16)#, enabled=prec == 'amp')
def stepfn(sample_i, batch, maxl):
    b = batch['text']
    #b = ['asdfjka'*10000]*bs
    #b = tokenizer(b, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
    b = tokenizer(b, return_tensors='pt', padding=True, truncation=True, max_length=maxl)
    input_ids = b['input_ids'][..., :maxl].to(device)
    outputs = model(input_ids)
    logits =  outputs.logits
    loss = loss_fn(input_ids, logits)
    loss.backward()

    if sample_i % grad_accum == 0:
        optim.step()
        optim.zero_grad()
        print(batch['text'][0])

    run.track(loss.detach().to('cpu').numpy().mean(), name='loss')
    return loss


@disk_cached
def sniff_batchsize(maxl):
    exp_fac = 4
    working_bs = 1
    while 1:
        test_bs = working_bs * exp_fac
        fake_batch = {'text': ['sniff_text;."]h$'*4*1024] * test_bs }
        try:
            # 2 grad/ accum batches for safety
            for i in tqdm(range(grad_accum*2+1)):
                stepfn(i, fake_batch, maxl)
            working_bs = test_bs
        except Exception as e:
            print(e)
            print(f'error with test_bs: {test_bs} stopping batch-expansion')
            exp_fac = exp_fac // 2

            if exp_fac == 1:
                print(f'working bs size is {working_bs}')
                break

    return working_bs



def print_text_example(t):
    width = 70
    height = 4
    lines = [x[:width] for x in mlt.split('\n')][height]
    return '\n'.join(lines)


    print(


# a handle/handler for the whole trainsate
# possibly eventually consistent
@dataclass
class Trainstate:
    model: Any
    optimizer: Any

def state_to_checkpoint(trainstate):
    # create a synchronized trainstate, i.e. checkpoint
    # stub for now, bc not distributed
    cpid = str(uuid4())
    metadata = {
            "timestamp": (datetime.now().isoformat()),
            'id': cpid,
            'model_config': ''}
    print(f'saving checkpoint {cpid}')
    data = {"model": trainstate.model, 
            'optimizer': trainstate.optimizer}
    # checkpoint : tuple(metadata, data)
    return Checkpoint(metadata, data)


def train(batch_size):
    gc.collect()
    ds = datasets.load_from_disk('data/oasst2_top1_en')
    trainset = ds['train']
    if stub:
        # run for 4 steps
        trainset = trainset.select(range(grad_accum*4))

    trainstate = Trainstate(model, optim)
    save_checkpoint(state_to_checkpoint(trainstate))

    for epoch in range(num_epochs):
        dl = torch.utils.data.DataLoader(
                trainset, 
                batch_size=batch_size,
                shuffle=True, 
                pin_memory=True
                )
        stt = time()

        # note, progress in batches, not steps
        for batch_i, batch in enumerate(tqdm(dl)):
            assert grad_accum % bs == 0, f'bs has to divide grad_accum but is {bs}, {grad_accum}'
            stepfn(batch_i*bs, batch, context_size)
            if stt > time() + 20 * 60:
                save_checkpoint(state_to_checkpoint(trainstate))

    save_checkpoint(state_to_checkpoint(trainstate))


if __name__ == '__main__':
    bs = sniff_batchsize(context_size)
    train(bs)
