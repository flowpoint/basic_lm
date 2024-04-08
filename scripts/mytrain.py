import random
import logging
import sys
from datetime import datetime
from time import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, Any
from uuid import uuid4
import json

import lmdb
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import datasets
import transformers
from transformers import MambaForCausalLM
from aim import Run


import optuna


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

run = None#Run()

stub = True

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

model = None

def get_model():
    #model = MambaLMHeadModel.from_pretrained(basemodel)
    global model
    del model
    gc.collect()
    model = MambaForCausalLM.from_pretrained(basemodel)
    if precision == 'fp16':
        model = model.half()

    model = model.to(device)
    return model

get_model()

#optim = transformers.Adafactor(model.parameters(), lr=lr, relative_step=False)
optim = torch.optim.AdamW(model.parameters(), lr=lr)#, relative_step=False)


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
def stepfn(sample_i, batch, maxl, run=None):
    b = batch['text']
    #b = ['asdfjka'*10000]*bs
    #b = tokenizer(b, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
    b = tokenizer(b, return_tensors='pt', padding='longest', truncation='do_not_truncate', max_length=maxl)
    # truncate here instead
    input_ids = b['input_ids'][..., :maxl].to(device)
    outputs = model(input_ids)
    logits =  outputs.logits
    loss = loss_fn(input_ids, logits)

    loss.backward()

    if sample_i % grad_accum == 0:
        optim.step()
        optim.zero_grad()
        #print(print_text_example(batch['text'][0]))

    iloss = loss.detach().to('cpu').numpy().mean()
    if run is not None:
        run.track(iloss, name='train_loss')
    return iloss

@torch.autocast(device_type='cuda', dtype=torch.float16)#, enabled=prec == 'amp')
def valid_step(sample_i, batch, maxl, run=None):
    with torch.no_grad():
        b = batch['text']
        b = tokenizer(b, return_tensors='pt', padding='longest', truncation='do_not_truncate', max_length=maxl)
        # truncate here instead
        input_ids = b['input_ids'][..., :maxl].to(device)
        outputs = model(input_ids)
        logits =  outputs.logits
        loss = loss_fn(input_ids, logits)
        iloss = loss.detach().to('cpu').numpy().mean()
        if run is not None:
            run.track(iloss, name='eval_loss')
        return iloss


@disk_cached
def sniff_batchsize(maxl):
    print('autotuning batchsize')
    exp_fac = 4
    working_bs = 1
    while 1:
        test_bs = working_bs * exp_fac
        fake_batch = {'text': ['sniff_text;."]h$'*32*maxl] * test_bs }
        try:
            # 2 grad/ accum batches for safety
            for i in tqdm(range(grad_accum+1)):
                stepfn(i, fake_batch, maxl)
            working_bs = test_bs
        except torch.cuda.OutOfMemoryError:
            print(f'error with test_bs: {test_bs} stopping batch-expansion')
            exp_fac = exp_fac // 2

            if exp_fac == 1:
                print(f'working bs size is {working_bs}')
                break

    return working_bs



def print_text_example(t):
    width = 70
    height = 4
    lines = [x[:width] for x in t.split('\n')][:height]
    return '\n'.join(lines)

# a handle/handler for the whole trainsate
# possibly eventually consistent
@dataclass
class Trainstate:
    model: Any
    optimizer: Any

def state_to_checkpoint(trainstate, header={}):
    # create a synchronized trainstate, i.e. checkpoint
    # stub for now, bc not distributed
    cpid = str(uuid4())
    metadata = {
            "timestamp": (datetime.now().isoformat()),
            'id': cpid,
            'model_config': ''}
    for k,v in header.items():
        metadata[k] = v
    print(f'saving checkpoint {cpid}')
    data = {"model": trainstate.model, 
            'optimizer': trainstate.optimizer}
    # checkpoint : tuple(metadata, data)
    return Checkpoint(metadata, data)

def trainloaders(ds, count, batch_size):
    for i in range(count):
        yield i, torch.utils.data.DataLoader(
                ds, 
                batch_size=batch_size,
                shuffle=True, 
                pin_memory=True,
                drop_last=True
                )

def validloaders(ds, count, batch_size):
    for i in range(count):
        yield i, torch.utils.data.DataLoader(
                ds, 
                batch_size=batch_size,
                shuffle=True, 
                pin_memory=True,
                drop_last=True
                )


def train(trial, hparams):
    print(f'starting training run, with hparams: {hparams}')
    batch_size = hparams['batch_size']
    gc.collect()
    ds = datasets.load_from_disk('data/oasst2_top1_en')
    trainset = ds['train']
    validset = ds['validation']
    if stub:
        # run for 4 steps
        trainset = trainset.select(range(grad_accum*4))

    trainstate = Trainstate(model, optim)
    trainplan = trainloaders(trainset, num_epochs, batch_size)
    valids_per_epoch = 2
    validplan = iter(validloaders(validset, num_epochs * valids_per_epoch, batch_size))
    epoch_size = len(trainset)
    eval_period = len(trainset) // valids_per_epoch

    eval_losses = []
    for epoch, dl in trainplan:
        stt = time()

        # note, progress in batches, not steps
        for t_batch_i, t_batch in enumerate(tqdm(dl)):
            i_sample = epoch * epoch_size * grad_accum + t_batch_i*batch_size
            l = stepfn(i_sample, t_batch, context_size, run=run)
            if stt > time() + 20 * 60:
                save_checkpoint(state_to_checkpoint(trainstate, {"hparams":hparams}))

            if t_batch_i % eval_period == 0:
                valid_count, valid_loader = next(validplan)
                eval_losses = []
                for v_batch_i, v_batch in enumerate(tqdm(valid_loader)):
                    i_sample = v_batch_i
                    l = valid_step(i_sample, t_batch, context_size, run=run)
                    eval_losses.append(l)

                trial.report(np.mean(eval_losses), valid_count)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    save_checkpoint(state_to_checkpoint(trainstate, {"hparams":hparams}))
    return np.mean(eval_losses)

def run_trial(trial):
    seed=42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # also maybe: env var: CUBLAS_WORKSPACE_CONFIG=:16:8
    # torch.utils.deterministic.fill_uninitialized_memory = True

    global grad_accum
    global lr
    global num_epochs
    grad_accum = 2 ** trial.suggest_int('grad_accum', 4, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 1, 5)

    global run
    run = Run()

    batch_size = min(sniff_batchsize(context_size), grad_accum)

    hparams = {'batch_size':batch_size, 'lr':lr, 'num_epochs':num_epochs, 'grad_accum':grad_accum}
    assert grad_accum % batch_size == 0, f'bs has to divide grad_accum but is {batch_size}, {grad_accum}'
    get_model()

    return train(trial, hparams)

def htune():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'study_1'
    storage_name = "sqlite:///data/studies/{}.db".format(study_name)
    #sampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.TPESampler(seed=42)

    # max number of total training steps, estimated
    # by dataset having 5000 samples, 3 top models, 5 epochs
    max_resource = 3*5 * 2000 
    pruner = optuna.pruners.HyperbandPruner(max_resource=max_resource)
    study = optuna.create_study(study_name=study_name, 
                                sampler=sampler,
                                direction=optuna.study.StudyDirection.MINIMIZE,
                                pruner=pruner,
                                storage=storage_name)
    study.optimize(run_trial, n_trials=20)

if __name__ == '__main__':
    htune()
