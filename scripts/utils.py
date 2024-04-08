from safetensors.torch import save_model, load_model
from dataclasses import dataclass
from typing import Optional, Literal, Any

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

from mamba_ssm import Mamba
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import json

def load_safe_mamba(path, device=None, dtype=None, **kwargs):
    #templ = Path('model_templ/config.json')
    with (path / 'config.json').open('r') as f:
        config_data = json.loads(f.read())
    config = MambaConfig(**config_data)
    #model = MambaLMHeadModel(config, device=device, dtype=dtype, **kwargs)
    model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')

    ckpt = torch.load(Path(path)/'ckpt.pt')
    model.load_state_dict(ckpt['model'])
    #load_model(model, path / 'model.safetensors')

    return model

def convert(checkpoint, output_dir):
    ''' convert pytorch checkpoint to safetensor 
    takes checkpoint dir
    then saves a safetensor checkpoint to output_dir
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = Path(checkpoint) / 'config.json'

    with config_file.open('r') as f:
        config_data = json.loads(f.read())

    #config_data.pop('architectures')
    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config)
    ckpt = torch.load(Path(checkpoint)/'ckpt.pt')
    model_d = ckpt['model']
    shutil.copy(str(config_file), str(output_dir/'config.json'))
    save_model(model, str(Path(output_dir/'model.safetensors')))


def checkpoint(data):
    ckpt_dir = (Path('data/ckpts') / (datetime.now().isoformat()))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'ckpt.pt'
    torch.save(data, str(ckpt_path))
    #save_model(data['model'], str(ckpt_path))




@dataclass
class Checkpoint:
    metadata: dict
    data: dict


def save_checkpoint(checkpoint):
    ''' checkpoint is a dict of the data to save
    this function does the formatting and saving to disk
    ''' 
    ckpt_dir = Path('data/ckpts') / checkpoint.metadata['id']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (ckpt_dir / 'ckpt.json').open('w') as f:
        f.write(json.dumps(checkpoint.metadata))
        
    for k, v in checkpoint.data.items():
        ckpt_path = ckpt_dir / (k+'.pt')
        torch.save(v.state_dict(), str(ckpt_path))

    #save_model(data['model'], str(ckpt_path))


def load_checkpoint(ckpt_id):
    ckpt_dir = Path('data/ckpts') / ckpt_id
    with (ckpt_dir / 'ckpt.json').open('r') as f:
        txt = f.read()
        ckpt_metadata = json.loads(txt)

    format_ = ["model", "optimizer"]
    ckpt_data = {}
    for k in format_:
        ckpt_path = ckpt_dir / (k+'.pt')
        d = torch.load(str(ckpt_path))
        ckpt_data[k] = d

    return Checkpoint(ckpt_metadata, ckpt_data)

