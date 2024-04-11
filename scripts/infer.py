from pathlib import Path
import shutil
import gc
import json

from tqdm import tqdm
# its apparently the 20b tokenizer:
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_model, load_model
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
import argparse
from transformers import MambaForCausalLM

from utils import Checkpoint, save_checkpoint, load_checkpoint

from pdb import set_trace as bp

def infer_loop(modelname):
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model = load_safe_mamba(Path(modelname))
    model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    ckpt = load_checkpoint(args.modelname)
    model.load_state_dict(ckpt.data['model'])


    device = 'cpu'
    #device = 'cuda'
    '''
    prec = 'half'
    if prec == 'half':
        model = model.half()
    '''

    model = model.to(device)
    def infer_sample(txt):
        #input_ids = tokenizer(txt, return_tensors='pt', truncation=False, padding=True).to(device)
        input_ids = {'input_ids': torch.tensor([list(txt.encode('utf-8'))], dtype=torch.int64, device=device)}
        outputs = model.generate(input_ids=input_ids['input_ids'],
                                 max_length=200,
                                 #cg=True,
                                 #return_dict_in_geenerate=False,
                                 #temperature=0.
                                 )
        #txt_out = tokenizer.decode(outputs[0])
        # 0x7b = 127  is the maximum utf-8 ascii
        outs = outputs[0].to('cpu').numpy().clip(0,127)
        print(outs)
        txt_out = bytes(list(outs)).decode('utf-8')
        return txt_out

    while 1:
        try:
            inp = input("prompt: ")
            #fmt = f"prompter: {inp}\nassistant: "
            fmt = inp
            print(infer_sample(fmt))
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    # path to the model
    parser.add_argument('modelname')
    args = parser.parse_args()
    #convert(args.modelname, 'data/ckpts/converted')
    infer_loop(args.modelname)
    #infer_loop('data/ckpts/converted')
