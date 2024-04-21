from pathlib import Path
import shutil
import gc
import json

from tqdm import tqdm
# its apparently the 20b tokenizer:
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_model, load_model
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
import argparse
from transformers import MambaForCausalLM

from utils import Checkpoint, save_checkpoint, load_checkpoint

from pdb import set_trace as bp

#device = 'cpu'
device = 'cuda'
#model_type = 'hf_130m'
model_type = 'mybyte'

def infer_loop(modelname):
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #model = load_safe_mamba(Path(modelname))
    if model_type == 'mybyte':
        model_config = MambaConfig(
                d_model=768,
                n_layer=24,
                vocab_size=255,
                )
        model = MambaLMHeadModel(model_config)

        ckpt = load_checkpoint(args.modelname)
        model.load_state_dict(ckpt.data['model'])
    elif model_type == 'hf_130m':
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')


    '''
    prec = 'half'
    if prec == 'half':
        model = model.half()
    '''

    model = model.to(device)
    def forw(tok):
        out = model(tok)
        out
    def infer_sample(txt):
        if model_type == 'hf_130m':
            input_ids = tokenizer(txt, return_tensors='pt', truncation=False, padding=True).to(device)
        else:
            input_ids = {'input_ids': torch.tensor([list(txt.encode('utf-8'))], dtype=torch.int64, device=device)}

        outputs = model.generate(input_ids=input_ids['input_ids'],
                                 max_length=200,
                                 #do_sample=False,
                                 #repetition_penalty=1.,
                                 #no_repeat_ngram_size=3,
                                 #cg=True,
                                 #return_dict_in_geenerate=False,
                                 #temperature=0.2
                                 )
        # 0x7b = 127  is the maximum utf-8 ascii
        if model_type == 'hf_130m':
            txt_out = tokenizer.decode(outputs[0])
        else:
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




import torch

import transformers
from transformers import AutoTokenizer

import lm_eval

# bad initial model
mpath = 'a61eb315-8933-46ab-b7f3-df7f45fb1cb5'
# trained good model
#mpath = "18f3e7c2-f4df-45fc-adc5-6b842808c21d"

class MyMamba(lm_eval.api.model.LM):
    def __init__(self):
        model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        ckpt = load_checkpoint(mpath)
        model.load_state_dict(ckpt.data['model'])
        self.model = model

    def loglikelihood(self, requests):
        pass

    def loglikelihood_rolling(self, requests):
        pass

    def generate_until(self, requests):

        '''
        ans = []
        for request in requests:
            txt, kwargs = request
            ans.append(infer_sample(txt, kwargs))
        return ans
        '''
        pass

    def infer_sample(txt, kwargs):
        #input_ids = tokenizer(txt, return_tensors='pt', truncation=False, padding=True).to(device)
        input_ids = {'input_ids': torch.tensor([list(txt.encode('utf-8'))], dtype=torch.int64, device=device)}
        outputs = self.model.generate(input_ids=input_ids['input_ids'],
                                 max_length=kwargs['max_gen_toks'],
                                 do_sample=False,
                                 #repetition_penalty=1.,
                                 #no_repeat_ngram_size=3,
                                 #cg=True,
                                 #return_dict_in_geenerate=False,
                                 #temperature=0.2
                                 )
        #txt_out = tokenizer.decode(outputs[0])
        # 0x7b = 127  is the maximum utf-8 ascii
        outs = outputs[0].to('cpu').numpy().clip(0,127)
        print(outs)
        txt_out = bytes(list(outs)).decode('utf-8')
        return txt_out


m = MyMamba()
m.generate_until([['hello',{'max_gen_toks':20}]])


import torch

import transformers
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

class MyTok:
    def __init__(self, *args, **kwargs):
        self.vocab_size=255
        pad_char = '\t'.encode('utf-8')
        self.pad_char = pad_char
        self.pad_token = '\t'
        self.pad_token_id = '\t'.encode('utf-8')
        self.eos_token = '\t'
        self.eos_token_id = '\t'.encode('utf-8')

    def _encode(self, txt, *args, **kwargs):
        #bp()
        #b = [txt.replace('\t', ' ')]
        b = txt
    
        byt = [x.encode('utf-8') for x in b]
        bmax = max([len(x) for x in byt])
        bmax = min(max([len(x) for x in byt]), 768)
        byt = [list(x.ljust(bmax, self.pad_char)) for x in byt]
        #b = {'input_ids': torch.tensor(byt, dtype=torch.int64)}
        return byt#b['input_ids']

    def encode(self, text, text_pair=None, *args, **kwargs):
        assert isinstance(text, str)
        ret = []
        if isinstance(text, str):
            ret.append( self._encode([text], *args, **kwargs) )
            return ret[0][0]
        elif isinstance(text, list):
            ret.append( [self._encode(x, *args, **kwargs) for x in text] )
            return ret
        
        '''
        if text_pair is not None:
            if isinstance(text_pair, str):
                ret.append( self._encode(text_pair, *args, **kwargs) )
            elif isinstance(text_pair, list):
                ret.append( [self._encode(x, *args, **kwargs) for x in text_pair] )

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
        '''

    def decode(self, output_ids, *args, **kwargs):
        outs = output_ids[0].to('cpu').numpy().clip(0,127)
        txt_out = bytes(list(outs)).decode('utf-8')#.replace('\t', '')
        return txt_out

#python -m pdb scripts/infer.py --model mamba --model_args pretrained=state-spaces/mamba-130m  --tasks winogrande  --device cpu --batch_size 32

from pdb import set_trace as bp

@register_model("mamba")
class MyMambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=None, device=device,
                 dtype=torch.float16):
        LM.__init__(self)
        #self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)

        model = MambaForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
        ckpt = load_checkpoint(mpath)
        model.load_state_dict(ckpt.data['model'])
        self._model = model

        #bp()
        self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        mtok = MyTok()

        def la(*args, **kwargs):
            #bp()
            #return self.tokenizer.encode2(*args, **kwargs)
            return mtok.encode(*args, **kwargs)
        def lb(*args, **kwargs):
            #bp()
            #return self.tokenizer.encode2(*args, **kwargs)
            return mtok.decode(*args, **kwargs)
        #self.tokenizer.encode2 = self.tokenizer.encode
        #self.tokenizer.decode2 = self.tokenizer.decode
        #self.tokenizer.encode = la
        #self.tokenizer.decode = lb
        self.tokenizer = MyTok()
        #self.tokenizer = Ntok

        pad_char = ' '.encode('utf-8')
        self.tokenizer.pad_token_id = pad_char
        #self.tokenizer.eos_token_id = 

        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        #self._max_length = max_length
        self._max_length = 768
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()

'''
if __name__ == '__main__':
    cli_evaluate()
'''
