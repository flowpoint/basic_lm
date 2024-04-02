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

def infer_loop(modelname)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = load_safe_mamba(str(Path(modelname) / "model.safetensors"))

    #device = 'cpu'
    device = 'cuda'
    prec = 'half'
    if prec == 'half':
        model = model.half()

    model = model.to(device)
    def infer_sample(txt):
        input_ids = tokenizer(txt, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model.generate(input_ids=input_ids['input_ids'],
                                 max_length=200,
                                 cg=True,
                                 #return_dict_in_geenerate=False,
                                 temperature=0.
                                 )
        return tokenizer.decode(outputs[0])

    while 1:
        try:
            inp = input("prompt: ")
            fmt = f"prompter: {inp}\nassistant: "
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
    infer_loop(modelname)


def convert(checkpoint, output_dir):
    ''' convert pytorch checkpoint to safetensor 
    takes checkpoint dir
    then saves a safetensor checkpoint to output_dir
    '''
    output_dir.mkdir()
    config_file = Path(checkpoint) / 'config.json'

    with config_file.open('r') as f:
        config_data = json.loads(f.read())

    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config)
    ckpt = torch.load(checkpoint)
    model_d = ckpt['model']
    shutil.copy(str(config_file), str(output_dir/'config.json'))
    save_model(model, str(Path(output_dir/'model.safetensors')))

def load_safe_mamba(path, device=None, dtype=None, **kwargs):
    templ = Path('model_templ/config.json')
    with templ.open('r') as f:
        config_data = json.loads(f.read())
    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config, device=device, dtype=dtype, **kwargs)
    load_model(model, path)
    return model
