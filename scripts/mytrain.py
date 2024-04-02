from datetime import datetime
from time import time
import gc

from tqdm import tqdm
import torch
import torch.nn.functional as F
import transformers
import datasets

from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# hparams
lr = 1.5e-3
grad_accum = 64
context_size = 1024
batch_size = 2
basemodel = 'state-spaces/mamba-2.8b-slimpj'
precision = 'half'
num_epochs = 1

# run settings
device = 'cuda'


# its apparently the 20b tokenizer:
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = MambaLMHeadModel.from_pretrained(basemodel)
if precision == 'half':
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
@torch.autocast(device_type='cuda')#, enabled=prec == 'amp')
def stepfn(i, batch, maxl):
    b = batch['text']
    #b = ['asdfjka'*10000]*bs
    b = tokenizer(b, return_tensors='pt', padding=True, truncation=True)
    input_ids = b['input_ids'][..., :maxl].to(device)
    outputs = model(input_ids)
    logits =  outputs.logits
    loss = loss_fn(input_ids, logits)
    loss.backward()

    if i % grad_accum == 0:
        optim.step()
        optim.zero_grad()
    return loss


def checkpoint(data):
    ckpt_dir = Path('data/ckpts') / (datetime.now().isoformat()).mkdir()
    ckpt_path = ckpt_dir / 'ckpt.pt'
    torch.save(data, str(ckpt_path))


def train():
    gc.collect()
    ds = datasets.load_from_disk('data/oasst2_top1_en')
    trainset = ds['train']

    for epoch in range(num_epochs):
        dl = torch.utils.data.DataLoader(
                trainset, 
                batch_size=batch_size,
                shuffle=True, 
                pin_memory=True
                )
        stt = time()

        for i, batch in enumerate(tqdm(dl)):
            stepfn(i, batch, context_size)
            if stt > time() + 20 * 60:
                data = {'model':model.state_dict(), 
                        'optimizer': optim.state_dict()}
                checkpoint(data)

        data = {'model':model.state_dict(), 
                'optimizer': optim.state_dict()}
        checkpoint(data)


if __name__ == '__main__':
    train()
