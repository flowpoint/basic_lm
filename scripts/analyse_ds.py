import datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
import zlib

# pip install pycairo
# pip install pygobject
#dnf install cairo-devel python3-gobject-devel python3-cairo-devel gtk4 cairo-gobject-devel gtk4
matplotlib.use('GTK4Agg')
##

path = '/home/flowpoint/lfs/SlimPajama-627B'
datasets.config.HF_DATASETS_OFFLINE = True

ds = datasets.load_dataset(path,
                           data_files={
                               "train":'train/chunk1/example_train_1??.jsonl.zst',
                               #"valid":'validation/chunk1/example_holdout_0.jsonl.zst'
                            },
                           )
##
trainset = ds['train']
##

stats = []
space_counts = []
lens = []
cratios = []

for s in tqdm(trainset):
    txt = s['text']
    space_count = s['text'].count(' ')
    space_counts.append(space_count)
    lens.append( len(txt))
    #stats['space_count'] = space_count
    etxt = txt.encode('utf-8')
    cetxt = zlib.compress(etxt)
    cratio = len(etxt) / len(cetxt)
    cratios.append(cratio)



##
ssa = list(reversed(sorted(enumerate(space_counts), key=lambda x: x[1])))
##
ssl = list(reversed(sorted(enumerate(lens), key=lambda x: x[1])))
##
scr = list(reversed(sorted(enumerate(cratios), key=lambda x: x[1])))
##
trainset[ssl[1][0]]


##
inp = scr
counts, bins = np.histogram(inp, bins=100)
#plt.stairs(np.log2(counts), bins)
plt.stairs(counts, bins)
plt.show()
##

