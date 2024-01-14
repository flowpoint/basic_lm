# the license for the following is at the end of the file
from datasets import load_dataset, Dataset, DatasetDict
from copy import copy

##
ds = load_dataset("OpenAssistant/oasst2")

print("if you use this without much change, consider attributing me (flowpoint)")

def build_split(ds, split):
    def filter_lang(ds, langs=['en']):
        ds = ds.filter(lambda s: s['lang'] in langs)
        return ds

    langs = ['en']
    if langs != None:
        ds = filter_lang(ds[split], langs)
    ##

    unvisited = set()
    sd = {}

    # build index from message id to the full data
    for sample in ds:
        mid = sample['message_id'] 
        sd[mid] = sample
        unvisited.add(mid)

    parents = {}
    max_tree_depth = 100
    tree_depth = 0

    # build threads, threads are identified by them all pointing to their first message

    # run repeated times through unused samples to build threads
    while len(unvisited) > 0 and tree_depth < max_tree_depth:
        for sample in list(unvisited):
            mid = sd[sample]['message_id'] 
            pid = sd[sample]['parent_id'] 

            # root of a tree
            if pid is None:
                # mark thread beginnings by setting their parent to themself
                # because we need different keys for different threads
                parents[mid] = mid
                unvisited.discard(mid)
            # append new leaf
            if pid in parents:
                # save the root transitively as the value of this child in the parents dict
                parents[mid] = parents[pid]
                unvisited.discard(mid)

        tree_depth += 1

    # now collect all messages for a thread, not just the indices
    trees = {}

    for k,v in parents.items():
        if v not in trees:
            # thread beginning
            trees[v] = [sd[v]]
        else:
            trees[v] = trees[v] + [sd[k]]


    # now collect the highest rated path in each thread
    def rank_value(sample):
        if sample['rank'] is None:
            return 1000
        else:
            return int(sample['rank'])

    def tree_to_sample(tree):
        treec = copy(tree)
        res = []

        for msg in treec:
            #print(msg['parent_id'])
            if msg['parent_id'] is None:
                res = [msg]
                treec.remove(msg)

        if res == []:
            return None

        while treec != []:
            current_leaf = list(filter(lambda x: x['parent_id'] == res[-1]['message_id'], treec))
            if current_leaf == []:
                break

            best = min(current_leaf, key=rank_value)
            res.append(best)
            for msg in current_leaf:
                treec.remove(msg)

        return res 

    processed_ds = [tree_to_sample(t) for t in trees.values()]

    # now that we have the ordered samples of each thread, concatenate/format their text
    def sample_to_text(sample):
        res = ''
        for s in sample:
            res += f"{s['role']}: {s['text']}\n"

        return res

    processed_ds_threads = [sample_to_text(s) for s in processed_ds]
    output_ds = Dataset.from_dict({'text':processed_ds_threads}, split=split)
    return output_ds

oasst2_guanaco = DatasetDict(
        {'train': build_split(ds, 'train'),
         'validation': build_split(ds, 'validation')
        })

oasst2_guanaco.save_to_disk('oasst2_guanaco')
print("if you use this without much change, consider attributing me (flowpoint)")

'''
Copyright (c) 2024 flowpoint

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
