# the license for the following is at the end of the file
from datasets import load_dataset, Dataset, DatasetDict
import json

##
ds = load_dataset("OpenAssistant/oasst2")

print("if you use this without much change, consider attributing me (flowpoint)")

skip_checks = False

def collect_langs(ds):
    available_langs = set()
    for message in ds:
        available_langs.add(message['lang'])
    return list(available_langs)


def filter_langs(ds, langs):
    if not skip_checks:
        available_langs = collect_langs(ds)
        print(f"langs in dataset are: {available_langs}")
        for lang in langs:
            assert lang in available_langs
    return ds.filter(lambda s: s['lang'] in langs)


def build_message_index(ds):
    # forward index: message_id -> message_body
    sd = {}
    roots = set()
    # reversed index: parent_id -> child_id's
    inv_idx = {}

    for message in ds:
        mid = message['message_id'] 
        pid = message['parent_id'] 
        sd[mid] = message

        if pid is None:
            roots.add(mid)
        else:
            if pid in inv_idx:
                inv_idx[pid] = inv_idx[pid] + [mid]
            else:
                inv_idx[pid] = [mid]

    return sd, roots, inv_idx


def get_tree(sd, current_msg_id, inv_idx):
    ''' tree of messages '''
    if current_msg_id not in inv_idx:
        replies = []
    else:
        replies = inv_idx[current_msg_id]

    tree = {'id': current_msg_id, 
              'message_data':sd[current_msg_id], 
              'replies':[get_tree(sd, reply_id, inv_idx) for reply_id in replies]
              }
    return tree



def best_reply(replies):
    def rank_value(message):
        # sometimes the ranks are 0, therefore i assign a very high rank to them
        if message['message_data']['rank'] is None:
            return 1000
        else:
            return int(message['message_data']['rank'])
    return list(sorted(replies, key=rank_value))[0]


def tree_to_thread(tree):
    ''' walk down the tree along the path with the lowest rank '''
    thread = []
    replies = tree['replies']
    while replies != []:
        thread.append(tree['message_data'])
        tree = best_reply(replies)
        #assert tree['message_data']['rank'] == 0, tree['message_data']
        replies = tree['replies']

    thread.append(tree['message_data'])
    return thread

def msg_to_text(msg):
    return f"{msg['role']}: {msg['text']}\n\n"

def thread_to_text(thread):
    return "".join(map(msg_to_text, thread))

def build_split(ds, split, langs=['en']):
    ds = ds[split]
    if langs != None:
        ds = filter_langs(ds, langs)

    sd, roots, inv_idx = build_message_index(ds)

    threads = []
    for root in roots:
        tree = get_tree(sd, root, inv_idx)
        thread = tree_to_thread(tree)
        threads.append(thread)

    samples = list(map(thread_to_text, threads))
    jsons = list(map(json.dumps, threads))
    output_ds = Dataset.from_dict({'text':samples, 'json':jsons}, split=split)
    return output_ds

oasst2_guanaco = DatasetDict(
        {'train': build_split(ds, 'train'),
         'validation': build_split(ds, 'validation')
        })

oasst2_guanaco.save_to_disk('oasst2_top1_en')
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
