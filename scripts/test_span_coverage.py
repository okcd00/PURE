import jsonlines
from tqdm import tqdm
from collections import Counter


def flatten(nested_list, unique=False):
    ret = [elem for sub_list in nested_list for elem in sub_list]
    if unique:
        return list(set(ret))
    return ret


ner_length_list = []
# Resume 16 / MSRA 16 / OntoNote4 16 / FinDoc 32
# pure_path = '/home/chendian/PURE/data/onto4/test.json'
pure_path = '/home/chendian/PURE/data/msra/test.json'
with jsonlines.open(pure_path, 'r') as reader:
    for obj in tqdm(reader):
        _nll = [[ent[1]-ent[0]+1 for ent in sent if len(ent)]
                for sent in obj["ner"]]
        ner_length_list.extend(flatten(_nll))
    ct = Counter(ner_length_list)
sum([v for k, v in ct.items() if k <= 16]) / sum(ct.values())
