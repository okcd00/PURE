import jsonlines
from tqdm import tqdm
from collections import defaultdict


# in DOC_NER
import sys
sys.path.append('../doc_ner')
from data_utils import *
from file_register import get_path
from modules.Database import Database
from scripts import add_span_in_sample


PURE_JSON_HELP = """
A pure-json is like this:
{
  # document ID (please make sure doc_key can be used to identify a certain document)
  "doc_key": "CNN_ENG_20030306_083604.6",

  # sentences in the document, each sentence is a list of tokens
  "sentences": [
    [...],
    [...],
    ["tens", "of", "thousands", "of", "college", ...],
    ...
  ],

  # entities (boundaries and entity type) in each sentence
  "ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 14, "PER"], ...], #the boundary positions are indexed in the document level
    ...,
  ],

  # relations (two spans and relation type) in each sentence
  "relations": [
    [...],
    [...],
    [[14, 14, 10, 10, "ORG-AFF"], [14, 14, 12, 13, "ORG-AFF"], ...],
    ...
  ]
}
"""


def sldb_to_ner_file(sl_db_path, ner_path, delimeter=' '):
    sl_db = Database(sl_db_path)

    with open(ner_path, 'wb') as f:
        for sample in tqdm(sl_db):
            words = [word for word in sample['words']
                     if word['word'].strip()]

            type_tag = ent_type = sample['target_type']
            l, r = sample['target_span']
            span_length = r - l

            for i, word in enumerate(words):
                if i not in range(l, r):
                    t = 'O'
                elif i == l:
                    t = 'B-{}'.format(type_tag)
                    if span_length == 1:
                        t = 'S-{}'.format(type_tag)
                elif i == r - 1:
                    t = 'E-{}'.format(type_tag)
                else:
                    t = 'M-{}'.format(type_tag)
                f.write('{}{}{}\n'.format(
                    word['word'],
                    delimeter,
                    t
                ).encode('utf-8'))
            f.write('\n'.encode('utf-8'))


def ner_to_pure_json(ner_path, pure_path="", tag_dict=None,
                     doc_key="default-doc", max_sentence_length=500):
    # ner file for one single pure json

    def _defaultdict_int():
        return defaultdict(int)

    ent_len = defaultdict(_defaultdict_int)
    ent_type_set = set()
    ret = [{"doc_key": doc_key,
            "sentences": [],
            "ner": [],
            "relations": []}]

    def _head(text, c):
        if text.strip().__len__() == 0:
            return False
        return text.lower()[0] in c.lower()

    def _tag_type(tag):
        tp = tag.strip().split('-')[1].strip()
        if tag_dict:
            tp = tag_dict.get(tp, tp)
        return tp

    def generate_sample(char_list, tag_list, offset=0):
        words = char_list
        entities = []

        flag, head = 0, -1
        ent_type = ""
        for idx, tag in enumerate(tag_list):
            if flag == 1:
                if _head(tag, 'e'):
                    # no idx+1 here for PURE's span design
                    entities.append(
                        (head + offset, idx + offset, ent_type))
                    ent_len[ent_type][idx - head + 1] += 1
                    ent_type_set.add(ent_type)
                    flag, head = 0, -1
                    ent_type = ""
                elif _head(tag, 'obs'):
                    entities.append(  # o means ends the last
                        # not idx+1 here for PURE's span design
                        (head + offset, idx + offset - 1, ent_type))
                    ent_len[ent_type][idx - head] += 1
                    ent_type_set.add(ent_type)
                    flag, head = 0, -1
                    ent_type = ""
            if flag == 0:
                if _head(tag, 'b'):
                    flag = 1
                    head = idx
                    ent_type = _tag_type(tag)
                elif _head(tag, 's'):
                    # flag = 0
                    head = idx
                    ent_type = _tag_type(tag)
                    entities.append(
                        (head + offset, idx + offset, ent_type))
                    ent_len[ent_type][1] += 1
                    ent_type_set.add(ent_type)
        else:
            if flag == 1:
                entities.append(
                    (head + offset, idx + offset, ent_type))
                ent_len[ent_type][idx - head + 1] += 1
                ent_type_set.add(ent_type)

        return words, entities

    def add_sample(char_list, tag_list, off):
        # split too-long sentences
        while len(char_list) > max_sentence_length:
            pivot = max_sentence_length
            while tag_list[pivot - 1] != 'O':
                pivot -= 1
            print("cut at:", char_list[pivot - 2:pivot + 3],
                  tag_list[pivot - 2:pivot + 3])
            off = add_sample(char_list[:pivot], tag_list[:pivot], off)
            char_list, tag_list = char_list[pivot:], tag_list[pivot:]

        words, entities = generate_sample(
            char_list, tag_list, offset=off)
        ret[0]['sentences'].append(words)
        ret[0]['ner'].append(entities)
        ret[0]['relations'].append([])
        off += len(words)
        return off

    offset = 0
    char_list, tag_list = [], []
    for line in tqdm(open(ner_path, 'rb')):
        if line.strip():
            line = line.strip().decode('utf-8').split()
            if len(line) == 1:
                continue
            c, t = line
            char_list.append(c)
            tag_list.append(t)
        else:
            offset = add_sample(char_list, tag_list, offset)
            char_list, tag_list = [], []
    else:
        if len(char_list) + len(tag_list) > 0:
            offset = add_sample(char_list, tag_list, offset)
            char_list, tag_list = [], []

    print("PURE_PATH", pure_path)
    print("")
    for type_name in ent_len:
        print(type_name, ':', sorted(ent_len[type_name].items()))
    print("")
    print("# sentences:", len(ret[0]['ner']))
    print("# words in longest sentence:", max(map(len, ret[0]['sentences'])))
    print(sorted(ent_type_set))
    if pure_path:
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)
    return ret


def sl_db_to_pure_json():
    dataset_name = 'data'
    for phase in ['train', 'valid', 'test']:
        sl_db_path = get_path('sl_{}_{}'.format(dataset_name, phase))
        # sl_db_path = DATA_PATH + 'sl_{}5texts_{}_folder/'.format(phase, dataset_name)
        print("SLDB_PATH", sl_db_path)

        ret = []
        db = Database(sl_db_path)
        doc2sid = {}
        sid2idx = {}
        for idx, sample in enumerate(db):
            doc = sample['info'].get('doc_id', 'doc-0')
            sid = sample['info']['sid']
            doc2sid.setdefault(doc, [])
            doc2sid[doc].append(sid)
            sid2idx[sid] = idx

        for doc, sid_list in tqdm(doc2sid.items()):
            doc_pivot = 0
            sample = {"doc_key": doc, "sentences": [], "ner": [], "relations": []}
            for sid in sid_list:
                _sp = add_span_in_sample(db[sid2idx[sid]])
                sample["sentences"].append([w['word'] for w in _sp['words']])
                ner_list = []
                for ent in _sp['entities']:
                    l, r = ent['span']
                    r -= 1  # in DQ Chen's PURE project, span is [l, r0]
                    ent_type = ent['type']
                    ner_list.append([l+doc_pivot, r+doc_pivot, ent_type])
                sample["ner"].append(ner_list)
                sample["relations"].append([])
                doc_pivot += len(_sp['words'])
            ret.append(sample)

        pure_path = '/home/chendian/PURE/data/{}/{}.json'.format(
            dataset_name.replace('data', 'findoc'), phase.replace('valid', 'dev'))
        print("PURE_PATH", pure_path)
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)


if __name__ == "__main__":

    resume_tag_dict = {
        'ORG': '公司',
        'EDU': '学历',
        'PRO': '专业',
        'LOC': '地址',
        'NAME': '人名',
        'CONT': '国籍',
        'RACE': '民族',
        'TITLE': '职称',
    }

    phase = 'train'
    dataset_name = 'msra'
    ret_test = ner_to_pure_json(
        ner_path='/home/chendian/PURE/data/{}/{}.ner'.format(dataset_name, phase),
        pure_path='/home/chendian/PURE/data/{}/{}.json'.format(dataset_name, phase),
        # tag_dict={'NR': '人名', 'NS': '地址', 'NT': '公司'},  # msra
        tag_dict={'PER': '人名', 'LOC': '地址', 'ORG': '公司'},  # msra-origin
        # tag_dict=resume_tag_dict,
        # tag_dict={'PER': '人名', 'LOC': '地址', 'ORG': '公司', 'GPE': '位置'},  # onto4
        doc_key=dataset_name + '-doc',
    )

