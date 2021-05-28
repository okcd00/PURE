import sys
import jsonlines
from tqdm import tqdm
sys.path.append('/home/chendian/doc_ner')


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


def ner_to_pure_json(ner_path, pure_path):
    # ner file for one single pure json
    ret = [{"doc_key": "default-doc",
            "sentences": [],
            "ner": [],
            "relations": []}]

    def _head(text, c):
        if text.strip().__len__() == 0:
            return False
        return text.lower()[0] in c.lower()

    def generate_sample(char_list, tag_list, offset=0):
        words = char_list
        entities = []

        flag, head = 0, -1
        ent_type = ""
        for idx, tag in enumerate(tag_list):
            if flag == 0 and _head(tag, 'b'):
                flag = 1
                head = idx
                ent_type = tag.strip().split('-')[1]
            if flag == 1 and _head(tag, 'eso'):
                # no idx+1 here for PURE's span design
                entities.append(
                    (head + offset, idx + offset, ent_type))
                flag, head = 0, -1
                ent_type = ""
        else:
            if flag == 1:
                entities.append(
                    (head + offset, idx + offset, ent_type))

        return words, entities

    offset = 0
    char_list, tag_list = [], []
    for line in tqdm(open(ner_path, 'rb')):
        if line.strip():
            line = line.strip().decode('utf-8').split()
            c, t = line
            char_list.append(c)
            tag_list.append(t)
        else:
            words, entities = generate_sample(
                char_list, tag_list, offset=offset)
            ret[0]['sentences'].append(words)
            ret[0]['ner'].append(entities)
            offset += len(words)
            char_list, tag_list = [], []
    else:
        if len(char_list) + len(tag_list) > 0:
            words, entities = generate_sample(
                char_list, tag_list)
            ret[0]['sentences'].append(words)
            ret[0]['ner'].append(entities)
            offset += len(words)
            char_list, tag_list = [], []

    print("PURE_PATH", pure_path)
    with jsonlines.open(pure_path, mode='w') as writer:
        writer.write_all(ret)
    return ret


def sldb_to_pure_json(sl_db_path, pure_path='/home/chendian/PURE/data/test_msra/dev.json'):

    import jsonlines
    from tqdm import tqdm

    # in doc_ner
    from modules.Database import Database
    from scripts import add_span_in_sample
    # phase = 'train'
    # sl_db_path = get_path('sl_data_{}'.format(phase))
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
            l, r = _sp['target_span']
            ent_type = _sp['target_type']
            ner_list.append([l + doc_pivot, r + doc_pivot - 1, ent_type])
            """
            for ent in _sp['entities']:
                l, r = ent['span']
                r -= 1  # in DQ Chen's PURE project, span is [l, r0]
                ent_type = ent['type']
                ner_list.append([l + doc_pivot, r + doc_pivot, ent_type])
            """
            sample["ner"].append(ner_list)
            sample["relations"].append([])
            doc_pivot += len(_sp['words'])
        ret.append(sample)

    # pure_path = '/home/chendian/PURE/data/findoc/{}.json'.format(phase.replace('valid', 'dev'))
    print("PURE_PATH", pure_path)
    with jsonlines.open(pure_path, mode='w') as writer:
        writer.write_all(ret)
    return ret


if __name__ == "__main__":
    pass
