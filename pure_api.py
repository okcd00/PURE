# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : pure_api.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-05-28
#   desc     : api class for loading and inference
# ==========================================================================
from run_entity import *
import jsonlines
from shared.data_structures import DictArgs, Document
from shared.const import task_ner_labels, CONFIG_FOR_PURE_API


class PureApi(object):
    # For inferring only.
    def __init__(self, args=None, load_model_dir=''):
        self.model = None
        self.device = torch.device('cuda:0')

        # data containers
        self.js = None
        self.documents = None

        # configs, from ArgParses or DictArgs
        if args:
            if isinstance(args, dict):
                args = DictArgs(args)
            self.args = args
        else:
            self.args = DictArgs(CONFIG_FOR_PURE_API)
        self.ner_label2id, self.ner_id2label = get_labelmap(
            task_ner_labels[self.args.task])

        # load models
        if load_model_dir:
            self.init_models(self.args, load_model_dir)

    def init_models(self, args, load_model_dir):
        # load from a directory with vocab, config and model.
        args.bert_model_dir = load_model_dir
        num_ner_labels = len(task_ner_labels[args.task]) + 1
        self.model = EntityModel(args, num_ner_labels=num_ner_labels)

    def generate_document_from_sentences(self, sentences, save_path=None):
        # generate document obj without labels
        dict_list = [{  # in this case, there's only one doc
            "doc_key": 'default-doc',
            "sentences": [[c for c in text] for text in sentences],
            "ner": [[]] * len(sentences),
            "relations": [[]] * len(sentences)}]
        if save_path:
            self.save_as_jsonline(dict_list, save_path)
        self.js = dict_list
        self.documents = [Document(js) for js in self.js]
        return self.documents

    def save_as_jsonline(self, dict_list, save_path):
        # Save the documents as a jsonline file
        with jsonlines.open(save_path, mode='w') as writer:
            writer.write_all(dict_list)

    def load_from_jsonline(self, jsonl_file):
        self.js = [json.loads(line) for line in open(jsonl_file)]
        self.documents = [Document(js) for js in self.js]
        return self.documents

    def dump_prediction(self, save_path):
        # Save the prediction as a json file
        with open(save_path, 'w') as f:
            # turn numpy objects into python objects
            f.write('\n'.join(json.dumps(doc, cls=NpEncoder)
                              for doc in self.js))

    def turn_documents_into_batches(self, test_data):
        test_samples, test_ner = convert_dataset_to_samples(
            test_data, self.args.max_span_length,
            ner_label2id=self.ner_label2id,
            context_window=self.args.context_window)
        test_batches = batchify(
            test_samples, self.args.eval_batch_size)
        return test_batches

    def ner_predictions(self, batches, js=None):
        ner_result = {}
        span_hidden_table = {}
        tot_pred_ett = 0
        for i in range(len(batches)):
            output_dict = self.model.run_batch(
                batches[i], training=False)
            pred_ner = output_dict['pred_ner']
            for sample, preds in zip(batches[i], pred_ner):
                off = sample['sent_start_in_doc'] - sample['sent_start']
                k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
                ner_result[k] = []
                for span, pred in zip(sample['spans'], preds):
                    span_id = '%s::%d::(%d,%d)' % (
                        sample['doc_key'],
                        sample['sentence_ix'],
                        span[0] + off, span[1] + off)
                    if pred == 0:
                        continue
                    ner_result[k].append([
                        span[0] + off, span[1] + off,
                        self.ner_id2label[pred]
                    ])
                tot_pred_ett += len(ner_result[k])

        logger.info('Total pred entities: %d' % tot_pred_ett)

        if js is None:
            js = self.js

        for i, doc in enumerate(js):
            doc["predicted_ner"] = []
            doc["predicted_relations"] = []
            for j in range(len(doc["sentences"])):
                k = doc['doc_key'] + '-' + str(j)
                if k in ner_result:
                    doc["predicted_ner"].append(ner_result[k])
                else:
                    logger.info('%s not in NER results!' % k)
                    doc["predicted_ner"].append([])

                doc["predicted_relations"].append([])
            js[i] = doc
        return js

    def output_results(self, js=None):
        # consider about doc-level NER
        outputs = {}
        if js is None:
            js = self.js
        for doc in js:
            doc_id = doc['doc_key']
            results_in_doc = []
            sentences, ner = doc['sentences'], doc["predicted_ner"]
            offset = 0
            for sent, ents in zip(sentences, ner):
                results_in_sent = []
                sent_len = len(sent)
                # sentence_text = ''.join(sent)
                for l, r, tp in ents:
                    results_in_sent.append(dict(
                        value=''.join(sent[l - offset: r - offset + 1]),
                        span=[int(l - offset), int(r - offset + 1)],
                        type=tp
                    ))
                results_in_doc.append(results_in_sent)
                offset += sent_len
            outputs[doc_id] = results_in_doc
        return outputs

    def output_results_for_p5(self, js):
        # no ``docs'' here, inputs are in a sentence text list.
        if js is None:
            js = self.js
        doc = js[0]
        results_in_doc = []
        sentences, ner = doc['sentences'], doc['predicted_ner']
        offset = 0
        for sent, ents in zip(sentences, ner):
            results_in_sent = []
            sent_len = len(sent)
            # sentence_text = ''.join(sent)
            for l, r, tp in ents:
                results_in_sent.append(dict(
                    value=''.join(sent[l - offset: r - offset + 1]),
                    span=[int(l - offset), int(r - offset + 1)],
                    type=tp
                ))
            results_in_doc.append(results_in_sent)
            offset += sent_len
        return results_in_doc

    def output_results_for_ccks(self, js):
        # no ``docs'' here, inputs are in a sentence text list.
        if js is None:
            js = self.js
        doc = js[0]
        results = []
        sentences, ner = doc['sentences'], doc['predicted_ner']
        offset = 0
        for s_idx, (sent, ents) in enumerate(zip(sentences, ner)):
            sent_len = len(sent)
            tags = ['O'] * sent_len
            # sentence_text = ''.join(sent)
            for l, r, tp in ents:
                for pivot in range(l - offset, r - offset + 1):
                    if pivot == l - offset:
                        pos_t = 'B'
                        if l == r:
                            pos_t = 'S'
                    elif pivot == r - offset:
                        pos_t = 'E'
                    else:
                        pos_t = 'I'
                    tags[pivot] = '{}-{}'.format(pos_t, tp)
            results.append(u'\u0001'.join([
                str(s_idx + 1),
                ''.join(sent),
                ' '.join(tags)
            ]))
            offset += sent_len
        return results

    def evaluate(self, documents=None):
        # predict on documents with labels, evaluate for performance
        test_samples, test_ner = convert_dataset_to_samples(
            documents, self.args.max_span_length,
            ner_label2id=self.ner_label2id,
            context_window=self.args.context_window)
        test_batches = batchify(
            test_samples, self.args.eval_batch_size)
        f1 = evaluate(self.model, test_batches, test_ner)
        return f1

    def extract(self, documents=None):
        if documents is None:
            # or load documents with
            # documents = self.load_from_jsonline('./test.json')
            documents = self.documents
        test_batches = self.turn_documents_into_batches(
            test_data=documents)
        self.js = self.ner_predictions(
            batches=test_batches, js=self.js)
        return self.js

    def batch_extract(self, sentences, output_method='p5'):
        documents = self.generate_document_from_sentences(sentences)
        answers = self.extract(documents)
        if output_method in ['ccks']:
            return self.output_results_for_ccks(answers)
        if output_method in ['p5']:
            return self.output_results_for_p5(answers)
        return self.output_results(answers)


def test_10000_cases():
    pure_api = PureApi(
        load_model_dir='/home/chendian/PURE/output_dir/findoc_old/')

    # input:  ['今天我在庖丁科技有限公司吃饭。',
    #          '拼多多这个公司真是太拼了。']
    documents = pure_api.load_from_jsonline(
        jsonl_file='/home/chendian/PURE/data/findoc/test.json')

    result_js = pure_api.extract(documents=documents)

    # output: [[{'span': [4, 12], 'value': '庖丁科技有限公司', 'type': 'company'}],
    #          [{'span': [0, 3], 'value': '拼多多', 'type': 'company'}]]
    answers = pure_api.output_results_for_p5(js=result_js)
    return answers


if __name__ == "__main__":
    pa = PureApi(args=None, load_model_dir='/home/chendian/PURE/output_dir/findoc_old/')
    answers = pa.batch_extract(['庖丁科技是一家金融科技公司', '国务院发布了新的《行政管理办法》'])
