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
        self.args = args or DictArgs(CONFIG_FOR_PURE_API)
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

    def load_from_jsonline(self, jsonl_file):
        self.js = [json.loads(line) for line in open(jsonl_file)]
        self.documents = [Document(js) for js in self.js]
        return self.documents

    def save_as_jsonline(self, dict_list, save_path):
        # Save the documents as a jsonline file
        with jsonlines.open(save_path, mode='w') as writer:
            writer.write_all(dict_list)

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


if __name__ == "__main__":
    pure_api = PureApi(load_model_dir='')
