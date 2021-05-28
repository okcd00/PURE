task_ner_labels = {
    # Chinese Corpus
    'msra':     [u"公司", u"人名", u"地址"],
    'onto4':    ['ORG', 'PER', 'LOC', 'GPE'],
    'resume':   [u"公司", u"人名", u"地址", u"学历", u"专业", u"国籍", u"民族", u"职称"],
    'findoc':   [u"公司", u"人名", u"地址", u"产品业务", u"文件"],
    'ccks':     ['cellno', 'devzone', 'distance', 'poi', 'floorno',
                 'roadno', 'intersection', 'city', 'houseno',
                 'village_group', 'town', 'subpoi', 'assist',
                 'community', 'district', 'road', 'prov'],
    # English Corpus
    'ace04':    ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05':    ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc':   ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}


task_max_span_length = {
    # Chinese Corpus
    'msra': 16,
    'onto4': 16,
    'resume': 16,
    'ccks': 18,
    'findoc': 40,

    # English Corpus
    'ace04': 8,
    'ace05': 8,
    'scierc': 8,
}


task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}


TASK_NAME = 'findoc'
CONFIG_FOR_PURE_API = {
    # basic settings for prediction
    'task': TASK_NAME,
    'do_eval': True,
    'eval_test': True,

    # hyper-parameters
    'context_window': 0,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'max_span_length': task_max_span_length[TASK_NAME],
    'take_context_module': False,

    # path to Bert
    'use_albert': False,
    'model': 'bert-base-chinese',  # use known bert
    'bert_model_dir': '/data/chend/model_files/chinese_L-12_H-768_A-12/',

    # path to the file for predicting, and path to the dump file
    'data_dir': './data/test_files/',  # put test.json here for predicting
    'output_dir': './data/output_files/',  # the results in this directory
    'test_pred_filename': 'ent_pred_test.json',  # the results file's name
}


is_training = False
if is_training:
    CONFIG_FOR_PURE_API.update({
        'seed': 0,
        'num_epoch': 100,
        'learning_rate': 1e-5,
        'task_learning_rate': 1e-4,
        'warmup_proportion': 0.1,
        'print_loss_step': 500,
        'eval_per_epoch': 1,
        'do_train': True,
    })


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
