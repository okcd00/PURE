import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from allennlp.nn.util import batched_index_select
from allennlp.nn import util, Activation
from allennlp.modules import FeedForward

import numpy as np

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import AlbertTokenizer, AlbertPreTrainedModel, AlbertModel

import os
import json
import logging
from pprint import pprint

logger = logging.getLogger('root')


class BertForEntity(BertPreTrainedModel):
    def __init__(self, config, num_ner_labels,
                 max_span_length=10,
                 head_hidden_dim=150,
                 width_embedding_dim=150,
                 args=None,
                 ):
        super().__init__(config)

        self.bert = BertModel(config)
        self.max_span_length = max_span_length
        self.hidden_dropout = nn.Dropout(
            config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim)

        # at least we need width_embedding
        inp_dim = width_embedding_dim

        # how to fusion multi-source features
        self.fusion_method = args.fusion_method if args else 'none'
        from modules.feature_fusion import FeatureFusion
        self.feature_fusion = FeatureFusion(method=self.fusion_method)

        self.take_name_module = args.take_name_module if args else True
        if self.take_name_module:
            name_hidden_size = config.hidden_size  # 768
            inp_dim += name_hidden_size * 2

        self.take_context_module = args.take_context_module if args else False
        if self.take_context_module:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = True

            self.add_pad = torch.nn.ConstantPad2d((0, 1, 0, 0), 0)  # [PAD]
            self.add_cls = torch.nn.ConstantPad2d((1, 0, 0, 0), 101)  # [CLS]
            self.add_sep = torch.nn.ConstantPad2d((0, 1, 0, 0), 102)  # [SEP]

            context_hidden_size = head_hidden_dim  # 150
            self.context_lstm = nn.LSTM(  # or nn.GRU
                input_size=config.hidden_size,  # 768
                hidden_size=context_hidden_size,  # 150
                num_layers=1, # dropout=0.1,
                bidirectional=True)
            inp_dim += context_hidden_size * 2

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=inp_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,  # 150
                        activations=F.relu,
                        dropout=0.2),
            nn.Linear(head_hidden_dim, num_ner_labels)
        )

        self.init_weights()

    def context_hidden(self, input_ids, token_type_ids=None, attention_mask=None):
        # [batch, sequence_length + 1], in case of right overflow
        am = attention_mask
        inp_ids = self.add_pad(input_ids)
        sequence_length = am.sum(-1)

        # ([batch], [batch])
        sent_indexes = torch.arange(am.shape[0], device=am.device)
        char_indexes = sequence_length.long().to(device=am.device)
        indexes = (sent_indexes, char_indexes)
        input_ids_with_sep = inp_ids.index_put_(
            indexes, torch.tensor(102, device=am.device))

        # [batch, sequence_length w/ [CLS] [SEP], embedding_size]
        embeddings = self.bert.embeddings(
            input_ids=input_ids_with_sep,  # self.add_sep(input_ids),
            token_type_ids=token_type_ids)

        # the LSTM part
        packed = pack(embeddings, sequence_length + 1,  # add [SEP]
                      batch_first=True, enforce_sorted=False)
        token_hidden, (h_n, c_n) = self.context_lstm(packed)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = unpack(token_hidden, batch_first=True)[0]

        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = token_hidden.view(token_hidden.shape[0], token_hidden.shape[1], 2, -1)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size] * 2 directions
        return token_hidden[:, :, 0], token_hidden[:, :, 1]

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        embedding_case = []
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,  # [batch_size, sequence_length]
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_end, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)

        # name hidden vectors
        if self.take_name_module:
            spans_start_embedding = batched_index_select(sequence_output, spans_start)
            spans_end_embedding = batched_index_select(sequence_output, spans_end)
            embedding_case.extend([
                spans_start_embedding,
                spans_end_embedding,
            ])

        # width embeddings
        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width = torch.clamp(spans_width, min=None, max=self.max_span_length)
        spans_width_embedding = self.width_embedding(spans_width)
        embedding_case.append(spans_width_embedding)

        # context hidden vectors
        if self.take_context_module:
            # [batch_size, sequence_length w/ [CLS] [SEP], hidden_size]
            context_lef, context_rig = self.context_hidden(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)  # New here
            ctx_left = spans_start - (spans_start > 0).long()
            ctx_right = spans_end + (spans_end > 0).long()
            try:
                ctx_start_embedding = batched_index_select(
                    context_lef, ctx_left)
                ctx_end_embedding = batched_index_select(
                    context_rig, ctx_right)
                embedding_case.extend([
                    ctx_start_embedding,
                    ctx_end_embedding
                ])
            except Exception as e:
                pprint(str(e))
                print(input_ids.shape, context_lef.shape, context_rig.shape)
                print(input_ids)
                print(spans_start - (spans_start > 0).long())
                print(spans_end + (spans_end > 0).long())
                raise ValueError()

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = self.feature_fusion(embedding_case)

        """
        w/ or w/o context hidden vectors.
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        spans_embedding: (batch_size, num_spans, hidden_size*2+head_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, input_ids, spans, spans_mask, spans_ner_label=None, token_type_ids=None, attention_mask=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            # from entity.label_smoothing import LabelSmoothingLoss
            # ls_loss = LabelSmoothingLoss(reduction='sum', smoothing=0.1))
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class AlbertForEntity(AlbertPreTrainedModel):
    def __init__(self, config, num_ner_labels,
                 max_span_length=10,
                 head_hidden_dim=150,
                 width_embedding_dim=150,
                 args=None):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.max_span_length = max_span_length
        self.hidden_dropout = nn.Dropout(
            config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim)

        # at least we need width_embedding
        inp_dim = width_embedding_dim

        self.take_name_module = args.take_name_module if args else True
        if self.take_name_module:
            name_hidden_size = config.hidden_size  # 768
            inp_dim += name_hidden_size * 2

        self.take_context_module = args.take_context_module if args else False
        if self.take_context_module:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = True

            self.add_pad = torch.nn.ConstantPad2d((0, 1, 0, 0), 0)  # [PAD]
            self.add_cls = torch.nn.ConstantPad2d((1, 0, 0, 0), 101)  # [CLS]
            self.add_sep = torch.nn.ConstantPad2d((0, 1, 0, 0), 102)  # [SEP]

            context_hidden_size = head_hidden_dim  # 150
            self.context_lstm = nn.LSTM(  # or nn.GRU
                input_size=config.embedding_size,  # 128
                hidden_size=context_hidden_size,  # 150
                num_layers=1,  # dropout=0.1,
                bidirectional=True)
            inp_dim += context_hidden_size * 2

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=inp_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,  # 150
                        activations=F.relu,
                        dropout=0.2),
            nn.Linear(head_hidden_dim, num_ner_labels)
        )

        self.init_weights()

    def context_hidden(self, input_ids, token_type_ids=None, attention_mask=None):
        # [batch, sequence_length + 1], in case of right overflow
        am = attention_mask
        inp_ids = self.add_pad(input_ids)
        sequence_length = am.sum(-1)

        # ([batch], [batch])
        sent_indexes = torch.arange(am.shape[0], device=am.device)
        char_indexes = sequence_length.long().to(device=am.device)
        indexes = (sent_indexes, char_indexes)
        input_ids_with_sep = inp_ids.index_put_(
            indexes, torch.tensor(102, device=am.device))

        # [batch, sequence_length w/ [CLS] [SEP], embedding_size]
        embeddings = self.albert.embeddings(
            input_ids=input_ids_with_sep,  # self.add_sep(input_ids),
            token_type_ids=token_type_ids)

        # the LSTM part
        packed = pack(embeddings, sequence_length + 1,  # add [SEP]
                      batch_first=True, enforce_sorted=False)
        token_hidden, (h_n, c_n) = self.context_lstm(packed)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = unpack(token_hidden, batch_first=True)[0]

        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = token_hidden.view(token_hidden.shape[0], token_hidden.shape[1], 2, -1)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size] * 2 directions
        return token_hidden[:, :, 0], token_hidden[:, :, 1]

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        embedding_case = []
        sequence_output, pooled_output = self.albert(
            input_ids=input_ids,  # [batch_size, sequence_length]
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_end, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)

        # name hidden vectors
        if self.take_name_module:
            spans_start_embedding = batched_index_select(sequence_output, spans_start)
            spans_end_embedding = batched_index_select(sequence_output, spans_end)
            embedding_case.extend([
                spans_start_embedding,
                spans_end_embedding,
            ])

        # width embeddings
        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width = torch.clamp(spans_width, min=None, max=self.max_span_length)
        spans_width_embedding = self.width_embedding(spans_width)
        embedding_case.append(spans_width_embedding)

        # context hidden vectors
        if self.take_context_module:
            # [batch_size, sequence_length w/ [CLS] [SEP], hidden_size]
            context_lef, context_rig = self.context_hidden(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)  # New here
            ctx_left = spans_start - (spans_start > 0).long()
            ctx_right = spans_end + (spans_end > 0).long()
            try:
                ctx_start_embedding = batched_index_select(context_lef, ctx_left)
                ctx_end_embedding = batched_index_select(context_rig, ctx_right)
                embedding_case.extend([
                    ctx_start_embedding,
                    ctx_end_embedding
                ])
            except Exception as e:
                pprint(str(e))
                print(input_ids.shape, context_lef.shape, context_rig.shape)
                print(input_ids)
                print(spans_start - (spans_start > 0).long())
                print(spans_end + (spans_end > 0).long())
                raise ValueError()

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(embedding_case, dim=-1)

        """
        w/ or w/o context hidden vectors.
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        spans_embedding: (batch_size, num_spans, hidden_size*2+head_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, input_ids, spans, spans_mask, spans_ner_label=None, token_type_ids=None, attention_mask=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            # from entity.label_smoothing import LabelSmoothingLoss
            # ls_loss = LabelSmoothingLoss(reduction='sum', smoothing=0.1))
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class EntityModel():

    def __init__(self, args, num_ner_labels):
        super().__init__()

        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        if args.use_albert:
            # self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = AlbertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=args.max_span_length,
                args=args)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = BertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=args.max_span_length,
                args=args)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)  # Add [CLS] here
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(tokens, spans,
                                                                                               spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list,
                                                                            spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        # logger.info(final_tokens_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_bert_spans_tensor)
        # logger.info(final_bert_spans_tensor.shape)
        # logger.info(final_spans_mask_tensor.shape)
        # logger.info(final_spans_ner_label_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def run_batch(self, samples_list, try_cuda=True, training=True):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list, training)

        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                spans=bert_spans_tensor.to(self._model_device),
                spans_mask=spans_mask_tensor.to(self._model_device),
                spans_ner_label=spans_ner_label_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids=tokens_tensor.to(self._model_device),
                    spans=bert_spans_tensor.to(self._model_device),
                    spans_mask=spans_mask_tensor.to(self._model_device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._model_device),
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict


