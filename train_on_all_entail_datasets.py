# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import sys
import codecs
import numpy as np
import torch
from torch.nn import functional as F
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
import logging

import json_lines

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaForSequenceClassification

from load_data import store_transformers_models



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, task_label = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task_label = task_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, task_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.task_label = task_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def get_MNLI_train_and_dev(self, train_filename, dev_filename):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''
        examples_per_file = []
        for filename in [train_filename, dev_filename]:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "train-"+str(line_co-1)
                    # text_a = 'MNLI. '+line[8].strip()
                    text_a = line[8].strip()
                    text_b = line[9].strip()
                    label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, task_label=0))
                line_co+=1
                # if line_co > 20000:
                #     break
            readfile.close()
            print('loaded  MNLI size:', len(examples))
            examples_per_file.append(examples)
        return examples_per_file[0], examples_per_file[1] #train, dev

    def get_SNLI_train_and_dev(self, train_filename, dev_filename):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''

        examples_per_file = []
        for filename in [train_filename, dev_filename]:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "train-"+str(line_co-1)
                    # text_a = 'SNLI. '+line[7].strip()
                    text_a = line[7].strip()
                    text_b = line[8].strip()
                    label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, task_label=0))
                line_co+=1
                # if line_co > 20000:
                #     break
            readfile.close()
            print('loaded  SNLI size:', len(examples))
            examples_per_file.append(examples)
        return examples_per_file[0], examples_per_file[1] #train, dev

    def get_SciTail_train_and_dev(self, train_filename, dev_filename):
        '''
        classes: entails, neutral
        '''
        examples_per_file = []
        for filename in [train_filename, dev_filename]:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:

                line=row.strip().split('\t')
                if len(line) == 3:
                    guid = "train-"+str(line_co-1)
                    # text_a = 'SciTail. '+line[0].strip()
                    text_a = line[0].strip()
                    text_b = line[1].strip()
                    label = line[2].strip()
                    if label == 'entails':
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label='entailment', task_label=0))
                    else:
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label='entailment', task_label=1))

            readfile.close()
            print('loaded  SciTail size:', len(examples))
            examples_per_file.append(examples)
        return examples_per_file[0], examples_per_file[1] #train, dev

    def get_RTE_train_and_dev(self, train_filename, dev_filename):
        '''
        classes: entailment, not_entailment
        '''
        examples_per_file = []
        for filename in [train_filename, dev_filename]:
            examples=[]
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "dev-"+str(line_co-1)
                    # text_a = 'RTE. '+line[1].strip()
                    text_a = line[1].strip()
                    text_b = line[2].strip()
                    label = line[3].strip() #["entailment", "not_entailment"]
                    if label == 'entailment':
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label='entailment', task_label = 0))
                    else:
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label='entailment', task_label = 1))
                line_co+=1
            readfile.close()
            print('loaded  RTE size:', len(examples))
            examples_per_file.append(examples)
        return examples_per_file[0], examples_per_file[1] #train, dev

    def get_ANLI_train_and_dev(self, train_prefix, dev_prefix, path):
        #path = /export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1
        folders = ['R1', 'R2', 'R3']
        label2label = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}

        examples_per_file = []
        for prefix in [train_prefix, dev_prefix]:
            examples = []
            guid_id = 0
            # labeltype_set = set()

            for folder in folders:
                filename = path+folder+'/'+prefix+'.jsonl'
                # print('loading ANLI...', filename)
                with open(filename, 'r') as f:
                    for line in json_lines.reader(f):
                        guid_id+=1
                        # premise = 'ANLI. '+line.get('context')
                        premise = line.get('context')
                        hypothesis = line.get('hypothesis')

                        # label = 'entailment' if line.get('label') == 'e' else 'not_entailment'
                        label = label2label.get(line.get('label').strip())
                        # labeltype_set.add(line.get('label'))
                        if len(premise) == 0 or len(hypothesis)==0:
                            continue
                        examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=hypothesis, label=label, task_label = 0))
            print(prefix, ' ANLI size:', len(examples))
            examples_per_file.append(examples)
        return examples_per_file[0], examples_per_file[1] #train, dev


    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              task_label = example.task_label))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()










def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")


    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



    task_name = args.task_name.lower()


    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()


    num_labels = len(["entailment", "neutral", "contradiction"])
    # pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli'
    pretrain_model_dir = '/export/home/Dataset/BERT_pretrained_mine/TrainedModelReminder/RoBERTa_on_MNLI_SNLI_SciTail_RTE_ANLI_SpecialToken_epoch_2_acc_4.156359461121103' #'roberta-large' , 'roberta-large-mnli'
    model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)



    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)




    #MNLI-SNLI-SciTail-RTE-SICK
    train_examples_MNLI, dev_examples_MNLI = processor.get_MNLI_train_and_dev('/export/home/Dataset/glue_data/MNLI/train.tsv', '/export/home/Dataset/glue_data/MNLI/dev_mismatched.tsv') #train_pu_half_v1.txt
    train_examples_SNLI, dev_examples_SNLI = processor.get_SNLI_train_and_dev('/export/home/Dataset/glue_data/SNLI/train.tsv', '/export/home/Dataset/glue_data/SNLI/dev.tsv')
    train_examples_SciTail, dev_examples_SciTail = processor.get_SciTail_train_and_dev('/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_train.tsv', '/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_dev.tsv')
    train_examples_RTE, dev_examples_RTE = processor.get_RTE_train_and_dev('/export/home/Dataset/glue_data/RTE/train.tsv', '/export/home/Dataset/glue_data/RTE/dev.tsv')
    train_examples_ANLI, dev_examples_ANLI = processor.get_ANLI_train_and_dev('train', 'dev', '/export/home/Dataset/para_entail_datasets/ANLI/anli_v0.1/')

    train_examples = train_examples_MNLI+train_examples_SNLI+train_examples_SciTail+train_examples_RTE+train_examples_ANLI
    dev_examples_list = [dev_examples_MNLI, dev_examples_SNLI, dev_examples_SciTail, dev_examples_RTE, dev_examples_ANLI]

    dev_task_label = [0,0,1,1,0]
    task_names = ['MNLI', 'SNLI', 'SciTail', 'RTE', 'ANLI']


    '''filter challenging neighbors'''
    neighbor_id_list = []
    readfile = codecs.open('neighbors_indices_before_dropout_eud.v3.txt', 'r', 'utf-8')
    for line in readfile:
        neighbor_id_list.append(int(line.strip()))
    readfile.close()
    print('neighbor_id_list size:', len(neighbor_id_list))
    truncated_train_examples = [train_examples[i] for i in neighbor_id_list]
    train_examples = truncated_train_examples




    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()



    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_task_label_ids = torch.tensor([f.task_label for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_task_label_ids)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    '''dev data to features'''
    valid_dataloader_list = []
    for valid_examples_i in dev_examples_list:
        valid_features = convert_examples_to_features(
            valid_examples_i, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        logger.info("***** valid_examples *****")
        logger.info("  Num examples = %d", len(valid_examples_i))
        valid_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_task_label_ids = torch.tensor([f.task_label for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label_ids, valid_task_label_ids)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)
        valid_dataloader_list.append(valid_dataloader)


    iter_co = 0
    for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, task_label_ids = batch
            logits = model(input_ids, input_mask, None, labels=None)

            prob_matrix = F.log_softmax(logits[0].view(-1, num_labels), dim=1)
            '''this step *1.0 is very important, otherwise bug'''
            new_prob_matrix = prob_matrix*1.0
            '''change the entail prob to p or 1-p'''
            changed_places = torch.nonzero(task_label_ids, as_tuple=False)
            new_prob_matrix[changed_places, 0] = 1.0 - prob_matrix[changed_places, 0]

            loss = F.nll_loss(new_prob_matrix, label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()


            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            iter_co+=1

            # if iter_co % len(train_dataloader) ==0:
            if iter_co % (len(train_dataloader)//5) ==0:
                '''
                start evaluate on  dev set after this epoch
                '''
                # if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                #     model = torch.nn.DataParallel(model)
                model.eval()
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.track_running_stats=False
                # logger.info("***** Running evaluation *****")
                # logger.info("  Num examples = %d", len(valid_examples_MNLI))
                # logger.info("  Batch size = %d", args.eval_batch_size)

                dev_acc_sum = 0.0
                for idd, valid_dataloader in enumerate(valid_dataloader_list):
                    task_label = dev_task_label[idd]
                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    gold_label_ids = []
                    # print('Evaluating...', task_label)
                    # for _, batch in enumerate(tqdm(valid_dataloader, desc=task_names[idd])):
                    for _, batch in enumerate(valid_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, label_ids, task_label_ids = batch
                        if task_label == 0:
                            gold_label_ids+=list(label_ids.detach().cpu().numpy())
                        else:
                            '''SciTail, RTE'''
                            task_label_ids_list = list(task_label_ids.detach().cpu().numpy())
                            gold_label_batch_fake = list(label_ids.detach().cpu().numpy())
                            for ex_id, label_id in enumerate(gold_label_batch_fake):
                                if task_label_ids_list[ex_id] ==  0:
                                    gold_label_ids.append(label_id) #0
                                else:
                                    gold_label_ids.append(1) #1
                        with torch.no_grad():
                            logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=None, labels=None)
                        logits = logits[0]
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                    preds = preds[0]
                    pred_probs = softmax(preds,axis=1)
                    pred_label_ids_3way = np.argmax(pred_probs, axis=1)
                    if task_label == 0:
                        '''3-way tasks MNLI, SNLI, ANLI'''
                        pred_label_ids = pred_label_ids_3way
                    else:
                        '''SciTail, RTE'''
                        pred_label_ids = []
                        for pred_label_i in pred_label_ids_3way:
                            if pred_label_i == 0:
                                pred_label_ids.append(0)
                            else:
                                pred_label_ids.append(1)
                    assert len(pred_label_ids) == len(gold_label_ids)
                    hit_co = 0
                    for k in range(len(pred_label_ids)):
                        if pred_label_ids[k] == gold_label_ids[k]:
                            hit_co +=1
                    test_acc = hit_co/len(gold_label_ids)
                    dev_acc_sum+=test_acc
                    print(task_names[idd], ' dev acc:', test_acc)

                '''store the model, because we can test after a max_dev acc reached'''
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                store_transformers_models(model_to_save, tokenizer, '/export/home/Dataset/BERT_pretrained_mine/TrainedModelReminder/', 'RoBERTa_on_MNLI_SNLI_SciTail_RTE_ANLI_SpecialToken_Filter_1_epoch_'+str(epoch_i)+'_acc_'+str(dev_acc_sum))






if __name__ == "__main__":
    main()

'''
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -u train_on_all_entail_datasets.py --task_name rte --do_lower_case --learning_rate 2e-6 --num_train_epochs 100 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64

note:
RTE--> MNLI, SNLI, SciTail, RTE, ANLI
remove special token, using roberta-large, check after each epoch

RoBERTa_on_MNLI_SNLI_SciTail_RTE_ANLI_SpecialToken_Filter_1_epoch_74_acc_4.1927169133373905
RoBERTa_on_MNLI_SNLI_SciTail_RTE_ANLI_SpecialToken_Filter_1_epoch_52_acc_4.195574264788893
RoBERTa_on_MNLI_SNLI_SciTail_RTE_ANLI_SpecialToken_Filter_1_epoch_51_acc_4.199802825942953
MNLI  dev acc: 0.9011391375101708
SNLI  dev acc: 0.9285714285714286
SciTail  dev acc: 0.9631901840490797
RTE  dev acc: 0.9025270758122743
ANLI  dev acc: 0.504375

'''
