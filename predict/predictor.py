
import json
import os, sys

sys.path.append("../tplinker_plus")

from tqdm import tqdm
import re
import pandas as pd
from IPython.core.debugger import set_trace
from pprint import pprint
import unicodedata
from transformers import BertModel, BasicTokenizer, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
from common.utils import Preprocessor
from tplinker_plus_run import (DataMaker4Bert,
                           DataMaker4BiLSTM,
                           TPLinkerPlusBert,
                           TPLinkerPlusBiLSTM,
                           MetricsCalculator)
import wandb
import config
from glove import Glove
import numpy as np

class HandshakingTaggingScheme(object):
    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"
        self.link_types = {"SH2OH",  # subject head to object head
                           "OH2SH",  # object head to subject head
                           "ST2OT",  # subject tail to object tail
                           "OT2ST",  # object tail to subject tail
                           }
        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.link_types}

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        '''
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)

        #         # if entity_list exist, need to distinguish entity types
        #         if self.ent2id is not None and "entity_list" in sample:
        for ent in sample["entity_list"]:
            add_spot(
                (ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            #             if self.ent2id is None: # set all entities to default type
            #                 add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            #                 add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                add_spot((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                add_spot((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
            if subj_tok_span[1] <= obj_tok_span[1]:
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        '''
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return:
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag, shaking_tag_conf):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx, shaking_tag_conf[shaking_idx, tag_idx].item())
            spots.append(spot)
        return spots

    def decode_rel(self,
                   text,
                   shaking_tag,
                   shaking_tag_conf,
                   tok2char_span,
                   tok_offset=0, char_offset=0):
        '''
        shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag, shaking_tag_conf)

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            if link_type != "EH2ET" or sp[0] > sp[
                1]:  # for an entity, the start position can not be larger than the end pos.
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            if not ent_text:
                continue
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
                "conf": sp[3]
            }
            head_key = str(sp[0])  # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = str(sp[0]), str(sp[1])
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = str(sp[1]), str(sp[0])
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            subj_list = head_ind2entities[subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": rel,
                        "conf": sp[3]
                    })

        # recover the positons in the original text
        for ent in ent_list:
            ent["char_span"] = [ent["char_span"][0] + char_offset, ent["char_span"][1] + char_offset]
            ent["tok_span"] = [ent["tok_span"][0] + tok_offset, ent["tok_span"][1] + tok_offset]

        return rel_list, ent_list

    def trans2ee(self, rel_list, ent_list):
        sepatator = "_"  # \u2E80
        trigger_set, arg_iden_set, arg_class_set = set(), set(), set()
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
            _, event_type = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                event_type, 0) + 1

        # get candidate trigger types from entity types
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger":  # trigger
                event_type = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1.1  # if even, entity type makes the call

        # voting
        tirigger_offset2event = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            event_type = sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            tirigger_offset2event[trigger_offet_str] = event_type  # final event type

        # generate event list
        trigger_offset2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, event_type = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            if tirigger_offset2event[trigger_offset_str] != event_type:  # filter false relations
                #                 set_trace()
                continue

            # append arguments
            if trigger_offset_str not in trigger_offset2arguments:
                trigger_offset2arguments[trigger_offset_str] = []
            trigger_offset2arguments[trigger_offset_str].append({
                "text": rel["subject"],
                "type": argument_role,
                "char_span": rel["subj_char_span"],
                "tok_span": rel["subj_tok_span"],
            })
        event_list = []
        for trigger_offset_str, event_type in tirigger_offset2event.items():
            arguments = trigger_offset2arguments[
                trigger_offset_str] if trigger_offset_str in trigger_offset2arguments else []
            event = {
                "trigger": trigger_offset2trigger_text[trigger_offset_str],
                "trigger_char_span": trigger_offset2trigger_char_span[trigger_offset_str],
                "trigger_tok_span": trigger_offset_str.split(","),
                "trigger_type": event_type,
                "argument_list": arguments,
            }
            event_list.append(event)
        return event_list

# %%

config = config.eval_config

# %%

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
# batch_size = config["hyper_parameters"]["batch_size"]
# rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
# ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
# max_seq_len = config["hyper_parameters"]["max_seq_len"]
# for reproductivity
torch.backends.cudnn.deterministic = True

# %% md

# Load Data

test_data_path_dict = {}
for file_path in glob.glob(test_data_path):
    file_name = re.search("(.*?)\.json", file_path.split("/")[-1]).group(1)
    test_data_path_dict[file_name] = file_path

test_data_dict = {}
for file_name, path in test_data_path_dict.items():
    test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))

# ori_test_data_dict = copy.deepcopy(test_data_dict)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Merics

# %%

# metrics = MetricsCalculator(handshaking_tagger)


# get model state paths
model_state_dir = config["model_state_dict_dir"]
target_run_ids = set(config["run_ids"])
run_id2model_state_paths = {}
for root, dirs, files in os.walk(model_state_dir):
    for file_name in files:
        run_id = root.replace('/files', '')[-8:]
        if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
            if run_id not in run_id2model_state_paths:
                run_id2model_state_paths[run_id] = []
            model_state_path = os.path.join(root, file_name)
            run_id2model_state_paths[run_id].append(model_state_path)

# %%

def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key=lambda x: int(re.search("(\d+)", x.split("/")[-1]).group(1)))
    #     pprint(path_list)
    return path_list[-k:]


# %%

# only last k models
k = config["last_k_model"]
for run_id, path_list in run_id2model_state_paths.items():
    run_id2model_state_paths[run_id] = get_last_k_paths(path_list, k)


# %%

def filter_duplicates(rel_list, ent_list):
    rel_memory_set = set()
    filtered_rel_list = []

    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                 rel["subj_tok_span"][1],
                                                                 rel["predicate"],
                                                                 rel["obj_tok_span"][0],
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)

    ent_memory_set = set()
    filtered_ent_list = []
    for ent in ent_list:
        ent_memory = "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0],
                                                 ent["tok_span"][1],
                                                 ent["type"])
        if ent_memory not in ent_memory_set:
            filtered_ent_list.append(ent)
            ent_memory_set.add(ent_memory)

    return filtered_rel_list, filtered_ent_list


# %%
def get_ent_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ent_cpg, rel_cpg):
    pred_ent_mem_list = [(*ent["char_span"], ent['type']) for ent in pred_ent_list]
    gold_ent_mem_list = [(*ent["char_span"], ent['type']) for ent in gold_ent_list]
    for ent_type, value, in ent_cpg.items():
        value[2] += len([gold_ent for gold_ent in gold_ent_list if gold_ent["type"] == ent_type])

    for ent_mem in pred_ent_mem_list:
        pred_ent_type = ent_mem[-1]
        ent_cpg[pred_ent_type][1] += 1
        if ent_mem in gold_ent_mem_list:
            ent_cpg[pred_ent_type][0] += 1

    pred_rel_mem_list = [(*rel["subj_char_span"], *rel["obj_char_span"], rel['predicate']) for rel in pred_rel_list]
    gold_rel_mem_list = [(*rel["subj_char_span"], *rel["obj_char_span"], rel['predicate']) for rel in gold_rel_list]
    for rel_type, value, in rel_cpg.items():
        value[2] += len([gold_rel for gold_rel in gold_rel_list if gold_rel['predicate'] == rel_type])

    for rel_mem in pred_rel_mem_list:
        pred_rel_type = rel_mem[-1]
        rel_cpg[pred_rel_type][1] += 1
        if rel_mem in gold_rel_mem_list:
            rel_cpg[pred_rel_type][0] += 1


class Predictor():
    def __init__(self, config):
        # self.config = config.eval_config

        self.config = config
        self.data_home = config["data_home"]
        self.experiment_name = config["exp_name"]
        self.test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
        self.batch_size = config["hyper_parameters"]["batch_size"]
        self.rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
        self.ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
        self.save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
        self.max_seq_len = config["hyper_parameters"]["max_seq_len"]
        self.rel2id = json.load(open(self.rel2id_path, "r", encoding="utf-8"))
        self.ent2id = json.load(open(self.ent2id_path, "r", encoding="utf-8"))
        self.handshaking_tagger = HandshakingTaggingScheme(self.rel2id, self.max_seq_len, self.ent2id)
        self.tag_size = self.handshaking_tagger.get_tag_size()
        self.metrics = MetricsCalculator(self.handshaking_tagger)

        # Dataset
        if config["encoder"] == "BERT":
            tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False,
                                                          do_lower_case=False)
            tokenize = tokenizer.tokenize
            self.data_maker = DataMaker4Bert(tokenizer, self.handshaking_tagger)
            get_tok2char_span_map = lambda text: \
                tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        elif config["encoder"] in {"BiLSTM", }:
            token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
            token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
            idx2token = {idx: tok for tok, idx in token2idx.items()}

            tokenize = lambda text: text.split(" ") if config["ignore_subword"] else list(text)

            def get_tok2char_span_map(text):
                tokens = text.split(" ")
                tok2char_span = []
                char_num = 0
                for tok in tokens:
                    tok2char_span.append((char_num, char_num + len(tok)))
                    char_num += len(tok) + 1  # +1: whitespace
                return tok2char_span

            def text2indices(text, max_seq_len):
                input_ids = []
                tokens = text.split(" ") if config["ignore_subword"] else list(text)
                for tok in tokens:
                    if tok not in token2idx:
                        input_ids.append(token2idx['<UNK>'])
                    else:
                        input_ids.append(token2idx[tok])
                if len(input_ids) < max_seq_len:
                    input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
                input_ids = torch.tensor(input_ids[:max_seq_len])
                return input_ids

            self.data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, self.self.handshaking_tagger)

        # Model
        if config["encoder"] == "BERT":
            encoder = BertModel.from_pretrained(config["bert_path"])
            hidden_size = encoder.config.hidden_size
            self.rel_extractor = TPLinkerPlusBert(encoder,
                                                  self.tag_size,
                                                  config["hyper_parameters"]["shaking_type"],
                                                  config["hyper_parameters"]["inner_enc_type"],
                                                  )
            tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False,
                                                          do_lower_case=False)
            self.tokenize = tokenizer.tokenize
            get_tok2char_span_map = lambda text: \
                tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

        elif config["encoder"] in {"BiLSTM", }:
            glove = Glove()
            glove = glove.load(config["pretrained_word_embedding_path"])

            # prepare embedding matrix
            word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), config["word_embedding_dim"]))
            count_in = 0

            # 在预训练词向量中的用该预训练向量
            # 不在预训练集里的用随机向量
            for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
                if tok in glove.dictionary:
                    count_in += 1
                    word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]

            print(
                "{:.4f} tokens are in the pretrain word embedding matrix".format(
                    count_in / len(idx2token)))  # 命中预训练词向量的比例
            word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

            self.rel_extractor = TPLinkerPlusBiLSTM(word_embedding_init_matrix,
                                                    config["emb_dropout"],
                                                    config["enc_hidden_size"],
                                                    config["dec_hidden_size"],
                                                    config["rnn_dropout"],
                                                    tag_size,
                                                    config["hyper_parameters"]["shaking_type"],
                                                    config["hyper_parameters"]["inner_enc_type"],
                                                    )

            self.tokenize = lambda text: text.split(" ")

            def get_tok2char_span_map(text):
                tokens = text.split(" ")
                tok2char_span = []
                char_num = 0
                for tok in tokens:
                    tok2char_span.append((char_num, char_num + len(tok)))
                    char_num += len(tok) + 1  # +1: whitespace
                return tok2char_span

        self.preprocessor = Preprocessor(tokenize_func=tokenize,
                                         get_tok2char_span_map_func=get_tok2char_span_map)
        self.rel_extractor = self.rel_extractor.to(device)

    def predict(self, test_data):
        '''
        test_data: if split, it would be samples with subtext
        ori_test_data: the original data has not been split, used to get original text here
        '''
        ori_data = copy.deepcopy(test_data)
        max_tok_num = 0
        for sample in tqdm(ori_data, desc="Calculate the max token number"):
            tokens = self.tokenize(sample["text"])
            max_tok_num = max(len(tokens), max_tok_num)

        # %%

        split_test_data = False
        if max_tok_num > self.max_seq_len:
            split_test_data = True
            print(
                "max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num,
                                                                                                     self.max_seq_len))
        else:
            print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num,
                                                                                                          self.max_seq_len))
        self.max_seq_len = min(max_tok_num, self.max_seq_len)
        short_data = self.preprocessor.split_into_short_samples(ori_data,
                                                                self.max_seq_len,
                                                                sliding_len=config["hyper_parameters"][
                                                                    "sliding_len"],
                                                                encoder=config["encoder"],
                                                                data_type="test")
        indexed_test_data = self.data_maker.get_indexed_data(short_data, self.max_seq_len,
                                                        data_type="test")  # fill up to max_seq_len
        test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False,
                                     collate_fn=lambda data_batch: self.data_maker.generate_batch(data_batch, data_type="test"),
                                     )

        tmp_data = []
        pred_sample_list = []
        for i, batch_test_data in enumerate(tqdm(test_dataloader, desc="Predicting")):
            if self.config["encoder"] == "BERT":
                sample_list, batch_input_ids, \
                batch_attention_mask, batch_token_type_ids, \
                tok2char_span_list, _ = batch_test_data

                batch_input_ids, \
                batch_attention_mask, \
                batch_token_type_ids = (batch_input_ids.to(device),
                                        batch_attention_mask.to(device),
                                        batch_token_type_ids.to(device),
                                        )

            elif self.config["encoder"] in {"BiLSTM", }:
                sample_list, batch_input_ids, \
                tok2char_span_list, _ = batch_test_data

                batch_input_ids = batch_input_ids.to(device)

            with torch.no_grad():
                if self.config["encoder"] == "BERT":
                    batch_pred_shaking_tag, _ = self.rel_extractor(batch_input_ids,
                                                              batch_attention_mask,
                                                              batch_token_type_ids,
                                                              )
                elif self.config["encoder"] in {"BiLSTM", }:
                    batch_pred_shaking_tag, _ = self.rel_extractor(batch_input_ids)

            batch_pred_shaking_tag_conf = batch_pred_shaking_tag.softmax(dim=-1)
            batch_pred_shaking_tag = (batch_pred_shaking_tag > 0.).long()

            for ind in range(len(sample_list)):
                gold_sample = sample_list[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                tok2char_span = tok2char_span_list[ind]
                pred_shaking_tag = batch_pred_shaking_tag[ind]
                pred_shaking_tag_conf = batch_pred_shaking_tag_conf[ind]
                tok_offset, char_offset = 0, 0
                if split_test_data:
                    tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
                rel_list, ent_list = self.handshaking_tagger.decode_rel(text,
                                                                   pred_shaking_tag,
                                                                   pred_shaking_tag_conf,
                                                                   tok2char_span,
                                                                   tok_offset=tok_offset, char_offset=char_offset)
                pred_sample_list.append({
                    "text": text,
                    "id": text_id,
                    "relation_list": rel_list,
                    "entity_list": ent_list,
                })
                ent_reason = ""
                for item in ent_list:
                    if item['conf'] < self.config['activate_learning_thresh']:
                        ent_reason += f"{item['type']}(text='{item['text']}', offset={item['char_span']}, conf={item['conf']})\n"
                rel_reason = ""
                for item in rel_list:
                    rel_reason += f"{item['predicate']}(subject='{item['subject']}', subj_offset={item['subj_char_span']}, " \
                                  f"object='{item['object']}', obj_offset={item['obj_char_span']}, conf={item['conf']:.4f})\n"

                if ent_reason or rel_reason:
                    tmp_data.append((gold_sample['id'], ent_reason.strip(), rel_reason.strip()))

        df_al_ = pd.DataFrame(tmp_data, columns=["filename", "ents", "rels"])

        # merge
        text_id2pred_res = {}
        for sample in pred_sample_list:
            text_id = sample["id"]
            if text_id not in text_id2pred_res:
                text_id2pred_res[text_id] = {
                    "rel_list": sample["relation_list"],
                    "ent_list": sample["entity_list"],
                }
            else:
                text_id2pred_res[text_id]["rel_list"].extend(sample["relation_list"])
                text_id2pred_res[text_id]["ent_list"].extend(sample["entity_list"])

        text_id2text = {sample["id"]: sample["text"] for sample in ori_data}
        merged_pred_sample_list = []
        for text_id, pred_res in text_id2pred_res.items():
            filtered_rel_list, filtered_ent_list = filter_duplicates(pred_res["rel_list"], pred_res["ent_list"])
            merged_pred_sample_list.append({
                "id": text_id,
                "text": text_id2text[text_id],
                "relation_list": filtered_rel_list,
                "entity_list": filtered_ent_list,
            })

        return merged_pred_sample_list, df_al_

    def predict_single_data(self, text):
        pass

    def get_test_prf(self, pred_sample_list, gold_test_data, pattern="whole_text", if_only_lis=False):
        lis_item_rel = ["检验项目-结果", "检验项目-单位", "检验项目-敏感度", "检验项目-参考范围", "检验项目-代号"]
        lis_item_ent = ["检验项目", "结果", "单位", "敏感度", "参考范围", "代号"]

        text_id2gold_n_pred = {}  # text id to gold and pred results

        if if_only_lis:
            for sample in gold_test_data:
                text_id = sample["id"]
                text_id2gold_n_pred[text_id] = {
                    "gold_relation_list": sample["relation_list"],
                    "gold_entity_list": sample["entity_list"],
                    # "gold_event_list": sample["event_list"] if 'event_list' in sample else None,
                }
        else:
            for sample in gold_test_data:
                text_id = sample["id"]
                text_id2gold_n_pred[text_id] = {
                    "gold_relation_list": sample["relation_list"],
                    "gold_entity_list": sample["entity_list"],
                    # "gold_event_list": sample["event_list"] if 'event_list' in sample else None,
                }

            for sample in pred_sample_list:
                text_id = sample["id"]
                text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]
                text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

        correct_num, pred_num, gold_num = 0, 0, 0
        ent_correct_num, ent_pred_num, ent_gold_num = 0, 0, 0
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }
        rel_cpg = {rel: [0, 0, 0] for rel in self.rel2id}
        ent_cpg = {ent: [0, 0, 0] for ent in self.ent2id}
        for gold_n_pred in text_id2gold_n_pred.values():
            gold_rel_list = gold_n_pred["gold_relation_list"]
            pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
            gold_ent_list = gold_n_pred["gold_entity_list"]
            pred_ent_list = gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []

            if pattern == "event_extraction":
                pred_event_list = self.handshaking_tagger.trans2ee(pred_rel_list, pred_ent_list)  # transform to event list
                gold_event_list = gold_n_pred["gold_event_list"]  # *
                self.metrics.cal_event_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
            else:
                self.metrics.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern)
                get_ent_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ent_cpg, rel_cpg)

        if pattern == "event_extraction":
            trigger_iden_prf = self.metrics.get_prf_scores(ee_cpg_dict["trigger_iden_cpg"][0],
                                                      ee_cpg_dict["trigger_iden_cpg"][1],
                                                      ee_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = self.metrics.get_prf_scores(ee_cpg_dict["trigger_class_cpg"][0],
                                                       ee_cpg_dict["trigger_class_cpg"][1],
                                                       ee_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = self.metrics.get_prf_scores(ee_cpg_dict["arg_iden_cpg"][0], ee_cpg_dict["arg_iden_cpg"][1],
                                                  ee_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = self.metrics.get_prf_scores(ee_cpg_dict["arg_class_cpg"][0], ee_cpg_dict["arg_class_cpg"][1],
                                                   ee_cpg_dict["arg_class_cpg"][2])
            prf_dict = {
                "trigger_iden_prf": trigger_iden_prf,
                "trigger_class_prf": trigger_class_prf,
                "arg_iden_prf": arg_iden_prf,
                "arg_class_prf": arg_class_prf,
            }
            return prf_dict
        else:
            rel_prf = self.metrics.get_prf_scores(ere_cpg_dict["rel_cpg"][0], ere_cpg_dict["rel_cpg"][1],
                                             ere_cpg_dict["rel_cpg"][2])
            ent_prf = self.metrics.get_prf_scores(ere_cpg_dict["ent_cpg"][0], ere_cpg_dict["ent_cpg"][1],
                                             ere_cpg_dict["ent_cpg"][2])
            prf_dict = {
                "rel_prf": rel_prf,
                "ent_prf": ent_prf,
            }
            df_columns = ["correct", "predict", "gold", "precision", "recall", "f1"]
            df_indices = ["rel", "ent"]
            df_data = [(*ere_cpg_dict["rel_cpg"], *rel_prf), (*ere_cpg_dict["ent_cpg"], *ent_prf)]
            for k, v in rel_cpg.items():
                prf_dict["rel_" + k + "_prf"] = self.metrics.get_prf_scores(*v)
                df_indices.append("rel_" + k)
                df_data.append((*v, *prf_dict["rel_" + k + "_prf"]))
            for k, v in ent_cpg.items():
                prf_dict["ent_" + k + "_prf"] = self.metrics.get_prf_scores(*v)
                df_indices.append("ent_" + k)
                df_data.append((*v, *prf_dict["ent_" + k + "_prf"]))

            df_ = pd.DataFrame(df_data, df_indices, df_columns)
            return prf_dict, df_

    def get_test_prf_zhoujian(self, pred_sample_list, gold_test_data, pattern="whole_text"):
        text_id2gold_n_pred = {}  # text id to gold and pred results

        for sample in gold_test_data:
            text_id = sample["id"]
            relation_list, entity_list = filter_duplicates(sample.get("relation_list", []),
                                                           sample.get("entity_list", []))
            text_id2gold_n_pred[text_id] = {
                "gold_relation_list": relation_list,
                "gold_entity_list": entity_list,
                # "gold_event_list": sample["event_list"],
            }

        for sample in pred_sample_list:
            text_id = sample["id"]
            text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]
            text_id2gold_n_pred[text_id]["pred_entity_list"] = sample["entity_list"]

        correct_num, pred_num, gold_num = 0, 0, 0
        ent_correct_num, ent_pred_num, ent_gold_num = 0, 0, 0
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }
        rel_cpg = {rel: [0, 0, 0] for rel in rel2id}
        ent_cpg = {ent: [0, 0, 0] for ent in ent2id}
        for gold_n_pred in text_id2gold_n_pred.values():
            gold_rel_list = gold_n_pred["gold_relation_list"]
            pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
            gold_ent_list = gold_n_pred["gold_entity_list"]
            pred_ent_list = gold_n_pred["pred_entity_list"] if "pred_entity_list" in gold_n_pred else []

            if pattern == "event_extraction":
                pred_event_list = self.handshaking_tagger.trans2ee(pred_rel_list, pred_ent_list)  # transform to event list
                gold_event_list = gold_n_pred["gold_event_list"]  # *
                self.metrics.cal_event_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
            else:
                self.metrics.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern)
                get_ent_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ent_cpg, rel_cpg)

        if pattern == "event_extraction":
            trigger_iden_prf = self.metrics.get_prf_scores(ee_cpg_dict["trigger_iden_cpg"][0],
                                                      ee_cpg_dict["trigger_iden_cpg"][1],
                                                      ee_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = self.metrics.get_prf_scores(ee_cpg_dict["trigger_class_cpg"][0],
                                                       ee_cpg_dict["trigger_class_cpg"][1],
                                                       ee_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = self.metrics.get_prf_scores(ee_cpg_dict["arg_iden_cpg"][0], ee_cpg_dict["arg_iden_cpg"][1],
                                                  ee_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = self.metrics.get_prf_scores(ee_cpg_dict["arg_class_cpg"][0], ee_cpg_dict["arg_class_cpg"][1],
                                                   ee_cpg_dict["arg_class_cpg"][2])
            prf_dict = {
                "trigger_iden_prf": trigger_iden_prf,
                "trigger_class_prf": trigger_class_prf,
                "arg_iden_prf": arg_iden_prf,
                "arg_class_prf": arg_class_prf,
            }
            return prf_dict
        else:
            rel_prf = self.metrics.get_prf_scores(ere_cpg_dict["rel_cpg"][0], ere_cpg_dict["rel_cpg"][1],
                                             ere_cpg_dict["rel_cpg"][2])
            ent_prf = self.metrics.get_prf_scores(ere_cpg_dict["ent_cpg"][0], ere_cpg_dict["ent_cpg"][1],
                                             ere_cpg_dict["ent_cpg"][2])
            prf_dict = {
                "rel_prf": rel_prf,
                "ent_prf": ent_prf,
            }
            df_columns = ["correct", "predict", "gold", "precision", "recall", "f1"]
            df_indices = ["rel", "ent"]
            df_data = [(*ere_cpg_dict["rel_cpg"], *rel_prf), (*ere_cpg_dict["ent_cpg"], *ent_prf)]
            for k, v in rel_cpg.items():
                prf_dict["rel_" + k + "_prf"] = self.metrics.get_prf_scores(*v)
                df_indices.append("rel_" + k)
                df_data.append((*v, *prf_dict["rel_" + k + "_prf"]))
            for k, v in ent_cpg.items():
                prf_dict["ent_" + k + "_prf"] = self.metrics.get_prf_scores(*v)
                df_indices.append("ent_" + k)
                df_data.append((*v, *prf_dict["ent_" + k + "_prf"]))

            df_ = pd.DataFrame(df_data, df_indices, df_columns)
            return prf_dict, df_





if __name__ == '__main__':

    # predict
    MyPredictor = Predictor(config)
    res_dict_model = {}
    res_dict_ocr = {}
    predict_statistics_model = {}
    predict_statistics_ocr = {}
    for file_name, _ in test_data_dict.items():
        ori_test_data = copy.deepcopy(test_data_dict[file_name])
        for run_id, model_path_list in run_id2model_state_paths.items():
            save_dir4run = os.path.join(save_res_dir, run_id)
            if config["save_res"] and not os.path.exists(save_dir4run):
                os.makedirs(save_dir4run)

            for model_state_path in model_path_list:
                res_num = re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
                save_path = os.path.join(save_dir4run, "{}_res_model_{}.json".format(file_name, res_num))

                if os.path.exists(save_path):
                    pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding="utf-8")]
                    df_al = pd.read_excel(save_path[:-5] + "_al.xlsx")
                    print("{} already exists, load it directly!".format(save_path))
                else:
                    # load model state
                    MyPredictor.rel_extractor.load_state_dict(torch.load(model_state_path))
                    MyPredictor.rel_extractor.eval()
                    print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split("/")[-1]))

                    # predict
                    pred_sample_list, df_al = MyPredictor.predict(ori_test_data)

                res_dict_model[save_path] = (pred_sample_list, df_al)
                predict_statistics_model[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
    print('model_predict_res:', predict_statistics_model)

    # for file_name, short_data in ocr_data_dict.items():
    #     ori_test_data = ori_test_data_dict[file_name]
    #     for run_id, model_path_list in run_id2model_state_paths.items():
    #         save_dir4run = os.path.join(save_res_dir, run_id)
    #         if config["save_res"] and not os.path.exists(save_dir4run):
    #             os.makedirs(save_dir4run)
    #
    #         for model_state_path in model_path_list:
    #             res_num = re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
    #             save_path = os.path.join(save_dir4run, "{}_res_ocr_{}.json".format(file_name, res_num))
    #
    #             if os.path.exists(save_path):
    #                 pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding="utf-8")]
    #                 print("{} already exists, load it directly!".format(save_path))
    #             else:
    #                 # to do
    #                 pred_sample_list = None
    #
    #             res_dict_ocr[save_path] = pred_sample_list
    #             predict_statistics_ocr[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
    # print('ocr_predict_res:', predict_statistics_ocr)

    # # %%
    #
    # check
    # for path, res in res_dict_model.items():
    #     for sample in tqdm(res, desc="check char span"):
    #         text = sample["text"]
    #         for rel in sample["relation_list"]:
    #             assert rel["subject"] == text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
    #             assert rel["object"] == text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]

    # # %%
    #
    # # save
    # if config["save_res"]:
    #     for path, res in res_dict.items():
    #         with open(path, "w", encoding="utf-8") as file_out:
    #             for sample in tqdm(res, desc="Output"):
    #                 if len(sample["relation_list"]) == 0:
    #                     continue
    #                 json_line = json.dumps(sample, ensure_ascii=False)
    #                 file_out.write("{}\n".format(json_line))
    #
    # # %%
    #
    # score
    if config["score"]:
        filepath2scores_model, filepath2scores_ocr = {}, {}
        for file_path, pred_samples in res_dict_model.items():
            file_name = re.search("(.*?)_res_model_\d+\.json", file_path.split("/")[-1]).group(1)
            gold_test_data = test_data_dict[file_name]
            prf_dict, df_score = MyPredictor.get_test_prf(pred_sample_list, gold_test_data,
                                              pattern=config["hyper_parameters"]["match_pattern"])
            filepath2scores_model[file_path] = prf_dict
            file_score_excel = file_path[:-5] + "_metric.xlsx"
            file_al_excel = file_path[:-5] + "_al.xlsx"
            save_dir = '/'.join(file_score_excel.split('/')[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df_score.to_excel(file_score_excel)
            df_al.to_excel(file_al_excel, index=False)
        print("---------------- Results -----------------------")
        print('model_predict_res:', filepath2scores_model)
        # print('ocr_predict_res:', filepath2scores_ocr)
