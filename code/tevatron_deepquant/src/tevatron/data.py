import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
import torch

from .arguments import DataArguments, ModelArguments
import numpy as np
from .trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
            model_args: ModelArguments = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages


class TrainDataset_deepquant(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
            model_args: ModelArguments = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)
        
    def create_num(self, num):
        keys = [k for k in num.keys()]
        return {k: torch.tensor(num[k], dtype=torch.float64) for k in keys}
    

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        # print("prep", qry)
        nums_qry = group["numbers_qry"]
        encoded_query = self.create_one_example(qry, is_query=True)
        query_nums = nums_qry
        # encoded_query["numbers"] = self.create_num(nums_qry)
        # encoded_query.update(self.create_num(nums_qry))

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']
        passage_nums = []
        
        nums_pos = group["numbers_pos"]
        nums_neg_input = group["numbers_neg"]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            pos_nums = nums_pos[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
            pos_nums = nums_pos[(_hashed_seed + epoch) % len(group_positives)]
        pos_psg = self.create_one_example(pos_psg)
        passage_nums.append(pos_nums)
        # pos_psg["numbers"] = self.create_num(pos_nums)
        # pos_psg.update(self.create_num(pos_nums))
        encoded_passages.append(pos_psg)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            idxs = [i for i in range(len(group_negatives))]
            negs, negs_nums_toadd = [], []
            negs_idxs = random.choices(idxs, k=negative_size)
            for idx in negs_idxs:
                negs.append(group_negatives[idx])
                negs_nums_toadd.append(nums_neg_input[idx])
            # negs = list(np.array(group_negatives)[negs_idxs])
            # negs_nums_toadd = list(np.array(nums_neg_input)[negs_idxs])
        elif self.data_args.train_n_passages == 1:
            negs = []
            negs_nums_toadd = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            negs_nums_toadd = nums_neg_input[:negative_size]
        else:
            # raise "This condition is not yet implemented"
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            negs_nums_toadd = [x for x in nums_neg_input]
            idxs = [i for i in range(len(negs))]
            random.Random(_hashed_seed).shuffle(negs)
            random.Random(_hashed_seed).shuffle(negs_nums_toadd)
            negs = negs * 2
            negs_nums_toadd = negs_nums_toadd * 2
            
            negs = negs[_offset: _offset + negative_size]
            negs_nums_toadd = negs_nums_toadd[_offset: _offset + negative_size]
            # print("negs", self.tok.convert_ids_to_tokens(negs[0]), negs_nums_toadd[0])
            # print("negs", self.tok.convert_ids_to_tokens(negs[1]), negs_nums_toadd[1])
            # print("negs", self.tok.convert_ids_to_tokens(negs[2]), negs_nums_toadd[2])
            # raise "Sd"

        for i, neg_psg in enumerate(negs):
            neg_psg = self.create_one_example(neg_psg)
            # neg_psg["numbers"] = self.create_num(negs_nums_toadd[i])
            # neg_psg.update()
            neg_num = negs_nums_toadd[i]
            passage_nums.append(neg_num)
            encoded_passages.append(neg_psg)
        # print("keys", [k for k in encoded_query.keys()], [k for k in encoded_passages[0].keys()])
        # print("negs", self.tok.convert_ids_to_tokens(negs[0]), negs_nums[0])
        # print("negs1", self.tok.convert_ids_to_tokens(encoded_passages[0]["input_ids"]),encoded_passages[0]["numbers"]["numbers"], passage_nums[0]["numbers"], len(passage_nums[0]))
        # print("negs1", self.tok.convert_ids_to_tokens(encoded_passages[1]["input_ids"]),encoded_passages[1]["numbers"]["numbers"], passage_nums[1]["numbers"], len(passage_nums[1]))
        # print("negs1", self.tok.convert_ids_to_tokens(encoded_passages[2]["input_ids"]),encoded_passages[2]["numbers"]["numbers"], passage_nums[2]["numbers"], len(passage_nums[2]))
        # print("negs1", self.tok.convert_ids_to_tokens(encoded_passages[3]["input_ids"]),encoded_passages[3]["numbers"]["numbers"], passage_nums[3]["numbers"], len(passage_nums[3]))
        # raise "Sd"
        return encoded_query, encoded_passages, query_nums, passage_nums


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class QPCollator_deepquant(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        qq_nums = [f[2] for f in features]
        dd = [f[1] for f in features]
        dd_nums = [f[3] for f in features]
        
        if isinstance(qq[0], list):
            qq = sum(qq, [])
            qq_nums = sum(qq_nums, [])
            
        if isinstance(dd[0], list):
            dd = sum(dd, [])
            dd_nums = sum(dd_nums, [])
            
        # print("Affter", dd)
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        # print("features", dd_nums)
        
        # print("COLLATED", len(dd), len(dd[0]), len(d_collated))
        # print("COLLATED", len(d_collated["input_ids"]))
        # print("qq",qq_nums)
        # print("KEYS", [key for key in dd_nums[0].keys()])
        # print("SIZES", str({key: [len(d[key]) for d in dd_nums] for key in dd_nums[0].keys()}), flush=True)
        q_nums_collated =  {key: torch.tensor([d[key] for d in qq_nums]) for key in qq_nums[0].keys()}
        d_nums_collated =  {key: torch.tensor([d[key] for d in dd_nums]) for key in dd_nums[0].keys()}
        keys = [k for k in qq_nums[0].keys()]
        for key in keys:
            q_collated[key] = q_nums_collated[key]
            d_collated[key] = d_nums_collated[key]
        # print("col", q_collated)
        # print("COLLABTED", len(q_collated), len(d_collated))
        # print("COLLABTED", len(q_collated), len(d_collated[0]))
        

        return q_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features