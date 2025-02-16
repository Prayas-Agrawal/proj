from cgitb import text
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from tevatron.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class RerankerTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.tok.prepare_for_model(
            query_encoding,
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len + self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        qry = group['query']
        group_positives = group['positives']
        group_negatives = group['negatives']
        encoded_pairs = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = random.sample(group_positives, 1)[0]
        encoded_pairs.append(self.create_one_example(qry, pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            negs = random.sample(group_negatives, negative_size)
        for neg_psg in negs:
            encoded_pairs.append(self.create_one_example(qry, neg_psg))

        return encoded_pairs


class RerankerInferenceDataset_new(Dataset):
    input_keys = ['query_id', 'query', 'text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_q_len=32, max_p_len=256):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.encode_data)
    
    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.max_q_len if is_query else self.max_p_len,
            padding=False,
            return_token_type_ids=True,
        )
        return item

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        query_id, query, text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_query = self.create_one_example(query, is_query=True)
        encoded_text = self.create_one_example(text, is_query=False)
        # encoded_pair = self.tok.prepare_for_model(
        #     query,
        #     text,
        #     max_length=self.max_q_len + self.max_p_len,
        #     truncation='only_first',
        #     padding=False,
        #     return_token_type_ids=True,
        # )
        return query_id, text_id, {"query": encoded_query, "passage": encoded_text}
    
class RerankerInferenceDataset(Dataset):
    input_keys = ['query_id', 'query', 'text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_q_len=32, max_p_len=256):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        query_id, query, text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_pair = self.tok.prepare_for_model(
            query,
            text,
            max_length=self.max_q_len + self.max_p_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=True,
        )
        return query_id, text_id, encoded_pair


@dataclass
class RerankerTrainCollator(DataCollatorWithPadding):

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        pp = sum(features, [])
        pair_collated = self.tokenizer.pad(
            pp,
            padding='max_length',
            max_length=self.max_q_len+self.max_p_len,
            return_tensors="pt",
        )
        return pair_collated


@dataclass
class RerankerInferenceCollator(DataCollatorWithPadding):
    max_q_len: int = 16
    max_p_len: int = 128
    def __call__(self, features):
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        query_features = [x[2]["query"] for x in features]
        text_features = [x[2]["passage"] for x in features]
        # collated_features = super().__call__(text_features)
        query_collated =  self.tokenizer.pad(
            query_features,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        text_collated =  self.tokenizer.pad(
            text_features,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        return query_ids, text_ids, {"query": query_collated, "passage": text_collated}



class RerankPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '
                 , num_concepts_q=0, num_concepts_p=0, inject_concept_tokens = False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.concept_string_q = ""
        self.concept_string_p = ""
        if(inject_concept_tokens):
            print("***Injecting Concept Tokens into data***")
            self.concept_string_q = "".join(["<concept>" for _ in range(num_concepts_q)]) if num_concepts_q else ""
            self.concept_string_p = "".join(["<concept>" for _ in range(num_concepts_p)]) if num_concepts_p else ""

    def __call__(self, example):
        example = example
        qry = self.concept_string_q + example['query']
        query = self.tokenizer.encode(qry,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        text = self.concept_string_p + example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        encoded_passages = self.tokenizer.encode(text,
                                            add_special_tokens=False,
                                            max_length=self.text_max_length,
                                            truncation=True)
        return {'query_id': example['query_id'], 'query': query, 'text_id': example['docid'], 'text': encoded_passages}

class HFRerankDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str,
                 num_concepts_q = 0, num_concepts_p = 0):
        data_files = data_args.encode_in_path
        self.num_concepts_q = num_concepts_q
        self.num_concepts_p = num_concepts_p
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        print("*"*30)
        print("data_files",data_files)
        print("*"*30)
        dataset_name = data_args.dataset_name
        if(data_args.dataset_format is not None):
            dataset_name = data_args.dataset_format
        self.dataset = datasets.load_dataset(dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = RerankPreProcessor
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, 
                                  self.p_max_len, self.separator,
                                  self.num_concepts_q, self.num_concepts_p, self.data_args.inject_concept_tokens),
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset
