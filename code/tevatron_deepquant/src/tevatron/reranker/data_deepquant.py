from cgitb import text
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from ..datasets import deepquant_dataset as dq
import torch
# from CQE import CQE
from tevatron.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class UnitTensorizer:
    def __init__(self):
        self.units = {}  # Dictionary to store unit-to-number mapping
        self.next_available_num = 0  # Counter for the next available number

    def get(self, unit):

        if unit in self.units:
            return self.units[unit]
        
        # Assign the next available number to the new unit
        self.units[unit] = self.next_available_num
        self.next_available_num += 1
        return self.units[unit]
    
    
class RerankerInferenceDataset_new_deepquant(Dataset):
    input_keys = ['query_id', 'query', 'text_id', 'text', 'numbers_qry', 'numbers_psg']

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
        query_id, query, text_id, text, numbers_qry, numbers_psg = (self.encode_data[item][f] for f in self.input_keys)
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
        return query_id, text_id, {"query": encoded_query, "passage": encoded_text, 
                                   "numbers_qry": numbers_qry, "numbers_psg": numbers_psg}


@dataclass
class RerankerInferenceCollator_deepquant(DataCollatorWithPadding):
    max_q_len: int = 16
    max_p_len: int = 128
    
    def __call__(self, features):
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        query_features = [x[2]["query"] for x in features]
        text_features = [x[2]["passage"] for x in features]
        query_nums = [x[2]["numbers_qry"] for x in features]
        text_nums = [x[2]["numbers_psg"] for x in features]
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
        # print("coll", len(query_collated), len(query_collated["input_ids"]))
        # print("coll query", self.tokenizer.convert_ids_to_tokens(query_collated["input_ids"][0]))
        # print("coll2", len(query_nums), query_nums[0])
        q_nums_collated =  {key: torch.tensor([d[key] for d in query_nums]) for key in query_nums[0].keys()}
        d_nums_collated =  {key: torch.tensor([d[key] for d in text_nums]) for key in text_nums[0].keys()}
        keys = q_nums_collated.keys()
        for key in keys:
            query_collated[key] = q_nums_collated[key]
            text_collated[key] = d_nums_collated[key]
        # print("coll query", self.tokenizer.convert_ids_to_tokens(query_collated["input_ids"][0]),q_nums_collated["numbers"][0])
        # print("coll3", len(q_nums_collated), len(q_nums_collated["numbers"]))
        return query_ids, text_ids, {"query": query_collated, "passage": text_collated}


class RerankPreProcessor_deepquant:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '
                 , num_concepts_q=0, num_concepts_p=0, inject_concept_tokens = False, data_args: DataArguments = None):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.concept_string_q = ""
        self.concept_string_p = ""
        self.data_args = data_args
        # self.parser = CQE.CQE(overload=True)
        self.parser = None
        self.prefix = ["cls", "q/d"] if data_args.colbert_stanford_format else ["cls"]
        self.Q_marker_token, self.Q_marker_token_id = "[unused0]", tokenizer.convert_tokens_to_ids("[unused0]")
        self.P_marker_token, self.P_marker_token_id = "[unused1]", tokenizer.convert_tokens_to_ids("[unused1]")
        self.sep_token, self.sep_token_id = tokenizer.sep_token, tokenizer.sep_token_id
        self.start_offset_q = len(self.prefix)+num_concepts_q
        self.start_offset_p = len(self.prefix)+num_concepts_p
        self.data_args = data_args
        self.all_units = UnitTensorizer()
        if(inject_concept_tokens):
            print("***Injecting Concept Tokens into data***")
            self.concept_string_q = "".join(["<concept>" for _ in range(num_concepts_q)]) if num_concepts_q else ""
            self.concept_string_p = "".join(["<concept>" for _ in range(num_concepts_p)]) if num_concepts_p else ""
            
    def process(self,prefix2, meta, text, start_offset, max_length, max_masks, prefix, 
                suffix, query, id=None, whole_obj=None):
        proc_text, numbers = dq.process_text_deepquant(prefix2, meta, text, self.tokenizer, 
                                                       start_offset, max_length, max_masks, self.data_args, 
                                                       query=query, parser=self.parser, id=id, whole_obj=whole_obj, all_units=self.all_units)
        proc_text = prefix + " " + proc_text + " " + suffix if self.data_args.colbert_stanford_format else proc_text
        return proc_text, numbers

    def __call__(self, example):
        qry_prefix = self.concept_string_q
        qry_text = qry_prefix + example['query']
        # print("QUERY", qry_text, example["meta"])
        qid = example['query_id']
        
        qry, numbers_qry = self.process(qry_prefix, example["query_meta"], qry_text, self.start_offset_q, self.query_max_length, 
                                        self.data_args.deepquant_q_maxnums, self.Q_marker_token, 
                                        self.sep_token, query=qry_text, id=qid, whole_obj=example)
            
        # qry = self.concept_string_q + example['query']
        query = self.tokenizer.encode(qry,
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        
        text_prefix = ""
        text = example['text']
        docid = example['docid']
        if ('title' in example and len(example['title']) > 0):
            text_prefix = ""
            # text = self.concept_string_p + example['title'] + self.separator + example['text']
            text = self.concept_string_p + example['text']
        text, numbers_psg = self.process(text_prefix, example["doc_meta"], text, self.start_offset_p, self.text_max_length, 
                                        self.data_args.deepquant_p_maxnums, self.P_marker_token, 
                                        self.sep_token, query=qry_text, id=docid, whole_obj=example)
        
        encoded_passages = self.tokenizer.encode(text,
                                            add_special_tokens=False,
                                            max_length=self.text_max_length,
                                            truncation=True)
        return {'query_id': example['query_id'], 'query': query, 
                'text_id': example['docid'], 'text': encoded_passages,
                "numbers_qry": numbers_qry, "numbers_psg": numbers_psg}


class HFRerankDataset_deepquant:
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
        self.preprocessor = RerankPreProcessor_deepquant
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
                                  self.num_concepts_q, self.num_concepts_p, self.data_args.inject_concept_tokens, data_args=self.data_args),
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset

