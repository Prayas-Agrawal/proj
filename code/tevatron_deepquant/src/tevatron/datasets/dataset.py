from datasets import load_dataset
from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, TrainPreProcessor_deepquant , QueryPreProcessor, CorpusPreProcessor
from ..arguments import DataArguments
from . import deepquant_dataset as dq
from tqdm import tqdm

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
DEFAULT_PROCESSORS_deepquant = [TrainPreProcessor_deepquant, QueryPreProcessor, CorpusPreProcessor]
# PROCESSOR_INFO = {
#     'Tevatron/wikipedia-nq': DEFAULT_PROCESSORS,
#     'Tevatron/wikipedia-trivia': DEFAULT_PROCESSORS,
#     'Tevatron/wikipedia-curated': DEFAULT_PROCESSORS,
#     'Tevatron/wikipedia-wq': DEFAULT_PROCESSORS,
#     'Tevatron/wikipedia-squad': DEFAULT_PROCESSORS,
#     'Tevatron/scifact': DEFAULT_PROCESSORS,
#     'Tevatron/msmarco-passage': DEFAULT_PROCESSORS,
#     'json': [None, None, None]
# }


class HFTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 data_args: DataArguments, cache_dir: str, num_concepts_q = 0,
                 num_concepts_p = 0
                 ):
        data_files = data_args.train_path
        self.num_concepts_q = num_concepts_q
        self.data_args = data_args
        self.num_concepts_p = num_concepts_p
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        print("*"*30)
        print("data_files",data_files)
        print("*"*30)
        dataset_name = data_args.dataset_name
        if(data_args.dataset_format is not None):
            dataset_name = data_args.dataset_format
        self.dataset = load_dataset(dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        default_proc = DEFAULT_PROCESSORS_deepquant if data_args.deepquant_process else DEFAULT_PROCESSORS
        # self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO\
            # else DEFAULT_PROCESSORS[0]
        self.preprocessor = default_proc[0]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    
    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, 
                                  self.p_max_len, 
                                  self.separator, self.num_concepts_q
                                  , self.num_concepts_p, self.data_args.inject_concept_tokens, data_args=self.data_args),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset

class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = PROCESSOR_INFO[script_prefix][2] \
            if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset
