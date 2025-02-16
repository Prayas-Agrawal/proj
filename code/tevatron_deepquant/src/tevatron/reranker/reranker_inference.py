import logging
import os
import pickle
import sys
from contextlib import nullcontext
from unittest import result
import pandas as pd

import numpy as np
from tqdm import tqdm
import json

import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from tevatron.reranker.data import HFRerankDataset, RerankerInferenceDataset_new, RerankerInferenceCollator
from tevatron.reranker.data_deepquant import HFRerankDataset_deepquant, RerankerInferenceDataset_new_deepquant, RerankerInferenceCollator_deepquant
from tevatron.reranker.modeling import RerankerModel
from tevatron.modeling import DenseModel, ColbertModel, ConceptModel, ConceptModel_Token, ColbertModel_stanford

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    
    arch = None
    if(model_args.model == "dense"):
        arch = DenseModel
    if(model_args.model == "colbert"):
        arch = ColbertModel
    elif model_args.model == "colbert_stanford":
        arch = ColbertModel_stanford
    if(model_args.model == "concept"):
        arch = ConceptModel
    if(model_args.model == "concept_token"):
        arch = ConceptModel_Token
        
    print("Running model", arch)
    
    if(model_args.model == "concept_token"):
        print("***Vocab size***", len(tokenizer))
        # new_tokens = set(["<concept>"]) - set(tokenizer.vocab.keys())
        # print("***Add status***", new_tokens)
        if(model_args.num_concepts_q != 0 and model_args.num_concepts_p != 0):
            print("***Adding special tokens****", len(tokenizer))
            tokenizer.add_tokens(["<concept>"])
            # model.add_tokens_model(tokenizer)
            print("***Added special tokens****", len(tokenizer))

    if(data_args.num_as_token):
        print("***Vocab size***", len(tokenizer))
        print("***Adding <num> token****", len(tokenizer))
        tokenizer.add_tokens(["<num>"])
        print("***Added <num> token****", len(tokenizer))
    
    
    if model_args.untie_encoder:
        model = arch.load(
            model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_args=model_args,
            tokenizer=tokenizer,
            data_args=data_args,
            meta = {
                "is_train": False,
            }
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            output_hidden_states=True,
            data_args=data_args,
        ) if not model_args.model in ["concept", "concept_token"] else None
        model = arch.load(
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            model_args=model_args,
            cache_dir=model_args.cache_dir,
            tokenizer=tokenizer,
            data_args=data_args,
            meta = {
                "is_train": False,
            }
        )
        
    
    
    if(data_args.deepquant_process):
        rerank_HF_method = HFRerankDataset_deepquant
        rerank_dataset_method = RerankerInferenceDataset_new_deepquant
        collator_method = RerankerInferenceCollator_deepquant
        
    else:
        rerank_HF_method = HFRerankDataset
        rerank_dataset_method = RerankerInferenceDataset_new
        collator_method = RerankerInferenceCollator
        
    rerank_dataset = rerank_HF_method(tokenizer=tokenizer, data_args=data_args, 
                        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
                        num_concepts_q=model_args.num_concepts_q, 
                        num_concepts_p=model_args.num_concepts_p)
        
    rerank_dataset = rerank_dataset_method(
        rerank_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
        tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator_method(
            tokenizer,
            # max_length=data_args.q_max_len+data_args.p_max_len,
            padding='max_length',
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    model = model.to(training_args.device)
    model.eval()
    all_results = {}
    query_map = {}
    doc_map = {}
    score_map = {}
    data_df = pd.read_json(data_args.encode_in_path[0], lines=True)
    
    for i, row in tqdm(data_df.iterrows()):
        docid = row["docid"]
        queryid = row["query_id"]
        doc_map[str(docid)] = {
            "docid": row["docid"],
            "title": row["title"],
            "text": row["text"],
            "doc_meta": row["doc_meta"]
        }
        query_map[str(queryid)] = {
            "query_id": queryid,
            "query": row["query"],
            "query_meta": row["query_meta"]
        }
        
    for (batch_query_ids, batch_text_ids, batch) in tqdm(rerank_loader):
        with torch.amp.autocast("cuda") if training_args.fp16 else nullcontext():
            with torch.no_grad():
                query = batch["query"]
                # query_nums = batch["numbers_qry"]
                passage = batch["passage"]
                # passage_nums = batch["numbers_psg"]
                for k, v in query.items():
                    query[k] = v.to(training_args.device)
                for k, v in passage.items():
                    passage[k] = v.to(training_args.device)
                # for k, v in query_nums.items():
                #     # print("query nums", k, v)
                #     query_nums[k] = v.to(training_args.device)
                # for k, v in passage_nums.items():
                #     passage_nums[k] = v.to(training_args.device)
                
                model_output = model(query, passage)
                # print("model out", model_output)
                scores = model_output.scores.cpu().detach().numpy()
                # print("scores", scores.shape)
                for i in range(len(scores)):
                    qid = batch_query_ids[i]
                    docid = batch_text_ids[i]
                    
                    # score = scores[i][0]
                    score = scores[i][i]
                    if qid not in all_results:
                        all_results[qid] = []
                    # if(data_args.encoded_overload):
                    #     all_results[qid].append((qid, batch["query_text"], docid, batch["doc_text"], score))
                    # else:
                    all_results[qid].append((docid, score))
                    if(str(qid) not in score_map.keys()): score_map[str(qid)] = {}
                    score_map[str(qid)][str(docid)] = score
                    
    RERANK_LIMIT = 1000
    def normalise_score_map(score_map):
        normalized_data = {}
        K = RERANK_LIMIT
        for query, docs in score_map.items():
            top_k_docs = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:K])
            
            max_score = max(top_k_docs.values()) if top_k_docs else 1
            
            normalized_data[query] = {doc: score / max_score for doc, score in top_k_docs.items()}
        return normalized_data
   
    def pd_write(all_results, query_map, doc_map, path):
        rows = []
        # f.write(f'query_id\tquery\tdocid\tdoc\tscore\n')
        for qid in sorted(all_results.keys()):
            doc_score_list = sorted(all_results[qid], key=lambda x: x[-1], reverse=True)
            query_text = query_map[str(qid)]["query"]
            for docid, score in doc_score_list:
                doc_text = doc_map[str(docid)]["text"]
                rows.append([qid, query_text, docid, doc_text, score])
        df = pd.DataFrame(rows, columns=["query_id", "query", "docid", "doc", "score"])
        df.to_csv(
            path,
            sep="\t",
            index=False,
            header=True,
            quoting=csv.QUOTE_ALL
        )
        
    def rerank_write(all_results, query_map, doc_map, path):
        rows = []
        # f.write(f'query_id\tquery\tdocid\tdoc\tscore\n')
        for qid in sorted(all_results.keys()):
            doc_score_list = sorted(all_results[qid], key=lambda x: x[-1], reverse=True)
            query_text = query_map[str(qid)]["query"]
            counter = 1
            for docid, score in doc_score_list:
                if(counter == RERANK_LIMIT): break
                norm_score = score_map[str(qid)][str(docid)]
                doc_text = doc_map[str(docid)]["text"]
                obj = {
                    "query_id": qid, 
                    "docid": str(docid), 
                    "score": norm_score, 
                    "query": query_text, 
                    "title": "", 
                    "text": doc_text, 
                    "query_meta": query_map[str(qid)]["query_meta"], 
                    "doc_meta": doc_map[str(docid)]["doc_meta"]
                }
                rows.append(obj)
                counter += 1
        with open(os.path.join(path), "w") as f:
            for content in tqdm(rows):
                json_line = json.dumps(content, default=str)
                f.write(json_line + '\n')
    
    score_map = normalise_score_map(score_map)
    if(data_args.encoded_format == "overload"):
        pd_write(all_results, query_map, doc_map, data_args.encoded_save_path)
    elif(data_args.encoded_format == "rerank"):
        rerank_write(all_results, query_map, doc_map, data_args.encoded_save_path)
    else:
        with open(data_args.encoded_save_path, 'w') as f:
            for qid in all_results:
                results = sorted(all_results[qid], key=lambda x: x[-1], reverse=True)
                for docid, score in results:
                    f.write(f'{qid}\t{docid}\t{score}\n')
                    
    

        

if __name__ == "__main__":
    main()
