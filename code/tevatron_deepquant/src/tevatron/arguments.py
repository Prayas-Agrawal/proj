import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments
from tevatron.trainer import TevatronTrainer as Trainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    
    model:str = field(default="dense")

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)
    q_len: int = field(default=32)
    p_len: int = field(default=128)
    semantic_loss: str = field(default=None)
    recon_loss:str = field(default=None)
    indep_loss:str = field(default="ce")
    cls_pool_mode: str = field(default=None)
    num_concepts_q: int = field(default=0)
    num_concepts_p: int = field(default=0)
    pooler_version: str = field(default="normal")
    concept_token_zero_pos_emb: bool = field(default=False)
    concept_token_sp_att_mask: bool = field(default=False)
    no_indep_loss: bool = field(default=False)
    mask_concept_tokens: bool = field(default=False)
    alignment_matrix: bool = field(default=False)
    temp: float = field(default=1.0)
    alpha: float = field(default=1.0)
    lamb: float = field(default=0.5)
    zero_pos_embs:bool = field(default=False)
    top_sim : int = field(default=1)
    score_func: str = field(default="dot")
    lm_head: bool = field(default=False)
    lexical_loss:str = field(default=None)
    
    mask_punctuation:bool = field(default=False)
    deepquant_alpha:float = field(default=0.5)
    deepquant_interaction_mode: str = field(default="joint")
    deepquant_masknonum:bool = field(default=False)
    deepquant_weightnode:bool = field(default=False)
    deepquant_neighbours:str = field(default=None)
    deepquant_graph_nonum_mask:bool = field(default=False)
    deepquant_weight_unsq:bool = field(default=False)
    deepquant_aggr_mode:str = field(default="mean")
    deepquant_quantscoring_collate_mode: str = field(default=None)
    deepquant_quantscoring_ver: str = field(default="v1")
    deepquant_quantscoring_combine_scores_mode: str = field(default=None)
    equal_score: str = field(default="gauss")
    asymdot_loss:str = field(default=None)
    latent_attribs: int = field(default=10)
    expand_nummask_window: float = field(default=0)
    
    num_encoder: str = field(default=None)
    deepquant_alpha_mode: str = field(default="constant")
    qcolbert_scoring: bool = field(default=False)
    numgpt_embed: bool = field(default=False)
    embed_exp_frac: float = field(default=None)
    
    add_relation_classifier_loss:bool = field(default=False) 
    add_num_pred_loss: bool = field(default=False)
    num_pred_loss_layerwise: bool = field(default=False)
    add_num_relation_pred_loss: bool = field(default=False)
    num_pred_loss_alpha: float = field(default=1)
    num_reps_from_hidden: bool = field(default=False)
    
    colbert_normalize: str = field(default=None)
    colbert_rep_normalize: bool = field(default=False)
    
    score_combine: str = field(default="linear")
    
    colbert_temp: float = field(default=1)
    num_temp: float = field(default=1)
    quant_pairwiseprob_temp: float = field(default=1)
    quant_combinescore_temp: float = field(default=1)
    quant_relationpred_temp: float = field(default=1)
    quant_numscoring_temp: float = field(default=1)
    new_mode:bool = field(default=False)
    num_get_ctx: bool = field(default=False)
    num_get_unit_ctx: bool = field(default=False)
    DEBUG_use_rawnum: bool = field(default=False)
    DEBUG_no_relpred: bool = field(default=False)
    DEBUG_score_pairprob_max: bool = field(default=False)
    DEBUG_score_relpred_num: str = field(default=None)
    relation_pred_mode: str = field(default="bi")
    op_pred_mode: str = field(default="pred")
    number_score_mode: str = field(default=None)
    REMOVE_pairwise_prog_norm: str = field(default=None)
    REMOVE_pairwise_prog_score: str = field(default="dot")
    REMOVE_num_rel_sigmoid: bool = field(default=False)
    gated_lamb: float = field(default=1)
    alpha_sigmoid: float = field(default=0)
    DEBUG_pairprob: str = field(default="pred")
    score_nosoftmax: bool = field(default=False)
    DEBUG_use_rerank_scores: bool = field(default=False)
    print_debug: bool = field(default=False)
    DEBUG_gate_thresh: float = field(default=None)
    single_step: str = field(default=None)
    
    
    
    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_format: str = field(default=None)
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encoded_format: str = field(default=None)
    deepquant_process: bool = field(default=False)
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)
    inject_concept_tokens: bool = field(default=False)
    deepquant_q_maxnums: int = field(default=1)
    deepquant_p_maxnums: int = field(default=3)
    expand_number:bool = field(default=False)
    colbert_stanford_format:bool = field(default=False)
    process_CQE:bool = field(default=False)
    process_whitespace:bool = field(default=False)
    num_as_token: bool = field(default=False)
    all_units: str = field(default=None)
    score_map: str = field(default=None)
    inject_units:bool = field(default=False)
    DEBUG_process_v1: bool = field(default=False)
    async_log: str = field(default=None)
    
    
    

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                # change all train directory paths to absolute
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files
                    if f.endswith('jsonl') or f.endswith('json')
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None
        print("*"*30)
        print("trainpath", self.train_path)
        print("*"*30)
        


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    resume_from_checkpoint: bool = field(default=False)
    max_grad_norm=1.0
    seed= 42
    # bf16 = True
    torch_compile: bool = True
    gradient_checkpointing=True   
    optim: str = "adamw_torch_fused"
    dataloader_pin_memory:bool = True
    dataloader_persistent_workers:bool = True
    

TRAINER: Trainer = None
VOCAB_SIZE: int = 0