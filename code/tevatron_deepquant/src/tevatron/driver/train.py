import logging
import os
import sys

import torch
import torch._dynamo.config
import torch.distributed
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import tevatron.arguments as tevargs
from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import TrainDataset, QPCollator, QPCollator_deepquant, TrainDataset_deepquant
from tevatron.modeling import DenseModel, ColbertModel, ConceptModel, ConceptModel_Token, ColbertModel_stanford
from tevatron.trainer import TevatronTrainer as Trainer, GCTrainer, TevatronTrainer_deepquant as Trainer_deepquant
from tevatron.datasets import HFTrainDataset
import os


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

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        output_hidden_states = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_proc=1
    )
    
    print("*"*30)
    print("Model config", config)
    print("*"*30)
    
    
    
    arch = None
    if model_args.model == "dense":
        arch = DenseModel
    elif model_args.model == "colbert":
        arch = ColbertModel
    elif model_args.model == "colbert_stanford":
        arch = ColbertModel_stanford
    elif model_args.model == "concept":
        arch = ConceptModel
    elif model_args.model == "concept_token":
        arch = ConceptModel_Token
    
    print("Running model arch", model_args.model)
    if(model_args.model == "concept_token"):
        if(model_args.num_concepts_q != 0 and model_args.num_concepts_p != 0):
            print("***Adding special tokens****", len(tokenizer))
            tokenizer.add_tokens(["<concept>"])
            model.add_tokens_model(tokenizer)
            print("***Added special tokens****", len(tokenizer))
    if(data_args.num_as_token):
        print("***Vocab size***", len(tokenizer))
        print("***Adding <num> token****", len(tokenizer))
        tokenizer.add_tokens(["<num>"])
        print("***Added <num> token****", len(tokenizer))
        
    
    
    # model =torch.compile(model)

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir, 
                                   num_concepts_q=model_args.num_concepts_q,
                                   num_concepts_p=model_args.num_concepts_p)
    
    
    # if training_args.local_rank > 0:
    #     print("Waiting for main process to perform the mapping")
    #     torch.distributed.barrier()
        
    if(data_args.deepquant_process):
        train_dataset = TrainDataset_deepquant(data_args, train_dataset.process(), 
                                 tokenizer, model_args=model_args)
    else:
        train_dataset = TrainDataset(data_args, train_dataset.process(), 
                                    tokenizer, model_args=model_args)
    # if training_args.local_rank == 0:
    #     print("Loading results from main process")
    #     torch.distributed.barrier()
    
    model = arch.build(
        model_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
        tokenizer=tokenizer,
        data_args=data_args,
        meta = {
            "is_train": True,
            "num_training_steps": len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        }
    )
        
    base_trainer = Trainer_deepquant if data_args.deepquant_process else Trainer
    trainer_cls = GCTrainer if training_args.grad_cache else base_trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=(QPCollator_deepquant if data_args.deepquant_process else QPCollator)(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    tevargs.TRAINER = trainer
    tevargs.VOCAB_SIZE = len(tokenizer)
    
    train_dataset.trainer = trainer
    
    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/nas/root/.huggingfaceCache'
    os.environ['HF_DATASETS_CACHE'] = '/mnt/nas/root/.huggingfaceCache/datasets'
    # torch.distributed.init_process_group(backend="nccl", init_method='env://')
    main()
