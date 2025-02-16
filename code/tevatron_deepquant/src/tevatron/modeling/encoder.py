import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput
from transformers import AutoConfig, AutoTokenizer
import tevatron.modeling.data_helpers as data_helpers
from .util import QuantScoring, NumGPTEmbed
import tevatron.arguments as tevargs
from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from .deepquant_grad.model import Quant_Grad
import logging
import pickle
from .data_helpers import *
from .helpers.helpers import *
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError('EncoderPooler is an abstract class')

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)

class EncoderModel_stanford(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 model_args:ModelArguments = None,
                 tokenizer=None,
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.linear = nn.Linear(model_args.projection_in_dim, model_args.projection_out_dim)
        self.pooler = pooler
        self.model_args = model_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer: AutoTokenizer = None,
            **hf_kwargs,
    ):
        
        return model_build(cls, model_args, train_args, tokenizer, **hf_kwargs)
    # @classmethod
    def build_old(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args:ModelArguments,
            tokenizer:AutoTokenizer=None,
            **hf_kwargs,
    ):
        return model_load(cls, model_name_or_path, model_args, tokenizer, **hf_kwargs)
    
    # @classmethod
    def load_old(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            config_qry = AutoConfig.from_pretrained(
                _qry_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            config_psg = AutoConfig.from_pretrained(
                _psg_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    config=config_qry,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    config=config_psg,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'), safe_serialization=False)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'), safe_serialization=False)
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 model_args:ModelArguments = None,
                 tokenizer=None,
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.model_args = model_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        self.tokenizer = tokenizer
        self.skiplist = data_helpers.get_skiplist(tokenizer, self.model_args.mask_punctuation)
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        # alt_q = self.encode_passage(query)
        # alt_p = self.encode_query(passage)
        # assert torch.equal(alt_q, q_reps)
        # assert torch.equal(alt_p, p_reps)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer: AutoTokenizer = None,
            **hf_kwargs,
    ):
        
        return model_build(cls, model_args, train_args, tokenizer, **hf_kwargs)
    # @classmethod
    def build_old(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args:ModelArguments,
            tokenizer:AutoTokenizer=None,
            **hf_kwargs,
    ):
        return model_load(cls, model_name_or_path, model_args, tokenizer, **hf_kwargs)
    
    # @classmethod
    def load_old(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            config_qry = AutoConfig.from_pretrained(
                _qry_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            config_psg = AutoConfig.from_pretrained(
                _psg_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    config=config_qry,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    config=config_psg,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'), safe_serialization=False)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'), safe_serialization=False)
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

# def pickle_load(path):
#     f = None
#     if(path is None): return None
#     with open(path, "rb") as fp:
#         f = pickle.load(fp)
#     return f


class EncoderModel_deepquant(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 model_args:ModelArguments = None,
                 tokenizer=None,
                 data_args:DataArguments =None,
                 meta= None,
                 ):
        super().__init__()
        # torch.autograd.set_detect_anomaly(True)

        print("Model Args", model_args)
        self.lm_q = lm_q
        self.lm_p = lm_p
        # self.units = pickle_load(data_args.all_units)
        # self.score_map = pickle_load(data_args.score_map)
        # self.units_rev = {self.units[k]:k for k in self.units.keys()}
        self.pooler = pooler
        self.model_args = model_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        # self.gcn = GCN_deepquant(self.model_args.projection_out_dim, model_args=self.model_args)
        self.quantScorer = QuantScoring(self.model_args.projection_out_dim, model_args=self.model_args)
        self.untie_encoder = untie_encoder
        # self.alpha_project = torch.nn.Linear(self.model_args.projection_out_dim, 1)
        self.temp_scheduler = SoftmaxTemperatureScheduler(initial_temp=1, 
                                                          final_temp=0.02, 
                                                          total_steps=meta.get("num_training_steps"), 
                                                          strategy="exponential",
                                                          inference=not meta["is_train"])
        self.tokenizer = tokenizer
        self.model_meta = meta
        self.saver = None
        if(data_args.async_log is not None):
            self.saver = AsyncDataSaver(output_file=f"{data_args.async_log}")
        self.deepquant_alpha_network = PairwiseScores_Cls(model_args.projection_out_dim, out_dim=1, num_layers=1)
        # self.num_encoder = Quant_Grad.from_pretrained(model_args.num_encoder)
        # self.freeze_model(self.num_encoder)
        # self.test = torch.randint(low=0, high=int(1e5), size=(2,)).to("cuda").float()
        # self.test = torch.tensor([200,1e5]).to("cuda").float()
        self.skiplist = data_helpers.get_skiplist(tokenizer, self.model_args.mask_punctuation)
        # torch.autograd.set_detect_anomaly(True)

        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
        if self.model_args.numgpt_embed:
            self.num_embedding = NumGPTEmbed(frac_exp=model_args.embed_exp_frac, output_dim=model_args.projection_in_dim)
            self.lm_q.resize_token_embeddings(len(tokenizer))
            self.lm_p.resize_token_embeddings(len(tokenizer))
        torch.set_printoptions(sci_mode=False)


            
    def freeze_model(self, model, freeze=True):
        for param in model.parameters():
            param.requires_grad = True
            
    

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        
        q_reps, q_hidden, q_meta = self.encode_query(query)
        p_reps, p_hidden, p_meta = self.encode_passage(passage)
        query_nums = {"numbers": query["numbers"], "number_mask":query["number_mask"], "unit_mask":query["unit_mask"], 
                      "op": query["op"], "units": query["units"], "id": query["id"]}
        passage_nums = {"numbers": passage["numbers"], "number_mask":passage["number_mask"], "unit_mask":passage["unit_mask"], 
                        "units": passage["units"], "id": passage["id"]}
        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            calc_loss = self.lm_p.training
            scores, extra_loss, meta = self.compute_similarity(q_reps, p_reps, query_nums,
                                                         passage_nums, q_hidden, p_hidden, 
                                                         calc_loss, q_meta, p_meta, query, passage)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(q_reps.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            print("CE LOSS:", loss, flush=True)
            loss += extra_loss
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores, extra_loss, meta =  self.compute_similarity(q_reps, p_reps, query_nums, 
                                                          passage_nums, q_hidden, p_hidden, None,
                                                          q_meta, p_meta, query, passage)
            # for i in range(query['input_ids'].shape[0]):
            #     qids = self.tokenizer.convert_ids_to_tokens(query["input_ids"][i])
            #     vals, idxs = torch.topk(torch.softmax(meta["qexp_logits"], dim=-1), 5, dim=-1)
            #     print("query_enc", qids, "\n",
            #         "len", len(qids), len(query["number_mask"][i][0]), "\n",
            #         "query_nums", query["numbers"][i], "\n",
            #         "meta shape", meta["qexp_logits"].shape, "\n",
            #         "meta", vals, idxs-10, "\n",
            #         "####################", flush=True  
            #     )
            # print("num", self.test, self.num_encoder(self.test).logits)
            loss = None
        
        self.temp_scheduler.step()
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    # def compute_similarity(self, q_reps, p_reps):
    #     return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer: AutoTokenizer = None,
            data_args: DataArguments = None,
            **hf_kwargs,
    ):
        
        return model_build(cls, model_args, train_args, tokenizer, data_args=data_args,**hf_kwargs)

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args:ModelArguments,
            tokenizer:AutoTokenizer=None,
            data_args: DataArguments = None,
            **hf_kwargs,
    ):
        return model_load(cls, model_name_or_path, model_args, tokenizer, data_args=data_args, **hf_kwargs)
    
    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'), safe_serialization=False)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'), safe_serialization=False)
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)


def checkNan(tensor, key):
    if(torch.isnan(tensor).any()):
        print("***********found nan ", key)
    return 1

class EncoderModel_Concept(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 model_args: ModelArguments,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 ):
        super().__init__()
        # print("Reconstruction loss mode", model_args.recon_loss)

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        raise NotImplementedError('EncoderModel is an abstract class')

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def compute_reconstruction_loss(self, hidden, reconstructed):
        batch = hidden.size(0)
        return self.mse(reconstructed.view(batch, -1), hidden.view(batch, -1))

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer: AutoTokenizer = None,
            **hf_kwargs,
    ):
        
        return model_build(cls, model_args, train_args, tokenizer, **hf_kwargs)

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args:ModelArguments,
            tokenizer:AutoTokenizer=None,
            **hf_kwargs,
    ):
        return model_load(cls, model_name_or_path, model_args, tokenizer, **hf_kwargs)
    
    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'), safe_serialization=False)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'), safe_serialization=False)
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

def zero_pos_embs(model:PreTrainedModel, start:int, num:int):
    pos = copy.deepcopy(model.embeddings.position_embeddings.weight.data)
    pos[start:start+num] = 0
    return pos

def set_pos_embs(model:PreTrainedModel, pos):
    model.embeddings.position_embeddings.weight.data = pos


class EncoderModel_Concept_token(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 model_args: ModelArguments = None,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 ):
        super().__init__()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        raise
        
    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg, cls_mean_pool=False):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry, cls_mean_pool=False):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def compute_reconstruction_loss(self, hidden, reconstructed):
        batch = hidden.size(0)
        return self.mse(reconstructed.view(batch, -1), hidden.view(batch, -1))

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer: AutoTokenizer = None,
            **hf_kwargs,
    ):
        
        return model_build(cls, model_args, train_args, tokenizer, **hf_kwargs)

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args:ModelArguments,
            tokenizer:AutoTokenizer=None,
            **hf_kwargs,
    ):
        return model_load(cls, model_name_or_path, model_args, tokenizer, **hf_kwargs)

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'), safe_serialization=False)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'), safe_serialization=False)
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False)
        if self.pooler:
            self.pooler.save_pooler(output_dir)


def model_build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        tokenizer: AutoTokenizer = None,
        data_args: DataArguments = None,
        meta = None,
        **hf_kwargs,
):
    # load local
    # print("######################building")
    TRANSFORMER_CLS = AutoModelForMaskedLM if model_args.lm_head else AutoModel
    print("CLS", TRANSFORMER_CLS, model_args.lm_head, model_args.model_name_or_path)
    if os.path.isdir(model_args.model_name_or_path):
        if model_args.untie_encoder:
            _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
            if not os.path.exists(_qry_model_path):
                _qry_model_path = model_args.model_name_or_path
                _psg_model_path = model_args.model_name_or_path
            logger.info(f'loading query model weight from {_qry_model_path}')
            lm_q = TRANSFORMER_CLS.from_pretrained(
                _qry_model_path,
                **hf_kwargs
            )
            logger.info(f'loading passage model weight from {_psg_model_path}')
            lm_p = TRANSFORMER_CLS.from_pretrained(
                _psg_model_path,
                **hf_kwargs
            )
        else:
            lm_q = TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path,
  **hf_kwargs)
            lm_p = lm_q
    # load pre-trained
    else:
        lm_q = TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q
    # print("#########POOLER ARGS", model_args.add_pooler)
    if model_args.add_pooler:
        pooler = cls.build_pooler(model_args)
    else:
        pooler = None
    # if model_args.concept_tokens > 0 and tokenizer is not None:
    #     cls.add_tokens_model(lm_q, tokenizer )
    #     cls.add_tokens_model(lm_p, tokenizer )

    model = cls(
        lm_q=lm_q,
        lm_p=lm_p,
        pooler=pooler,
        negatives_x_device=train_args.negatives_x_device,
        untie_encoder=model_args.untie_encoder,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        meta=meta
    )
    return model

def model_load(
        cls,
        model_name_or_path,
        model_args:ModelArguments,
        tokenizer: AutoTokenizer = None,
        data_args: DataArguments = None,
        meta = None,
        **hf_kwargs,
):
    # load local
    TRANSFORMER_CLS = AutoModelForMaskedLM if model_args.lm_head else AutoModel
    untie_encoder = True
    if os.path.isdir(model_name_or_path):
        _qry_model_path = os.path.join(model_name_or_path, 'query_model')
        _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
        
        if os.path.exists(_qry_model_path):
            config_qry = AutoConfig.from_pretrained(
                _qry_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            config_psg = AutoConfig.from_pretrained(
                _psg_model_path,
                num_labels= 1,
                **hf_kwargs,
            )
            logger.info(f'found separate weight for query/passage encoders')
            logger.info(f'loading query model weight from {_qry_model_path}')
            lm_q = TRANSFORMER_CLS.from_pretrained(
                _qry_model_path,
                config=config_qry,
                **hf_kwargs
            )
            logger.info(f'loading passage model weight from {_psg_model_path}')
            lm_p = TRANSFORMER_CLS.from_pretrained(
                _psg_model_path,
                config=config_psg,
                **hf_kwargs
            )
            untie_encoder = False
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = TRANSFORMER_CLS.from_pretrained(model_name_or_path,  **hf_kwargs)
            lm_p = lm_q
    else:
        logger.info(f'try loading tied weight')
        logger.info(f'loading model weight from {model_name_or_path}')
        lm_q = TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        lm_p = lm_q

    pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
    pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
    if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
        logger.info(f'found pooler weight and configuration')
        with open(pooler_config) as f:
            pooler_config_dict = json.load(f)
        pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
    else:
        pooler = None
        
    # if pooler_config.concept_tokens and pooler_config.concept_tokens > 0 and tokenizer is not None:
    #     cls.add_tokens_model(lm_q, tokenizer )
    #     cls.add_tokens_model(lm_p, tokenizer )

    model = cls(
        lm_q=lm_q,
        lm_p=lm_p,
        pooler=pooler,
        untie_encoder=untie_encoder,
        model_args=model_args,
        tokenizer=tokenizer,
        data_args=data_args,
        meta=meta,
    )
    return model
