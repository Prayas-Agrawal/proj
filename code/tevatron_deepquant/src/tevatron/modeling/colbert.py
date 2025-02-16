import torch
import torch.nn as nn
from torch import Tensor
import logging
import torch.nn.functional as F
from .encoder import EncoderPooler, EncoderModel, EncoderModel_deepquant
from . import data_helpers
from contextlib import nullcontext

logger = logging.getLogger(__name__)


class ColbertPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, tied=True):
        super(ColbertPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class ColbertModel(EncoderModel_deepquant):
    
    def align_number_embeds(self, numeric_embeds, num_masks, seq_len, dim):
        expanded_embeddings = numeric_embeds.unsqueeze(2).expand(-1, -1, seq_len, -1)
        expanded_masks = num_masks.unsqueeze(-1).expand(-1, -1, -1, dim)
        masked_embeddings = expanded_embeddings * expanded_masks  # (batch, num_slots, seq_len, dim)
        
        numeric_embeds = masked_embeddings.sum(dim=1) 
        return numeric_embeds
    
    
    def encode_with_nums(self, bert, qry):
        
        input_embeddings = bert.embeddings.word_embeddings(qry["input_ids"])  # [batch_size, seq_len, hidden_size]
        _, seq_len, dim = input_embeddings.shape
        
        numbers = qry["numbers"] # b, 6
        num_masks = qry["number_mask"] # b, 6, seq
        # print("numbers HERE", numbers)
        numeric_embeds = self.num_embedding(numbers)  # [batch_size, 6, hidden_size]
        numeric_embeds = self.align_number_embeds(numeric_embeds, num_masks, seq_len, dim)
        num_add = numeric_embeds
        enhanced_embeddings = input_embeddings + numeric_embeds  # [batch_size, seq_len, hidden_size]
        # enhanced_embeddings = input_embeddings

        embedding_output = bert.embeddings(
            inputs_embeds=enhanced_embeddings,
            position_ids=None,
            past_key_values_length=0,
        )
        outputs = bert.encoder(
            embedding_output,
            attention_mask=qry["attention_mask"].float()[:, None, None, :],
            head_mask=[None] * 12,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=True,
        )

        return outputs, num_add


    def encode_passage(self, psg):
        if psg is None:
            return None
        meta = None
        if(self.model_args.numgpt_embed):
            psg_out, num_add = self.encode_with_nums(self.lm_p, psg)
            meta = {
                "num_add":num_add
            }
        else:
            psg_out = self.lm_p(psg["input_ids"], psg["attention_mask"], return_dict=True)
            
        p_hidden = psg_out.last_hidden_state
        p_reps = p_hidden
        if(self.model_args.add_pooler):
            p_reps = self.pooler(p=p_hidden)
        p_reps *= psg['attention_mask'][:, :, None].float()
        p_hidden *= psg['attention_mask'][:, :, None].float()
        
        '''Punctuation Masking'''
        mask = data_helpers.mask(psg["input_ids"], self.skiplist, self.tokenizer.pad_token_id)
        if mask is not None:
            p_reps = p_reps*mask
            p_hidden = p_hidden*mask
            
        return p_reps, p_hidden, meta
    
    def encode_query(self, qry):
        if qry is None:
            return None
        meta = None
        if(self.model_args.numgpt_embed):
            qry_out, num_add = self.encode_with_nums(self.lm_q, qry)
            meta = {
                "num_add":num_add
            }
        else: 
            qry_out = self.lm_q(qry["input_ids"], qry["attention_mask"], return_dict=True)
        
        q_hidden = qry_out.last_hidden_state
        q_reps = q_hidden
        if(self.model_args.add_pooler):
            q_reps = self.pooler(q=q_hidden)
        q_reps *= qry['attention_mask'][:, :, None].float()
        q_hidden *= qry['attention_mask'][:, :, None].float()
        
        return q_reps, q_hidden, meta
    
    def colbert(self, q_reps, p_reps):
        q_reps, p_reps = q_reps[:, 1:], p_reps[:, 1:]
        def _dot_prod(q_reps, p_reps):
            q_reps = F.normalize(q_reps, dim=-1)
            p_reps = F.normalize(p_reps, dim=-1)
            return q_reps, p_reps
        if(self.model_args.colbert_rep_normalize):
            q_reps, p_reps = _dot_prod(q_reps, p_reps)
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        return scores
    
    def get_cold(self, q_reps, p_reps, q_mask, q_nums=None, p_nums=None):
        if(self.model_args.DEBUG_use_rerank_scores):
            # print("score", q_nums["score"].shape, flush=True)
            return torch.diag(q_nums["score"])
        # q_size, p_size = q_reps.size(1), p_reps.size(1)
        # q_reps, p_reps = q_reps[:, 1:], p_reps[:, 1:]
        # def _dot_prod(q_reps, p_reps):
        #     q_reps = F.normalize(q_reps, dim=-1)
        #     p_reps = F.normalize(p_reps, dim=-1)
        #     return q_reps, p_reps
        # if(self.model_args.colbert_rep_normalize):
        #     q_reps, p_reps = _dot_prod(q_reps, p_reps)
        # token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        # scores, _ = token_scores.max(-1)
        scores = self.colbert(q_reps, p_reps)
        
        # denom,_ = scores.max(1, keepdim=True)
        denom =  q_mask[:, 1:].sum(-1, keepdim=True) # FIXME: p mask ?
        scores = scores.sum(1)/denom
        return scores
        
    
    def get_scores(self, q_reps, p_reps, q_mask, num_scores=None, query=None, doc=None,
                      q_nums=None, p_nums=None, colb_no_grad=False):
        # q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=2)
        # p_reps = torch.nn.functional.normalize(p_reps, p=2, dim=2)
        # if(colb_no_grad): scores = None
        # else: scores = self.get_cold(q_reps, p_reps, q_mask)
        with torch.no_grad() if colb_no_grad else nullcontext():
            scores = self.get_cold(q_reps, p_reps, q_mask, q_nums=q_nums, p_nums=p_nums)
        if(num_scores is None): return scores/self.model_args.colbert_temp
        num_scores = num_scores/self.model_args.num_temp
        
        alpha1, alpha2 = self.get_alpha(q_reps, p_reps, colb_scores=scores, q_nums=q_nums, p_nums=p_nums)
        
        
        multiplier = 8 if q_reps.size(0) != p_reps.size(0) else 1
        doc_pos_idx = (torch.arange(q_reps.size(0)) * multiplier).tolist()
        qry_idx = torch.arange(q_reps.size(0)).tolist()
        print("alpha", alpha1[qry_idx, doc_pos_idx].shape,q_reps.shape, p_reps.shape, q_nums["id"].shape,
              p_nums["id"].shape, num_scores[qry_idx, doc_pos_idx].shape)
        
        
    
        print(
            "NUM SCORE:",num_scores[qry_idx, doc_pos_idx], "\n",
            "Alpha:", 
            "NEGATIVE1:", num_scores[0,:32], "\n",
            # "NEGATIVE2:", num_scores[1,:32], "\n",
            "SCORE:", scores[qry_idx, doc_pos_idx]/self.model_args.colbert_temp if scores is not None else None, "\n",
            flush=True)
        
        query_nums, query_units, qids = q_nums["numbers"], q_nums["units"], query["input_ids"]
        op = q_nums["op"]
        doc_nums, doc_units, dids = p_nums["numbers"], p_nums["units"], doc["input_ids"]
        
        # for i in qry_idx:
        #     for j in (8*i  + torch.arange(8)).tolist():
        #         qry_text = self.tokenizer.decode(qids[i], skip_special_tokens=True)
        #         # qry_tokens = self.tokenizer.convert_ids_to_tokens(qids[i])
        #         doc_text = self.tokenizer.decode(dids[j], skip_special_tokens=True)
        #         # doc_tokens = self.tokenizer.convert_ids_to_tokens(dids[j])
        #         _qid = q_nums["id"][i].item()
        #         _did = p_nums["id"][j].item()
        #         # qry_nums_zip = list(zip(qry_tokens,query["number_mask"][i].sum(0).tolist()))
        #         # doc_nums_zip = list(zip(doc_tokens,doc["number_mask"][j].sum(0).tolist()))
                
        #         print(
        #             "IJ:", i, j, "\n",
        #             "SHAPE:", qids.shape, dids.shape, "\n",
        #             "QRY:", _qid, qry_text, "NUMS:", query_nums[i], [self.units_rev[k.item()] for k in query_units[i] if k.item() != -1], "\n",
        #             "OP:", op[i], "\n",
        #             "DOC:",  _did, doc_text,"NUMS:", doc_nums[j],  [self.units_rev[k.item()] for k in doc_units[j] if k.item() != -1], "\n",
        #             # "SCORE_SUM", gr_sum[i][j].item(), le_sum[i][j].item(), eq_sum[i][j].item(), "\n",
        #             # "QRY_ZIP", qry_nums_zip, "\n" ,
        #             # "DOC_ZIP", doc_nums_zip, "\n" ,
        #             "FINAL: ", num_scores[i][j].item(), "\n",
        #             "####################", 
        #             flush=True  
        #         )
        
        # for i, j in zip(qry_idx, doc_pos_idx):
        #     qry_text = self.tokenizer.decode(qids[i], skip_special_tokens=True)
        #     qry_tokens = self.tokenizer.convert_ids_to_tokens(qids[i])
        #     doc_text = self.tokenizer.decode(dids[j], skip_special_tokens=True)
        #     doc_tokens = self.tokenizer.convert_ids_to_tokens(dids[j])
        #     _qid = q_nums["id"][i].item()
        #     _did = p_nums["id"][j].item()
        #     if(scores[i][j] == 0): continue
        #     qry_nums_zip = list(zip(qry_tokens,query["number_mask"][i].sum(0).tolist()))
        #     qry_units_zip = list(zip(qry_tokens,query["unit_mask"][i].sum(0).tolist()))
        #     doc_nums_zip = list(zip(doc_tokens,doc["number_mask"][j].sum(0).tolist()))
        #     doc_units_zip = list(zip(doc_tokens,doc["unit_mask"][j].sum(0).tolist()))
            
        #     print(
        #         "IJ:", i, j, "\n",
        #         "SHAPE:", qids.shape, dids.shape, "\n",
        #         "QRY:", _qid, qry_text, "NUMS:", query_nums[i], [self.units_rev[k.item()] for k in query_units[i] if k.item() != -1], "\n",
        #         "OP:", op[i], "\n",
        #         "DOC:",  _did, doc_text,"NUMS:", doc_nums[j],  [self.units_rev[k.item()] for k in doc_units[j] if k.item() != -1], "\n",
        #         # "SCORE_SUM", gr_sum[i][j].item(), le_sum[i][j].item(), eq_sum[i][j].item(), "\n",
        #         "QRY_UNIT_ZIP", qry_units_zip, "\n" ,
        #         "DOC_UNIT_ZIP", doc_units_zip, "\n" ,
        #         "QRY_ZIP", qry_nums_zip, "\n" ,
        #         "DOC_ZIP", doc_nums_zip, "\n" ,
        #         "FINAL: ", num_scores[i][j].item(), scores[i][j].item(), "\n",
        #         "####################", 
        #         flush=True  
        #     )
        
        if(self.model_args.score_combine == "linear"):
            scores = scores/self.model_args.colbert_temp
            full_scores = alpha1*scores + alpha2*num_scores
        if(self.model_args.score_combine == "no_colb"):
            # scores = scores/self.model_args.colbert_temp
            full_scores = num_scores
        elif(self.model_args.score_combine == "linear_relu"):
            full_scores = alpha1*scores/self.model_args.colbert_temp + torch.nn.functional.relu(scores - 0.6)*alpha2*num_scores
        elif(self.model_args.score_combine == "linear_gated"):
            lamb = self.model_args.gated_lamb
            full_scores = scores*(1 + lamb*num_scores*self.model_args.num_temp)/self.model_args.colbert_temp
        elif(self.model_args.score_combine == "geometric"):
            scores = scores/self.model_args.colbert_temp
            full_scores = torch.sqrt(scores*num_scores)
        if(self.saver is not None):
            self.saver.save_data(q_nums["id"], p_nums["id"], 
                                alpha1[qry_idx, doc_pos_idx], full_scores[qry_idx, doc_pos_idx])
        return full_scores
    
    def get_alpha(self, q_reps, p_reps, colb_scores=None, q_nums=None, p_nums=None):
        if(self.model_args.deepquant_alpha_mode == "dot_cls"):
            bs_q, bs_p, d = q_reps.size(0), p_reps.size(0), q_reps.size(-1)
            q_cls = q_reps[:,0]
            p_cls = p_reps[:,0]
            q_cls = torch.Tensor.repeat(q_cls, (1,bs_p)).view(-1, d)
            p_cls = torch.Tensor.repeat(p_cls, (bs_q,1))
            probs = torch.sigmoid(torch.sum(q_cls*p_cls, dim=1))
            probs = probs.view(bs_q, bs_p)
            
            print(
                "PROBS1:", probs[0, :32],  "\n",
                "PROBS2:", probs[1, :32],  "\n",
                  flush=True)
            return 1-probs, probs
        # if(self.model_args.deepquant_alpha_mode == "cls"):
        #     bs_q, bs_p, d = q_reps.size(0), p_reps.size(0), q_reps.size(-1)
        #     q_cls = q_reps[:,0]
        #     probs = torch.sigmoid(self.alpha_project(q_cls))
        #     # print("probs", probs.shape, flush=True)
        #     probs = probs.repeat(1, bs_p)
            
        #     print(
        #         "PROBS1 cls:", probs[0, :32],  "\n",
        #         "PROBS2 cls:", probs[1, :32],  "\n",
        #           flush=True)
        #     return 1-probs, probs
        if(self.model_args.deepquant_alpha_mode == "sigmoid"):
            lamb = self.model_args.alpha_sigmoid
            probs = torch.sigmoid((colb_scores - lamb) * 10)
            print(
                "PROBS1:", probs[0, :32],  "\n",
                "PROBS2:", probs[1, :32],  "\n",
                  flush=True)
            return 1, probs
        # if(self.model_args.deepquant_alpha_mode == "colbert"):
        #     '''
        #     if colb is high then go for numeric relevance, otherwise no point
        #     thus if colb is high then alpha is high or atleast "activated"
        #     '''
            
        #     probs = self.model_args.deepquant_alpha*torch.nn.functional.relu(colb_scores - 0.6)
        #     print(
        #         "PROBS1:", probs[0, :32],  "\n",
        #         "PROBS2:", probs[1, :32],  "\n",
        #           flush=True)
        #     return probs
        # do dot prod version too
        if(self.model_args.deepquant_alpha_mode == "cross"):
            q_cls = torch.nn.functional.normalize(q_reps[:,0])
            p_cls = torch.nn.functional.normalize(p_reps[:,0])
            scores = self.deepquant_alpha_network(q_cls, p_cls)
            probs = torch.sigmoid(scores)
            print(
                "PROBS1:", probs[0, :32],  "\n",
                "PROBS2:", probs[1, :32],  "\n",
                  flush=True)
            return 1, probs
        
        return 1-self.model_args.deepquant_alpha, self.model_args.deepquant_alpha
        
    def compute_similarity(self, q_reps, p_reps, q_nums, p_nums, q_hidden, 
                           p_hidden, calc_loss=False, q_meta=None, p_meta=None, query=None, doc=None):
        n1, k1, d = q_reps.size(0), q_reps.size(1), q_reps.size(2)
        n2, k2, d = p_reps.size(0), p_reps.size(1), p_reps.size(2)
        meta = None
        q_mask, p_mask = query["attention_mask"], doc["attention_mask"]
        loss = 0
        if(self.model_args.deepquant_interaction_mode == "disjoint_quantscorer"):
            num_reshaped_scores, loss, meta = self.quantScorer(q_reps, p_reps, q_nums, 
                                                         p_nums, q_hidden, p_hidden, calc_loss, q_meta, p_meta,
                                                         query=query, doc=doc)
            
            scores = self.get_scores(q_reps, p_reps, q_mask, num_reshaped_scores, query=query, doc=doc,
                                        q_nums=q_nums, p_nums=p_nums)
        elif(self.model_args.deepquant_interaction_mode == "disjoint_quantscorer_nocolb"):
            num_reshaped_scores, loss = self.quantScorer(q_reps, p_reps, q_nums, 
                                                         p_nums, q_hidden, p_hidden, calc_loss, q_meta, p_meta,
                                                         query=query, doc=doc)
            
            scores = self.get_scores(q_reps, p_reps, q_mask, num_reshaped_scores, query=query, doc=doc,
                                        q_nums=q_nums, p_nums=p_nums, colb_no_grad=True)
        elif(self.model_args.deepquant_interaction_mode == "disjoint_qcolbert"):
            with torch.no_grad():
                num_reshaped_scores = self.qcolbert_scores(q_nums, p_nums, query, doc)
            
            scores = self.get_scores(q_reps, p_reps, q_mask, num_reshaped_scores, query=query, doc=doc,
                                        q_nums=q_nums, p_nums=p_nums)
        elif(self.model_args.deepquant_interaction_mode == "no_quant"):
            scores = self.get_scores(q_reps, p_reps, q_mask)
        else:
            raise NotImplementedError
        
        
        return scores, loss, meta

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ColbertPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = ColbertPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
'''
baseline num embedding:
    - number = exponent and mantissa (scientific notation)
    - exponent embedding is learnt (-10, 10)
    - classification task for exponent, regression for mantissa
    - custom embedding (in progress)
    - 30k and 22k
    - query - laptip under 30k,
    - rerank docs(1000)
    - query-doc pair
        - find the comparable quantities
        - regression loss
        - local
        - perturb q -> loss (gsm8k, 15 pencil vs 1l)
    
remedies:

    - add MSE loss per layer as well
    - 2.19 > NUM
        - X.XX
        2.123 > X.XXX
    - digit wise loss ? (30 and 01, so loss between digits)
        - motivation to have exactness in representation
    
pretraining:
    - identified some datasets
    - unit 
    - scale
    - unit and scale

'''