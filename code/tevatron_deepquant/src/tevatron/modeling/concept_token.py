import torch
import torch.nn as nn
from torch import Tensor
import logging
import tevatron.arguments as tevargs 
from .encoder import EncoderPooler, EncoderModel_Concept_token, set_pos_embs

logger = logging.getLogger(__name__)

class MyEncoderPoolerTokenV2(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super().__init__()
        
        
        self.project = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.GELU(),
            nn.Linear(4*input_dim, output_dim),
        )
        
        self.lm = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.GELU(),
            nn.Linear(4*input_dim, tevargs.VOCAB_SIZE),
        )
    
    def project_only(self, input: Tensor, attn_mask: Tensor = None, num_concepts = None):
        concepts = input[:,1:1+num_concepts]
        concepts = self.project(concepts)
        tokens = input[:, 1+num_concepts:, ]
        lm_concepts = self.lm(concepts)
        lm_tokens = self.lm(tokens)
        return concepts, lm_concepts, lm_tokens
        
    def forward(self, input: Tensor, attn_mask: Tensor = None, num_concepts = 8, **kwargs):
        concepts, lm_concepts, lm_tokens = self.project_only(input, attn_mask, num_concepts=num_concepts)
        return concepts, lm_concepts, lm_tokens


class MyEncoderPoolerToken(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.att_in = nn.MultiheadAttention(input_dim, input_dim//64, 
                                            batch_first=True)
        # self.project_att_out = nn.Linear(input_dim, output_dim)
        self.att_out = nn.MultiheadAttention(output_dim, output_dim//64, 
                                             batch_first=True)
        
        # self.transformer = nn.TransformerEncoderLayer(output_dim, output_dim//64,
        #                                               batch_first=True, norm_first=True)
        # # self.att2 = nn.MultiheadAttention(input_dim, input_dim//64)
        # self.ff = self.project = nn.Sequential(
        #     nn.Linear(input_dim, 4*input_dim),
        #     nn.GELU(),
        #     nn.Linear(4*input_dim, output_dim),
        # )
        
        self.project = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.GELU(),
            nn.Linear(4*input_dim, output_dim),
        )
        
    # def project_transformer(self, input: Tensor, attn_mask: Tensor = None):
    #     tokens = input[:,1:1+self.num_concepts]
    #     projected = self.project(tokens)
    #     qkv = projected
    #     # print("input", qkv)
    #     emb = self.transformer(qkv)
    #     print("emb", emb.shape, projected.shape)
    #     return emb
    
    def project_attend(self, input: Tensor, attn_mask: Tensor = None, num_concepts = None):
        tokens = input[:,1:1+num_concepts]
        projected = self.project(tokens)
        qkv = projected
        projected_att, _ = self.att_out(qkv, qkv, qkv, need_weights=False)
        return projected_att
    
    def attend_project(self, input: Tensor, attn_mask: Tensor = None, num_concepts = None):
        tokens = input[:,1:1+num_concepts]
        qkv = tokens
        att, _ = self.att_in(qkv, qkv, qkv, need_weights=False)
        projected = self.project(att)
        return projected
    
    def project_only(self, input: Tensor, attn_mask: Tensor = None, num_concepts = None):
        tokens = input[:,1:1+num_concepts]
        projected = self.project(tokens)
        return projected
        
    def forward(self, input: Tensor, attn_mask: Tensor = None, num_concepts = 8, **kwargs):
        projected = self.project_only(input, attn_mask, num_concepts=num_concepts)
        return projected

class ConceptPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True, q_len=32,
                 p_len=128, num_concepts_q=8, num_concepts_p=8):
        super(ConceptPooler, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.num_concepts_p = num_concepts_p
        self.num_concepts_q = num_concepts_q
        print("Pooler with concepts", num_concepts_q, num_concepts_p)
        print("Tied", tied)
        self.query_pooler = MyEncoderPoolerToken(input_dim, output_dim)
        if(tied):
            self.passage_pooler = self.query_pooler
        else:
            self.passage_pooler = MyEncoderPoolerToken(input_dim, output_dim)
        
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 
                        'tied': tied, 'q_len': q_len, 'p_len': p_len,
                        'num_concepts_q': num_concepts_q, 'num_concepts_p': num_concepts_p
                        }

    def forward(self, q: Tensor = None, p: Tensor = None, attn_mask = None, **kwargs):
        if q is not None:
            return self.query_pooler(q, attn_mask,  num_concepts = self.num_concepts_q)
        elif p is not None:
            return self.passage_pooler(p, attn_mask, num_concepts = self.num_concepts_p )
        else:
            raise ValueError


class ConceptModel_Token(EncoderModel_Concept_token):
    
    def getHiddenAndLogits(self, out):
        if(self.model_args.lm_head):
            return out.hidden_states[-1], out.logits
        
        return out.last_hidden_state, None

        
    
    def encode_passage(self, psg, cls_mean_pool=False):
        if psg is None:
            return None
        att = psg["attention_mask"]
        # if(self.model_args.mask_concept_tokens):
        #     max_p =  att[0].size(0)
        #     batch = att.size(0)
        #     c_p = self.model_args.num_concepts_p
        #     att = att.repeat(1,max_p).view(batch, max_p, -1)
        #     att[:, 0, 1:1+c_p] = 0
        #     att[:, 1+c_p: ,1:1+c_p] = 0
        #     att = att.unsqueeze(1)
            
        psg_out = self.lm_p(input_ids=psg["input_ids"], attention_mask=att, return_dict=True)
        p_hidden, lm_logits = self.getHiddenAndLogits(psg_out)
        # print("phidden", p_hidden.shape)
        num_concepts = self.model_args.num_concepts_p
        if(lm_logits is not None):
            concept_logits = lm_logits[:,1:1+num_concepts]
            token_logits = lm_logits[:,1+num_concepts:]
        
        if(cls_mean_pool):
            cls_reps = torch.mean(p_hidden[:, 1+num_concepts:, :], dim=1)
        else: cls_reps = p_hidden[:, 0]
        
        if(self.model_args.add_pooler):
            p_reps = self.pooler(p=p_hidden, attn_mask=psg["attention_mask"])
        else:
            p_reps = p_hidden[:,1:1+num_concepts]
            
        return p_reps, cls_reps, p_hidden[:, 1+num_concepts:, :],  concept_logits, token_logits


    def encode_query(self, qry, cls_mean_pool=False):
        if qry is None:
            return None
        att = qry["attention_mask"]
        
        # if(self.model_args.mask_concept_tokens):
        #     max_q =  att[0].size(0)
        #     batch = att.size(0)
        #     c_q = self.model_args.num_concepts_q
        #     # num_heads = self.model.config.num_attention_heads
        #     att = att.repeat(1,max_q).view(batch, max_q, -1)
        #     att[:, 0, 1:1+c_q] = 0
        #     att[:, 1+c_q: ,1:1+c_q] = 0
        #     att = att.unsqueeze(1)
        # set_pos_embs(self.lm_q, self.pos_embs_q)
        qry_out = self.lm_q(input_ids=qry["input_ids"], attention_mask=att, return_dict=True)
        q_hidden, lm_logits = self.getHiddenAndLogits(qry_out)
        num_concepts = self.model_args.num_concepts_q
        if(lm_logits is not None):
            concept_logits = lm_logits[:,1:1+num_concepts]
            token_logits = lm_logits[:,1+num_concepts:]
         
        # print("qhidden", q_hidden.shape)
        if(cls_mean_pool):
            cls_reps = torch.mean(q_hidden[:, 1+num_concepts:, :], dim=1)
        else: cls_reps = q_hidden[:, 0]
        # q_reps = q_hidden[:,1:9]
        if(self.model_args.add_pooler):
            q_reps = self.pooler(q=q_hidden, attn_mask=qry["attention_mask"])
        else:
            q_reps = q_hidden[:,1:1+num_concepts]
        return q_reps, cls_reps, q_hidden[:, 1+num_concepts:, :], concept_logits, token_logits
    
    def sim_dot(self, q_reps, p_reps):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        top = self.model_args.top_sim
        if(self.model_args.alignment_matrix):
            scores = torch.einsum('qipj,ij->qp', token_scores, self.alignment_matrix)
        elif top > 1:
            scores = token_scores.topk(top, dim=-1)[0].mean(-1)
            scores = scores.sum(1)
        else:
            scores, _ = token_scores.max(-1)
            scores = scores.sum(1)
        return scores
    
    def sim_dot_asymmetric(self, q_reps, p_reps):
        
        q_reps_alt = torch.einsum('qid,dd->qid', q_reps, self.W)
        token_scores = torch.einsum('qid,pjd->qipj', q_reps_alt, p_reps)
        top = self.model_args.top_sim
        if(self.model_args.alignment_matrix):
            scores = torch.einsum('qipj,ij->qp', token_scores, self.alignment_matrix)
        elif top > 1:
            scores = token_scores.topk(top, dim=-1)[0].mean(-1)
            scores = scores.sum(1)
        else:
            scores, _ = token_scores.max(-1)
            scores = scores.sum(1)
        return scores
    
    def sim_hinge(self, q_reps, p_reps):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        top = self.model_args.top_sim
        if(self.model_args.alignment_matrix):
            scores = torch.einsum('qipj,ij->qp', token_scores, self.alignment_matrix)
        elif top > 1:
            scores = token_scores.topk(top, dim=-1)[0].mean(-1)
            scores = scores.sum(1)
        else:
            scores, _ = token_scores.max(-1)
            scores = scores.sum(1)
        return scores

    def compute_similarity(self, q_reps, p_reps):
        mode = self.model_args.score_func
        if(mode == "dot"):
            return self.sim_dot(q_reps, p_reps)
        if(mode == "dot_asym"):
            return self.sim_dot_asymmetric(q_reps, p_reps)
        if(mode == "hinge"):
            return self.sim_hinge(q_reps, p_reps)
    
    def add_tokens_model(self, tokenizer):
        self.lm_q.resize_token_embeddings(len(tokenizer))
        self.lm_p.resize_token_embeddings(len(tokenizer))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ConceptPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = ConceptPooler(
            input_dim=model_args.projection_in_dim,
            output_dim=model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
            q_len=model_args.q_len,
            p_len=model_args.p_len,
            num_concepts_q=model_args.num_concepts_q,
            num_concepts_p=model_args.num_concepts_p,
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
