import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional
import logging
from .encoder import EncoderPooler, EncoderModel_Concept, EncoderOutput

logger = logging.getLogger(__name__)

class MyEncoderPooler_attention(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, max_len = 32,
                 project_encode = 64, project_decode = 64, num_concepts = 8):
        super().__init__()
        # self.num_concepts = num_concepts
        # self.semantic_project = nn.Linear(output_dim, input_dim)
        self.att = nn.MultiheadAttention(output_dim, output_dim//64 )
        
        
    def forward(self, input: Tensor, **kwargs):
        concepts,_ = self.att(input, input, input)
        return concepts, None, input, None

class MyEncoderPooler_ae(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, max_len = 32,
                 project_encode = 64, project_decode = 64, num_concepts = 8):
        super().__init__()
        # self.num_concepts = num_concepts
        # self.semantic_project = nn.Linear(output_dim, input_dim)
        self.query = nn.Parameter(torch.randn((num_concepts, output_dim), requires_grad=True))
        self.project_k = nn.Linear(input_dim, output_dim)
        self.project_v = nn.Linear(input_dim, output_dim)
        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        self.att = nn.MultiheadAttention(output_dim, output_dim//64, batch_first=True )
        
        self.query_recon = nn.Parameter(torch.randn((max_len-1, input_dim), requires_grad=True))
        self.project_k_recon = nn.Linear(output_dim, input_dim)
        self.project_v_recon = nn.Linear(output_dim, input_dim)
        self.ff_recon = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        self.ff2_recon = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        self.att_recon = nn.MultiheadAttention(input_dim, input_dim//64, batch_first=True )

        
    def forward(self, input: Tensor, **kwargs):
        k = self.project_k(input)
        v = self.project_v(input)
        att = torch.nn.functional.scaled_dot_product_attention(self.query, k, v,)
        att = self.ff(att)
        att, _ = self.att(att, att, att, need_weights=False)
        concepts = self.ff2(att)
        
        k_recon = self.project_k_recon(concepts)
        v_recon = self.project_v_recon(concepts)
        att_recon = torch.nn.functional.scaled_dot_product_attention(self.query_recon
                                                                     , k_recon, v_recon)
        att_recon = self.ff_recon(att_recon)
        att_recon, _ = self.att_recon(att_recon, att_recon, att_recon, need_weights=False)
        reconstruted = self.ff2(att_recon)
        
        concepts,_ = self.att(input, input, input)
        return concepts, reconstruted, input, concepts




class MyEncoderPooler(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, max_len = 32,
                 project_encode = 64, project_decode = 64, num_concepts = 8):
        super().__init__()
        self.num_concepts = num_concepts
        self.semantic_project = nn.Linear(output_dim, input_dim)
        # self.project_encode = nn.Linear(input_dim, project_encode)
        # self.project_decode = nn.Linear(output_dim, project_decode)
        
        # self.down = nn.Sequential(
        #     nn.Linear(project_encode*max_len, output_dim*num_concepts),
        #     nn.GELU(),
        #     nn.Linear(output_dim*num_concepts, output_dim*num_concepts)
        # )
        # self.up = nn.Sequential(
        #     nn.Linear(project_decode*num_concepts, input_dim*max_len),
        #     nn.GELU(),
        #     nn.Linear(input_dim*max_len, input_dim*max_len)
        # )
        self.project = project_encode
        self.project_encode = nn.Linear(input_dim, self.project)
        self.project_decode = nn.Linear(self.project, input_dim)
        # self.project_decode.weight = self.project_encode.weight
        
        self.down = nn.Sequential(
            nn.Linear(self.project*max_len, output_dim*num_concepts),
            nn.GELU(),
            nn.Linear(output_dim*num_concepts, output_dim*num_concepts)
        )
        self.up = nn.Sequential(
            nn.Linear(output_dim*num_concepts, output_dim*num_concepts),
            nn.GELU(),
            nn.Linear(output_dim*num_concepts, self.project*max_len)
        )
        # self.up[0].weight = self.down[2].weight
        # self.up[2].weight = self.down[0].weight
        
    def forward(self, input: Tensor, **kwargs):
        batch = input.size(0)
        projected_encode = self.project_encode(input)
        flat = projected_encode.view(batch, -1)
        down = self.down(flat)
        concepts = down.view(batch, self.num_concepts, -1)
        # print("Down shape", down.shape)
        up = self.up(down)
        # print("upshape", up.view(batch, -1, self.project).shape)
        projected_decode = self.project_decode(up.view(batch, -1, self.project))
        reconstructed = projected_decode.view(input.shape)
        
        semantic_project = self.semantic_project(concepts)
        return concepts, reconstructed, input, semantic_project
        # return concepts, reconstructed, input, None
        
    def old_forward(self, input: Tensor, **kwargs):
        batch = input.size(0)
        projected_encode = self.project_encode(input)
        flat = projected_encode.view(batch, -1)
        concepts = self.concepts(flat).view(batch, self.num_concepts, -1)
        projected_decode = self.project_decode(concepts)
        reconstructed = self.up(projected_decode.view(batch, -1))
        reconstructed = reconstructed.view(input.shape)
        
        semantic_project = self.semantic_project(concepts)
        return concepts, reconstructed, input, semantic_project
        # return concepts, reconstructed, input, None

# model_msmarco_concept_768_b32_n8_cq8cp16 is lrl_sem
class ConceptPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, tied=True, q_len=32,
                 p_len=128, num_concepts_q=8, num_concepts_p=8, pooler_version="normal"):
        super(ConceptPooler, self).__init__()
        print("Pooler with concepts", num_concepts_q, num_concepts_p, pooler_version)
        print("dims", input_dim, output_dim)
        torch.autograd.set_detect_anomaly(True)
        project_encode = 64
        project_decode = 64
        pooler = MyEncoderPooler_ae if pooler_version == "attention_ae" else None
        self.query_pooler = pooler(max_len=q_len, 
                                        project_decode=project_decode, project_encode=project_encode,
                                        num_concepts=num_concepts_q, 
                                        input_dim=input_dim, output_dim=output_dim)
        self.passage_pooler = pooler(max_len=p_len, 
                                        project_decode=project_decode, project_encode=project_encode,
                                        num_concepts=num_concepts_p, 
                                        input_dim=input_dim, output_dim=output_dim)
        
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 
                        'tied': tied, 'q_len': q_len, 'p_len': p_len,
                        'num_concepts_q': num_concepts_q, 'num_concepts_p': num_concepts_p,
                        'pooler_version': pooler_version
                        }

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.query_pooler(q)
        
        elif p is not None:
            return self.passage_pooler(p)
        else:
            raise ValueError


class ConceptModel(EncoderModel_Concept):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        # clear masked tokens
        # p_hidden *= psg['attention_mask'][:, :, None].float()
        
        p_reps, recon_p, hidden_p, sem_p = self.pooler(p=p_hidden[:, 1:])
        if(hidden_p is not None): hidden_p *= psg['attention_mask'][:, 1:, None].float()
        if(recon_p is not None): recon_p *= psg['attention_mask'][:, 1:, None].float()
        return p_reps, recon_p, hidden_p, sem_p

    def encode_query(self, qry):
        if qry is None:
            return None
        # print(qry)
        # raise "sd"
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        # clear masked tokens
        # q_hidden *= qry['attention_mask'][:, :, None].float()
        
        q_reps, recon_q, hidden_q, sem_q = self.pooler(q=q_hidden[:, 1:])
        # print("hidden", hidden_q.shape, recon_q.shape, qry['attention_mask'].shape, qry['attention_mask'][:, 1:, None].shape )
        if(hidden_q is not None): hidden_q *= qry['attention_mask'][:, 1:, None].float()
        if(recon_q is not None): recon_q *= qry['attention_mask'][:, 1:, None].float()
        return q_reps, recon_q, hidden_q, sem_q

    def compute_similarity(self, q_reps, p_reps):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1)
        return scores

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
            pooler_version = model_args.pooler_version,
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = None
        p_reps = None
        if(query is not None):
            q_reps, reconstruct_q, hidden_q, semantic_project_q = self.encode_query(query)
            # q_reps, reconstruct_q, hidden_q= self.encode_query(query)
        if(passage is not None):
            p_reps, reconstruct_p, hidden_p, semantic_project_p = self.encode_passage(passage)
            # p_reps, reconstruct_p, hidden_p = self.encode_passage(passage)
        
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
                
                reconstruct_q = self._dist_gather_tensor(reconstruct_q)
                reconstruct_p = self._dist_gather_tensor(reconstruct_p)
                
                hidden_q = self._dist_gather_tensor(hidden_q)
                hidden_p = self._dist_gather_tensor(hidden_p)
                if(semantic_project_q is not None):
                    semantic_project_q = self._dist_gather_tensor(semantic_project_q)
                    semantic_project_p = self._dist_gather_tensor(semantic_project_p)
                
            # mask_q = torch.ones((q_reps.size(1), q_reps.size(1))).fill_diagonal_(0)
            # mask_q = mask_q.unsqueeze(0).repeat(q_reps.size(0), 1, 1).to(device=q_reps.device)
            # mask_p = torch.ones((p_reps.size(1), p_reps.size(1))).fill_diagonal_(0)
            # mask_p = mask_p.unsqueeze(0).repeat(p_reps.size(0), 1, 1).to(device=p_reps.device)
                
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss_ce = self.compute_loss(scores, target)
            
            loss_recon = 0
            
            if(reconstruct_p is not None and reconstruct_q is not None):
                loss_recon_q = self.compute_reconstruction_loss(hidden_q, reconstruct_q)
                loss_recon_p = self.compute_reconstruction_loss(hidden_p, reconstruct_p)
                loss_recon = loss_recon_p + loss_recon_q
                
            
            # loss_indep_q = torch.sum(torch.matmul(q_reps, q_reps.transpose(1,2))*mask_q)
            # loss_indep_p = torch.sum(torch.matmul(p_reps, p_reps.transpose(1,2))*mask_p)
            
            # q_reps_norm = F.normalize(q_reps, p=2, dim=-1)
            # p_reps_norm = F.normalize(p_reps, p=2, dim=-1)
            # loss_indep_q = torch.sum(torch.einsum('bnd,bmd -> bnm', q_reps_norm, q_reps_norm)*mask_q)
            # loss_indep_p = torch.sum(torch.einsum('bnd,bmd -> bnm', p_reps_norm, p_reps_norm)*mask_p)
            
            
            # scores_r_q = (q_reps@q_reps.transpose(1,2)).view(-1, q_reps.size(1)) # B,N,D -> B,N,N -> B*N, N
            scores_r_q = (q_reps@q_reps.transpose(1,2)).view(-1, q_reps.size(1)) # B,N,D -> B,N,N -> B*N, N
            targets_r_q = torch.arange(q_reps.size(1), device=scores.device, dtype=torch.long)
            targets_r_q = targets_r_q.unsqueeze(0).repeat(q_reps.size(0), 1).to(device=q_reps.device)
            targets_r_q = targets_r_q.view(-1)
            
            scores_r_p = (p_reps@p_reps.transpose(1,2)).view(-1, p_reps.size(1)) # B,N,D -> B,N,N -> B, N*N
            targets_r_p = torch.arange(p_reps.size(1), device=scores.device, dtype=torch.long)
            targets_r_p = targets_r_p.unsqueeze(0).repeat(p_reps.size(0), 1).to(device=p_reps.device)
            targets_r_p = targets_r_p.view(-1)
            
            loss_indep_q = self.compute_loss(scores_r_q, targets_r_q)
            loss_indep_p = self.compute_loss(scores_r_p, targets_r_p)
            loss_indep = (loss_indep_q + loss_indep_p)
            
            loss_semantic = 0
            if(self.model_args and self.model_args.semantic_loss is not None):
                loss_semantic_q = self.semanticLoss(hidden_q, semantic_project_q)
                loss_semantic_p = self.semanticLoss(hidden_p, semantic_project_p)
                loss_semantic = loss_semantic_q + loss_semantic_p
            
            loss = loss_ce + loss_recon + loss_indep + loss_semantic
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



