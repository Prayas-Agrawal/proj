import torch
import torch.nn as nn
import torch.nn.functional as F
from tevatron.arguments import ModelArguments
from .helpers.helpers import *


class QuantScoring(nn.Module):
    def __init__(self, node_dim, model_args:ModelArguments):
        super(QuantScoring, self).__init__()
        self.combine_funcs = nn.Linear(3,1)
        self.asym_dot_reg_alpha = 0.5
        self.model_args = model_args
        self.num_dim = self.model_args.projection_in_dim if self.model_args.num_reps_from_hidden else self.model_args.projection_out_dim
        if(model_args.add_num_pred_loss):
            self.numberpredicitonloss = NumberPredictionLoss(frac_exp=model_args.embed_exp_frac,
                                                             output_dim=model_args.projection_in_dim)
        self.feature_transform = nn.ModuleDict({
            "query_greater": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "greater": FFNLayer(node_dim, node_dim, 1, 0.2),
            "query_lesser": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "query_equal": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "psg": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "query": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "shared": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "passage": FFNLayer(node_dim, node_dim, node_dim, 0.2),
        })
        self.joint = nn.ModuleDict({
            "greater": FFNLayer(self.num_dim, self.num_dim, self.num_dim, 0.2),
            "lesser": FFNLayer(self.num_dim, self.num_dim, self.num_dim, 0.2),
            "equal": FFNLayer(self.num_dim, self.num_dim, self.num_dim, 0.2),
        })
        self.num_rel_score = nn.ModuleDict({
            "greater": FFNLayer(self.num_dim*2, self.num_dim, 1, 0.2),
            "lesser": FFNLayer(self.num_dim*2, self.num_dim, 1, 0.2),
            "equal": FFNLayer(self.num_dim*2, self.num_dim, 1, 0.2),
        })
        self.combine_noeq = nn.Linear(node_dim, 2)
        self.att_combine = nn.ModuleDict({
            "query_q_greater": nn.Linear(node_dim, node_dim),
            "query_q_lesser": nn.Linear(node_dim, node_dim),
            "query_q_equal": nn.Linear(node_dim, node_dim),
            "passage_k": nn.Linear(node_dim, node_dim),
        })
        self.get_unit = nn.Linear(node_dim, node_dim//2)
        self.attrib_dist = nn.Linear(node_dim, model_args.latent_attribs)
    
        self.spherical = nn.ModuleDict({
            "query": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "passage": FFNLayer(node_dim, node_dim, node_dim, 0.2),
            "shared": FFNLayer(node_dim, 32, node_dim, 0.2),
        })
        
        self.relation_classifier = FFNLayer(node_dim, node_dim, 3, 0.2)

        self.node_dim = node_dim
        self.dim_combine = nn.ModuleDict({
            "greater" : nn.Linear(self.num_dim, 1),
            "lesser" : nn.Linear(self.num_dim, 1),
            "equal" : nn.Linear(self.num_dim, 1)
        })
        self.layers_Q = nn.ModuleDict({
            "all": nn.Linear(self.num_dim, node_dim),
            "greater" : nn.Linear(self.num_dim, node_dim),
            "lesser" : nn.Linear(self.num_dim, node_dim),
            "equal" : nn.Linear(self.num_dim, node_dim)
        })
        self.layers_K = nn.ModuleDict({
            "all": nn.Linear(self.num_dim, node_dim),
            "greater" : nn.Linear(self.num_dim, node_dim),
            "lesser" : nn.Linear(self.num_dim, node_dim),
            "equal" : nn.Linear(self.num_dim, node_dim)
        })
        self.layers_V = nn.ModuleDict( {
            "all": nn.Linear(self.num_dim, node_dim),
            "greater" : nn.Linear(self.num_dim, node_dim),
            "lesser" : nn.Linear(self.num_dim, node_dim),
            "equal" : nn.Linear(self.num_dim, node_dim)
        })
        self.layers_V2 = nn.ModuleDict( {
            "all": nn.Linear(1, node_dim),
            "greater" : nn.Linear(1, node_dim),
            "lesser" : nn.Linear(1, node_dim),
            "equal" : nn.Linear(1, node_dim)
        })
        self.relation_pred_network = PairwiseScores_Cross(node_dim)
        self.num_scores_network = PairwiseScores_Cross(node_dim, num_layers=2)
        self.relation_score_joint_network = PairwiseScores_multipred(node_dim, num_layers=3, out_dim1=3, out_dim2=3)
        self.relation_score_same_network = PairwiseScores_Cross(node_dim, num_layers=3, out_dim=3)
        self.numreps = NumReps(node_dim=node_dim, model_args=model_args)
        self.softplus = nn.Softplus()
        self.collate_mode = model_args.deepquant_quantscoring_collate_mode
        self.combine_scores_mode =  model_args.deepquant_quantscoring_combine_scores_mode
        self.ver = model_args.deepquant_quantscoring_ver
        self.add_relation_classifier_loss = model_args.add_relation_classifier_loss
        self.cos_kernel = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.equal_score = model_args.equal_score
        self.asymdot_loss = model_args.asymdot_loss
        self.nummask_expand = model_args.expand_nummask_window
        self.latent_attribs =  model_args.latent_attribs
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def negative_correlation_loss(self, A, B):
        A = (A - A.mean()) / A.std()
        B = (B - B.mean()) / B.std()
        correlation = (A * B).mean()
        return 1 + correlation
    
    def _softmax(self, a, b):
        diff = b - a
        return torch.where(
            diff > 0,
            torch.exp(-diff) / (1 + torch.exp(-diff)),  # Stable when diff > 0
            1 / (1 + torch.exp(diff))                   # Stable when diff <= 0
        )
    def _softmax_sigmoid(self, a, b):
        _a, _b = torch.sigmoid(a), torch.sigmoid(b)
        return 1/(1 + torch.exp(_b-_a))
        
    def loss(self, Q, P):
        def rel_loss(key):
            p_inv = torch.nn.functional.normalize(self.feature_transform[key](P), dim=2)
            q = torch.nn.functional.normalize(self.feature_transform[key](Q), dim=2)
            q_inv_psg = torch.nn.functional.normalize(self.feature_transform["psg"](Q), dim=2)
            p = torch.nn.functional.normalize(self.feature_transform["psg"](P), dim=2)
            score = torch.matmul(q, torch.transpose(p, 1,2))
            inv_score = torch.matmul(q_inv_psg, torch.transpose(p_inv, 1,2))
            # score_mat = score + inv_score
            score_mat = (1-(score + inv_score))
            max_nums_q = Q.shape[1]
            max_nums_p = P.shape[1]
            denom = torch.sqrt(torch.tensor([max_nums_q*max_nums_p]).unsqueeze(0)).to(torch.get_device(Q))
            loss = torch.linalg.matrix_norm(score_mat, ord="fro")/denom
            loss = torch.mean(loss, dim=1)[0]
            return loss
            
        greater_loss = rel_loss("query_greater")
        lesser_loss = rel_loss("query_lesser")
        loss = greater_loss + lesser_loss
        
        
        return loss
        
    def lossv2(self, Q, P):
        def rel_loss():
            q_gr = torch.nn.functional.normalize(self.feature_transform["query_greater"](self.layers_V["greater"](Q)), dim=2)
            q_less = torch.nn.functional.normalize(self.feature_transform["query_lesser"](self.layers_V["lesser"](Q)), dim=2)
            p_gr = torch.nn.functional.normalize(self.feature_transform["psg"](self.layers_V["greater"](P)), dim=2)
            p_less = torch.nn.functional.normalize(self.feature_transform["psg"](self.layers_V["lesser"](P)), dim=2)
            score_gr = torch.matmul(q_gr, torch.transpose(p_gr, 1,2))
            score_less = torch.matmul(q_less, torch.transpose(p_less, 1,2))
            # score_mat = score + inv_score
            score_mat = (1-(score_gr + score_less))
            max_nums_q = Q.shape[1]
            max_nums_p = P.shape[1]
            denom = torch.sqrt(torch.tensor([max_nums_q*max_nums_p]).unsqueeze(0)).to(torch.get_device(Q))
            loss = torch.linalg.matrix_norm(score_mat, ord="fro")/denom
            loss = torch.mean(loss, dim=1)[0]
            return loss
            
        return rel_loss()
    
    def loss_minproduct(self, Q, P):
        def rel_loss():
            q_gr = torch.nn.functional.normalize(self.feature_transform["query_greater"](Q), dim=2)
            q_less = torch.nn.functional.normalize(self.feature_transform["query_lesser"](Q), dim=2)
            p = torch.nn.functional.normalize(self.feature_transform["psg"](P), dim=2)
            # B, max_q, max_p
            score_gr = torch.matmul(q_gr, torch.transpose(p, 1,2))
            score_less = torch.matmul(q_less, torch.transpose(p, 1,2))
            # score_mat = score + inv_score
            loss = self.negative_correlation_loss(score_gr, score_less)
            return loss
            
        return rel_loss()
    
    def get_all_scores_joint(self, Q, P, q_nums_nz, p_nums_nz):
        
        ver = self.ver
        if(ver == "sigmoid"):
            new_q = Q
            new_p = P
            q = self.feature_transform["query"](new_q)
            p = self.feature_transform["passage"](new_p)
            qp = q.unsqueeze(2) - p.unsqueeze(1)
            
            gr = torch.sigmoid(self.feature_transform["greater"](qp)).squeeze(-1)
            le = 1- gr
            norm_qp = torch.norm(qp, dim=-1)
            eq = torch.exp(-norm_qp/0.5)
            scores = torch.stack((gr,le,eq), dim=-1)
            
        elif(ver == "asym_dot"):
            new_p = self.feature_transform["psg"](P)
            new_p = torch.nn.functional.normalize(new_p, dim=2)
            
            new_q_gr = self.feature_transform["query_greater"](Q)
            new_q_gr = torch.nn.functional.normalize(new_q_gr, dim=2)
            gr = torch.matmul(new_q_gr, torch.transpose(new_p, 1,2))
            
            new_q_le = self.feature_transform["query_lesser"](Q)
            new_q_le = torch.nn.functional.normalize(new_q_le, dim=2)
            le = torch.matmul(new_q_le, torch.transpose(new_p, 1,2))
            
            new_q_eq = self.feature_transform["query_equal"](Q)
            new_q_eq = torch.nn.functional.normalize(new_q_eq, dim=2)
            eq = torch.matmul(new_q_eq, torch.transpose(new_p, 1,2))
            scores = torch.stack((gr,le,eq), dim=-1)
            
        elif(ver == "asym_dot_eq_laplace"):
            new_p = self.feature_transform["psg"](P)
            new_p = torch.nn.functional.normalize(new_p, dim=2)
            
            new_q_gr = self.feature_transform["query_greater"](Q)
            new_q_gr = torch.nn.functional.normalize(new_q_gr, dim=2)
            gr = torch.matmul(new_q_gr, torch.transpose(new_p, 1,2))
            
            new_q_le = self.feature_transform["query_lesser"](Q)
            new_q_le = torch.nn.functional.normalize(new_q_le, dim=2)
            le = torch.matmul(new_q_le, torch.transpose(new_p, 1,2))
            
            eq = torch.exp(-torch.abs(gr-le))
            scores = torch.stack((gr,le,eq), dim=-1)
            
        elif(ver == "asym_dot_eq_laplacev2"):
            new_p = self.feature_transform["psg"](P)
            new_p = torch.nn.functional.normalize(new_p, dim=2)
            
            new_q_gr = self.feature_transform["query_greater"](Q)
            new_q_gr = torch.nn.functional.normalize(new_q_gr, dim=2)
            gr = torch.matmul(new_q_gr, torch.transpose(new_p, 1,2))
            
            new_q_le = self.feature_transform["query_lesser"](Q)
            new_q_le = torch.nn.functional.normalize(new_q_le, dim=2)
            le = torch.matmul(new_q_le, torch.transpose(new_p, 1,2))
            
            new_q_eq = self.feature_transform["query_equal"](Q)
            new_q_eq = torch.nn.functional.normalize(new_q_eq, dim=2)
            eq = torch.matmul(new_q_eq, torch.transpose(new_p, 1,2))
            eq = torch.exp(-torch.abs(eq))
            scores = torch.stack((gr,le,eq), dim=-1)
            
        elif(ver == "same"):
            scores = self.relation_score_same_network(Q,P, q_nums_nz, p_nums_nz)
            
        return scores

    def get_pairwise_probs(self, Q_q, K_p, q_nums_nz, p_nums_nz, qnum_meta=None, pnum_meta=None):
        # find the most suitable quantity in doc for every quant in query
        # FIXME: Recheck this
        mode = self.model_args.DEBUG_pairprob 
        loss = 0
        gold = None
        def get_unit_mask(qnum_meta, pnum_meta):
            bs_q = qnum_meta["units"].size(0)
            bs_p = pnum_meta["units"].size(0)
            query_units = torch.Tensor.repeat(qnum_meta["units"], (1,bs_p)).view(-1, qnum_meta["numbers"].size(-1))
            doc_units = torch.Tensor.repeat(pnum_meta["units"], (bs_q, 1))
            
            query_units_exp = query_units.unsqueeze(2)  # (batch_size, max_nums_query, 1)
            doc_units_exp = doc_units.unsqueeze(1)  # (batch_size, 1, max_nums_doc)
            
            matching_units = (query_units_exp == doc_units_exp) & (query_units_exp != -1) & (doc_units_exp != -1)
            return matching_units
        
        def masked_bce_loss(scores, gold, valid_mask):
            valid_mask = valid_mask.float()
            valid_gold = gold.float() * valid_mask  
            valid_scores = scores * valid_mask 

            bce_loss = F.binary_cross_entropy_with_logits(valid_scores, valid_gold, reduction='none')
            masked_loss = bce_loss * valid_mask
            loss = masked_loss.sum() / valid_mask.sum()
            return loss
        
        if(mode == "pred"):
            pairwise_prob = Q_q@torch.transpose(K_p, 1,2)/self.model_args.quant_pairwiseprob_temp
            combined_mask = q_nums_nz.unsqueeze(2) & p_nums_nz.unsqueeze(1) 
            mask_sums = combined_mask.sum(dim=2, keepdim=True)  # Sum along N2 to check rows
            valid_rows = mask_sums > 0
            # print("BEFORE", pairwise_prob[0], flush=True )
            
            # print("MAKS", combined_mask[0], flush=True)
            
            masked_score = pairwise_prob.masked_fill(~combined_mask, float('-inf'))
            masked_score = masked_score.masked_fill(~valid_rows, 0)
            # Avoid NaNs by setting scores for invalid rows to 0
            if(self.model_args.REMOVE_pairwise_prog_norm == "sigmoid"):
                pairwise_prob = torch.sigmoid(masked_score) * combined_mask
            else:
                pairwise_prob = F.softmax(masked_score, dim=2) * combined_mask
            # OLD...
            # print("PROBS", pairwise_prob[0], flush=True)
            # pairwise_prob = q_nums_nz.float().unsqueeze(-1)*pairwise_prob
            # pairwise_prob = p_nums_nz.float().unsqueeze(1)*pairwise_prob
            # pairwise_prob = torch.softmax(pairwise_prob, dim=2)
        elif mode == "raw":
            matching_units = get_unit_mask(qnum_meta, pnum_meta)
            matching_units_sum = matching_units.long().sum(2).unsqueeze(-1)
            pairwise_prob = torch.where(matching_units_sum != 0, 
                                        matching_units.float() / matching_units_sum, torch.tensor(0.0))
        elif(mode == "pred_loss"):
            score_mode = self.model_args.REMOVE_pairwise_prog_score
            if(score_mode == "dot"):
                pairwise_prob = Q_q@torch.transpose(K_p, 1,2)/self.model_args.quant_pairwiseprob_temp
            if(score_mode == "cosine"):
                Q_q, K_p = torch.nn.functional.normalize(Q_q, dim=-1), torch.nn.functional.normalize(K_p, dim=-1)
                pairwise_prob = Q_q@torch.transpose(K_p, 1,2)/self.model_args.quant_pairwiseprob_temp
            combined_mask = q_nums_nz.unsqueeze(2) & p_nums_nz.unsqueeze(1) 
            mask_sums = combined_mask.sum(dim=2, keepdim=True)  # Sum along N2 to check rows
            valid_rows = mask_sums > 0
            
            masked_score = pairwise_prob.masked_fill(~combined_mask, float('-inf'))
            masked_score = masked_score.masked_fill(~valid_rows, 0)
            
            matching_units = get_unit_mask(qnum_meta, pnum_meta)
            gold = matching_units
            loss = masked_bce_loss(pairwise_prob, matching_units, combined_mask)
            
            # Avoid NaNs by setting scores for invalid rows to 0
            if(self.model_args.REMOVE_pairwise_prog_norm == "sigmoid"):
                pairwise_prob = torch.sigmoid(masked_score) * combined_mask
            else:
                pairwise_prob = F.softmax(masked_score, dim=2) * combined_mask
        
        return pairwise_prob, loss, gold
    

    
    def get_relevance(self, nums_q_reps, nums_p_reps, nums_q_reps_exp, 
                  nums_p_reps_exp, q_nums_nz, p_nums_nz, qnum_meta, 
                  pnum_meta, q_reps, p_reps, num_q_ctx_reps = None, num_p_ctx_reps=None,
                  query=None, doc=None):
        loss = 0
        qnum_ctx, pnum_ctx = self.layers_Q["all"](num_q_ctx_reps), self.layers_Q["all"](num_p_ctx_reps)
        q_rawnum, p_rawnum = qnum_meta["numbers"], pnum_meta['numbers']
        bs_q, bs_p = q_rawnum.size(0), p_rawnum.size(0)
        q_rawnum = torch.Tensor.repeat(q_rawnum, (1, bs_p)).view(-1, q_rawnum.size(1)).unsqueeze(-1)
        p_rawnum = torch.Tensor.repeat(p_rawnum, (bs_q, 1)).unsqueeze(-1)
        
        V_q, V_p = self.layers_V2["all"](q_rawnum), self.layers_V2["all"](p_rawnum)
        A, B = V_q, V_p
        
        relpred = self.model_args.DEBUG_score_relpred_num
        PRINT_RELPRED_DEBUG = True
        if(relpred == "joint"):
            relation_pred_scores, num_rel_scores = self.relation_score_joint_network(A,B, q_nums_nz, p_nums_nz)
            relation_pred_scores = relation_pred_scores/self.model_args.quant_relationpred_temp
            if(self.model_args.add_num_relation_pred_loss):
                num_relation_loss, labels, pairs = self.number_relation_loss(relation_pred_scores, qnum_meta, pnum_meta, return_pairs=True)
                print("NUM REL LOSS:", num_relation_loss, flush=True)
                loss += num_relation_loss
            relation_pred_scores = torch.softmax(relation_pred_scores, dim=-1)
            num_rel_scores = torch.softmax(num_rel_scores, dim=-1)
        elif(relpred == "same"):
            num_rel_scores = self.relation_score_same_network(A,B, q_nums_nz, p_nums_nz)
            rel_pred = num_rel_scores
            # rel_pred = rel_pred/self.model_args.quant_relationpred_temp
            if(self.model_args.add_num_relation_pred_loss):
                num_relation_loss, labels, pairs = self.number_relation_loss(rel_pred, qnum_meta, pnum_meta, return_pairs=True)
                print("NUM REL LOSS:", num_relation_loss, flush=True)
                loss += num_relation_loss
            relation_pred_scores = 1
            num_rel_scores = torch.softmax(num_rel_scores, dim=-1)
        elif(relpred == "disjoint_loss_raw"):
            num_rel_scores = self.get_all_scores_joint(A,B, q_nums_nz, p_nums_nz)
            # gr, le, eq = self.greater(A, B), self.lesser(A,B), self.equal(A,B)
            # num_rel_scores = torch.stack((gr, le, eq), dim=-1)
            num_rel_scores = num_rel_scores/self.model_args.quant_relationpred_temp
            rel_pred = num_rel_scores
            # rel_pred = rel_pred/self.model_args.quant_relationpred_temp
            if(self.model_args.add_num_relation_pred_loss):
                num_relation_loss, labels, pairs = self.number_relation_loss(rel_pred, qnum_meta, pnum_meta, return_pairs=True)
                print("NUM REL LOSS:", num_relation_loss, flush=True)
                loss += num_relation_loss
            PRINT_RELPRED_DEBUG = False
            if(not self.model_args.score_nosoftmax):
                num_rel_scores = torch.softmax(num_rel_scores, dim=-1)
            relation_pred_scores = 1
        elif(relpred == "disjoint_loss_bert"):
            A, B = self.layers_V["all"](nums_q_reps), self.layers_V["all"](nums_p_reps)
            num_rel_scores = self.get_all_scores_joint(A,B, q_nums_nz, p_nums_nz)
            num_rel_scores = num_rel_scores/self.model_args.quant_relationpred_temp
            rel_pred = num_rel_scores
            if(self.model_args.add_num_relation_pred_loss):
                num_relation_loss, labels, pairs = self.number_relation_loss(rel_pred, qnum_meta, pnum_meta, return_pairs=True)
                print("NUM REL LOSS:", num_relation_loss, flush=True)
                loss += num_relation_loss
            PRINT_RELPRED_DEBUG = False
            if(not self.model_args.score_nosoftmax):
                num_rel_scores = torch.softmax(num_rel_scores, dim=-1)
            relation_pred_scores = 1
        elif(relpred == "disjoint_raw"):
            gr, le, eq = self.greater(A, B), self.lesser(A,B), self.equal(A,B)
            PRINT_RELPRED_DEBUG = False
            num_rel_scores = torch.softmax(torch.stack((gr, le, eq), dim=-1)/self.model_args.quant_numscoring_temp, dim=-1)
            relation_pred_scores = 1
        else:
            raise NotImplementedError
            
        scores = relation_pred_scores*num_rel_scores
        
        pairwise_prob, prob_loss, gold_pairs = self.get_pairwise_probs(qnum_ctx, pnum_ctx, q_nums_nz, p_nums_nz, qnum_meta, pnum_meta)
        loss += prob_loss
        
        op, relation_classify_loss, op_gold = self.get_op(q_reps=q_reps, p_reps=p_reps, qnum_meta=qnum_meta, pnum_meta=pnum_meta)
        print("QP REL LOSS:", relation_classify_loss, flush=True)
        loss += relation_classify_loss
        if(not self.model_args.DEBUG_no_relpred and PRINT_RELPRED_DEBUG):
            print(
                "labels", labels[:3, 0, :3] if labels is not None else None, "\n",
                "RELPRED:",relation_pred_scores[:3, 0, :3] if not isinstance(relation_pred_scores, int) else None, "\n",
                "Pairs:", pairs[:3, 0, :3] if pairs is not None else None, "\n",
                flush=True
            )
        
        print(
            "NUM REL:",num_rel_scores.shape, num_rel_scores[:3, 0, :3], "\n",
            "PairGold:", gold_pairs[:3, 0, :3] if gold_pairs is not None else None, "\n",
            "PairPROBS:", pairwise_prob[:3, 0, :3], "\n",
            "SCOR",  scores[:3, 0, :3], "\n",
            "OP:", op[:3, :3], "\n",
            "OP_GOLD:", op_gold[:3, :3], "\n",
            flush=True
        )
        
         # B, 1, 3 (FIXME: change to B, N1, 3), that is diff op for every quant in query)
        op = op.unsqueeze(1)
        # B, N1, N2
        scores = torch.sum(op.unsqueeze(2)*scores, dim=-1)
        #FIXME: check if p_nums_nz is multiplied corectly everywhere
        scores = scores*(p_nums_nz.float().unsqueeze(1))
        useMax = self.model_args.DEBUG_score_pairprob_max
        if(useMax):
            bs, n1, _ = pairwise_prob.shape
            max_indices = torch.argmax(pairwise_prob, dim=-1) 
            scores = (pairwise_prob*scores)[torch.arange(bs).unsqueeze(1), torch.arange(n1).unsqueeze(0), max_indices]
        else: scores = torch.sum(pairwise_prob*scores, dim=-1)
        scores = scores*(q_nums_nz.float())
        
        # Query Quantity constraint severity
        # currently equal (its just one quant in the query)
        scores = torch.sum(scores, dim=-1)
            
        return scores, loss
    
    def get_op(self, q_reps, p_reps, qnum_meta, pnum_meta):
        # # B, 3
        # op = q_nums["op"].float()
        # # align with passages batch, FIXME: Is there an issue with alignment ? check encoder.py
        # op = torch.Tensor.repeat(op, (1, p_nums["numbers"].size(0))).view(-1, 3)
        mode = self.model_args.op_pred_mode
        op_gold = qnum_meta["op"]
        bs_p = pnum_meta["numbers"].size(0)
        op_gold = torch.Tensor.repeat(op_gold, (1,bs_p)).view(-1,3)
        if(mode == "pred"):
            op_logits, loss = self.relation_classifier_loss(q_reps, qnum_meta)
            op_prob = torch.softmax(op_logits/self.model_args.quant_combinescore_temp, dim=1)
            bs_p = pnum_meta["numbers"].size(0)
            op_prob = torch.Tensor.repeat(op_prob, (1,bs_p)).view(-1,3)
        if(mode == "pred_noloss"):
            op_logits, loss = self.relation_classifier_loss(q_reps, qnum_meta)
            op_prob = torch.softmax(op_logits/self.model_args.quant_combinescore_temp, dim=1)
            bs_p = pnum_meta["numbers"].size(0)
            op_prob = torch.Tensor.repeat(op_prob, (1,bs_p)).view(-1,3)
            loss = 0
        elif(mode == "raw"):
            op_prob = op_gold
            loss = 0 
            
        return op_prob, loss, op_gold

    def relation_classifier_loss(self, q_reps, q_nums):
        op = q_nums["op"].float()
        q_cls = q_reps[:,0]
        preds = self.relation_classifier(q_cls)
        # preds = preds/0.02
        loss = self.cross_entropy(preds, op)
        return preds, loss
    
    def number_prediction_loss(self, q_reps, p_reps, qnum_meta, pnum_meta, q_hidden, p_hidden):
        if not self.model_args.num_pred_loss_layerwise:
            q_num_hidden = self.numreps(qnum_meta["number_mask"], q_hidden)
            p_num_hidden = self.numreps(pnum_meta["number_mask"], p_hidden)
            qnum_loss, qexp_logits = self.numberpredicitonloss(q_num_hidden, qnum_meta["numbers"], qnum_meta["number_mask"])
            pnum_loss, _ = self.numberpredicitonloss(p_num_hidden, pnum_meta["numbers"], pnum_meta["number_mask"])
            return qnum_loss + pnum_loss, qexp_logits
        
        q_num_hidden = self.numreps(qnum_meta["number_mask"], q_hidden)
        p_num_hidden = self.numreps(pnum_meta["number_mask"], p_hidden)
        qnum_loss, qexp_logits = self.numberpredicitonloss(q_num_hidden, qnum_meta["numbers"], qnum_meta["number_mask"])
        pnum_loss, _ = self.numberpredicitonloss(p_num_hidden, pnum_meta["numbers"], pnum_meta["number_mask"])
        return qnum_loss + pnum_loss, qexp_logits
           
    '''
    Prompt: I have a tensor queryNums(batch, N1) and docNums(batch, N2), which denote numbers present in querry/docs.
    There can be maximum of N1, N2 nums respectively, and are padded with 0s.
    I want to construct a label matrix of shape batch, N1,N2 with labels for greater/lesser/equal.
    Give torch code.
    -----
    Now i have another matrix scores(batch, N1, N2, 3), where 3 denotes scores for gr/le/eq.
    I want make a classificaiton task with scores and labels above. Give torch code.
    Again, make sure padding values are handled properly.
    
    '''
    def number_relation_loss(self, scores, qnum_meta, pnum_meta, return_pairs=False):
        qnum, pnum = qnum_meta["numbers"], pnum_meta["numbers"]
        qnum_exp = torch.Tensor.repeat(qnum, (1, pnum.size(0))).view(-1, qnum.size(1))
        pnum_exp = torch.Tensor.repeat(pnum, (qnum.size(0), 1))
        # qnum_exp = qnum_exp.unsqueeze(-1)  # (batch, N1, 1)
        # pnum_exp = pnum_exp.unsqueeze(1)  # (batch, 1, N2)
        def construct_label_matrix(queryNums, docNums, padding_value=0, return_pairs=False):
            batch, N1 = queryNums.shape
            _, N2 = docNums.shape
            pairs = None
            queryNums_expanded = queryNums.unsqueeze(-1).expand(batch, N1, N2)
            docNums_expanded = docNums.unsqueeze(1).expand(batch, N1, N2)
            if(return_pairs):
                pairs = torch.stack((queryNums_expanded, docNums_expanded), dim=-1) 

            label_matrix = torch.full((batch, N1, N2), -1, dtype=torch.int64)

            label_matrix[queryNums_expanded < docNums_expanded] = 0  # Greater, YES, this is correct
            label_matrix[queryNums_expanded > docNums_expanded] = 1  # Lesser
            label_matrix[queryNums_expanded == docNums_expanded] = 2  # Equal

            query_padding_mask = queryNums.unsqueeze(-1) == padding_value
            doc_padding_mask = docNums.unsqueeze(1) == padding_value
            padding_mask = query_padding_mask | doc_padding_mask
            label_matrix[padding_mask] = -1
            return label_matrix.to(queryNums.device), pairs
        
        def classification_task(scores, label_matrix, padding_value=-1):
            valid_mask = label_matrix != padding_value
            
            # Flatten tensors to make them compatible with the loss function
            scores_flat = scores.view(-1, 3)  # Shape: (batch * N1 * N2, 3)
            labels_flat = label_matrix.view(-1)  # Shape: (batch * N1 * N2)
            valid_mask_flat = valid_mask.view(-1)  # Shape: (batch * N1 * N2)

            # Filter only valid entries
            scores_valid = scores_flat[valid_mask_flat]  # Shape: (num_valid_entries, 3)
            labels_valid = labels_flat[valid_mask_flat]  # Shape: (num_valid_entries,)

            # Compute cross-entropy loss
            loss = F.cross_entropy(scores_valid, labels_valid)
            return loss
        
        labels, pairs = construct_label_matrix(qnum_exp, pnum_exp, return_pairs=return_pairs)
        loss = classification_task(scores, labels)
        # labels = torch.zeros_like(qnum_exp.expand(-1, -1, pnum_exp.size(-1)))
        # labels[qnum_exp > pnum_exp] = 0  # Greater
        # labels[qnum_exp < pnum_exp] = 1  # Lesser
        # labels[qnum_exp == pnum_exp] = 2  # Equal
        # loss = self.cross_entropy(scores.view(-1, 3), labels.long().view(-1))
        
        return loss, labels, pairs

    def forward(self, q_reps, p_reps, qnum_meta, pnum_meta, 
                q_hidden, p_hidden, calc_loss, q_meta, p_meta,
                query=None, doc=None):
        '''
        nums -> B, N, D
        p torch.Size([32, 32, 128]) 
        p_nums torch.Size([256, 128, 128])
        '''
        loss = 0
        num_q_reps, num_p_reps = q_reps, p_reps
        if self.model_args.num_reps_from_hidden:
            num_q_reps, num_p_reps = q_hidden, p_hidden
        num_q_reps = self.numreps(qnum_meta["number_mask"], num_q_reps)
        num_p_reps = self.numreps(pnum_meta["number_mask"], num_p_reps)
        
        if(self.model_args.add_num_pred_loss): 
            numpred_loss, qexp_logits = self.number_prediction_loss(q_reps, p_reps, qnum_meta, pnum_meta, q_hidden, p_hidden)
            numpred_loss = self.model_args.num_pred_loss_alpha*numpred_loss
            print("NUM_PRED:", numpred_loss, flush=True)
            loss += numpred_loss
        bs_q, k1, d = num_q_reps.size(0), num_q_reps.size(1), num_q_reps.size(2)
        bs_p, k2, d = num_p_reps.size(0), num_p_reps.size(1), num_p_reps.size(2)
        
        num_q_reps = torch.Tensor.repeat(num_q_reps, (1,bs_p,1)).view(-1,k1, d)
        num_p_reps = torch.Tensor.repeat(num_p_reps, (bs_q,1,1))
        num_q_ctx_reps = num_q_reps
        num_p_ctx_reps= num_p_reps
        
        q_node_exp, p_node_exp = None, None
        if(self.nummask_expand):
            q_node_exp = self.numreps(qnum_meta["number_mask"], q_reps, self.nummask_expand)
            p_node_exp = self.numreps(pnum_meta["number_mask"], p_reps, self.nummask_expand)
            q_node_exp = torch.Tensor.repeat(q_node_exp, (1,bs_p,1)).view(-1,k1, d)
            p_node_exp = torch.Tensor.repeat(p_node_exp, (bs_q,1,1))
        if(self.model_args.num_get_ctx):
            if(self.model_args.num_get_unit_ctx): raise
            num_q_ctx_reps = self.numreps(qnum_meta["number_mask"], q_reps, self.nummask_expand, get_context=True)
            num_p_ctx_reps = self.numreps(pnum_meta["number_mask"], p_reps, self.nummask_expand, get_context=True)
            num_q_ctx_reps = torch.Tensor.repeat(num_q_ctx_reps, (1,bs_p,1)).view(-1,k1, d)
            num_p_ctx_reps = torch.Tensor.repeat(num_p_ctx_reps, (bs_q,1,1))
        elif(self.model_args.num_get_unit_ctx):
            num_q_ctx_reps = self.numreps(qnum_meta["unit_mask"], q_reps, self.nummask_expand, get_context=False)
            num_p_ctx_reps = self.numreps(pnum_meta["unit_mask"], p_reps, self.nummask_expand, get_context=False)
            num_q_ctx_reps = torch.Tensor.repeat(num_q_ctx_reps, (1,bs_p,1)).view(-1,k1, d)
            num_p_ctx_reps = torch.Tensor.repeat(num_p_ctx_reps, (bs_q,1,1))
            
            qid = qnum_meta["id"].unsqueeze(-1)
            qid = torch.Tensor.repeat(qid, (1,bs_p)).view(-1, 1)
            pid = pnum_meta["id"].unsqueeze(-1)
            pid = torch.Tensor.repeat(pid, (bs_q,1))
            
        q_nums_nz = torch.Tensor.repeat(qnum_meta["numbers"], (1,bs_p)).view(-1,k1) != 0
        p_nums_nz = torch.Tensor.repeat(pnum_meta["numbers"], (bs_q,1)) != 0
        
        q_cls = q_reps[:,0]
        p_cls = p_reps[:,0]
        q_cls = torch.Tensor.repeat(q_cls, (1,bs_p)).view(-1, q_reps.size(-1))
        p_cls = torch.Tensor.repeat(p_cls, (bs_q,1))
        func = None
        
        scores, relation_classify_loss = self.get_relevance(num_q_reps, num_p_reps,q_node_exp, 
                                            p_node_exp, q_nums_nz, p_nums_nz, 
                                            qnum_meta, pnum_meta, q_reps, p_reps, num_q_ctx_reps=num_q_ctx_reps,
                                            num_p_ctx_reps=num_p_ctx_reps, query=query, doc=doc)
        scores = scores.view(bs_q,bs_p)
        
        if(calc_loss and self.ver == "asym_dot" and self.asymdot_loss is not None):
            loss_func = {"v1": self.loss, "v2": self.lossv2, "minproduct": self.loss_minproduct}
            loss += loss_func[self.asymdot_loss](num_q_reps, num_p_reps)
            
        if(func is not None):
            func = func.view(bs_q,bs_p)
            print("func", torch.unique(func, return_counts=True))
            
        if(calc_loss and self.add_relation_classifier_loss):
            loss += relation_classify_loss
            
        return scores, loss, {"qexp_logits": qexp_logits}
    