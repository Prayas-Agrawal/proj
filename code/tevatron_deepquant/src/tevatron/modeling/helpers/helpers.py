import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tevatron.arguments import ModelArguments

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

def cosine_annealing_scheduler(initial_temp, final_temp, num_steps, current_step):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / num_steps))
    return final_temp + (initial_temp - final_temp) * cosine_decay

def exponential_decay_scheduler(initial_temp, final_temp, num_steps, current_step):
    decay_rate = (final_temp / initial_temp) ** (1 / num_steps)
    return initial_temp * (decay_rate ** current_step)


class SoftmaxTemperatureScheduler:
    def __init__(self, initial_temp=1.0, final_temp=0.1, total_steps=1000, strategy="linear", inference=False):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.strategy = strategy
        self.current_step = 0
        self.inference = inference

    def get(self):
        if self.inference:
            return self.final_temp
        if self.strategy == "linear":
            t = self.current_step / self.total_steps
            return self.initial_temp + t * (self.final_temp - self.initial_temp)
        elif self.strategy == "exponential":
            return self.initial_temp * (self.final_temp / self.initial_temp) ** (self.current_step / self.total_steps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def step(self):
        if(self.inference): return
        if self.current_step < self.total_steps and not self.inference:
            self.current_step += 1

    def reset(self):
        self.current_step = 0

NUM_RANGE = (-10, 20)
def scientific_notation(numbers):
    numbers = torch.clamp(numbers, min=1e-10, max=1e20)
    mask = numbers != 0
    # Avoid log(0) by adding a small epsilon where mask is True
    safe_numbers = torch.where(mask, numbers, torch.ones_like(numbers))  # Replace 0s with 1s for safe computation
    exponents = torch.floor(torch.log10(torch.abs(safe_numbers) + 1e-10))  # Compute exponents
    mantissas = numbers / (10 ** exponents)  # Compute mantissas
    exponents = torch.where(mask, exponents, torch.zeros_like(exponents))  # Set exponents to 0 for masked positions
    mantissas = torch.where(mask, mantissas, torch.zeros_like(mantissas))  # Set mantissas to 0 for masked positions
    return mantissas, exponents.to(torch.int64)  # Return as integers
    
class NumReps(nn.Module):
    def __init__(self, node_dim, model_args:ModelArguments):
        super(NumReps, self).__init__()
        self.aggr_mode = model_args.deepquant_aggr_mode
        # self.lstm = torch.nn.LSTM(node_dim, node_dim, num_layers=2, batch_first=True, bias=False)
        self.gru = torch.nn.GRU(node_dim, node_dim, num_layers=2, batch_first=True, bias=False)
        
        
    def get_context_mask(self, number_mask, K, include_number=False):
        batch, max_nums, seq_len = number_mask.shape

        seq_range = torch.arange(seq_len, device=number_mask.device)  # Shape: (seq_len,)

        distances = torch.abs(seq_range.unsqueeze(0) - seq_range.view(-1, 1))  # Shape: (seq_len, seq_len)

        context_window = (distances <= K).to(number_mask.device)  # Shape: (seq_len, seq_len)

        context_mask = torch.einsum("bns,ss->bns", number_mask, context_window)  # Shape: (batch, max_nums, seq_len)

        if include_number:
            context_mask = torch.maximum(context_mask, number_mask)  # Ensure number positions are included

        return context_mask
        
    def expand_window_vectorized(self, A, percentage):
        device = torch.get_device(A)
        
        B = A.clone()  
        B = B.to(device)
        B_size, N, M = A.shape  # B: batch size, N: number of rows, M: number of columns
        
        row_has_ones = A.sum(dim=2) > 0  # Shape: (B, N) -> True for rows with at least one 1
        
        starts = torch.full((B_size, N), -1, dtype=torch.long).to(device)
        ends = torch.full((B_size, N), -1, dtype=torch.long).to(device)

        diffs = A.diff(dim=2)

        ones_starts = (diffs == 1).nonzero(as_tuple=False)
        ones_ends = (diffs == -1).nonzero(as_tuple=False)
        
        starts[ones_starts[:, 0], ones_starts[:, 1]] = ones_starts[:, 2] + 1
        ends[ones_ends[:, 0], ones_ends[:, 1]] = ones_ends[:, 2]

        first_ones = torch.argmax(A, dim=2)
        last_ones = M - 1 - torch.argmax(A.flip(dims=[2]), dim=2)

        starts = torch.where(row_has_ones, first_ones, starts)
        ends = torch.where(row_has_ones, last_ones, ends)

        window_size = (ends - starts + 1).float()
        expand_size = (window_size * (percentage)).int()

        new_starts = torch.clamp(starts - (expand_size), min=0)
        new_ends = torch.clamp(ends + (expand_size), max=M - 1)

        for b in range(B_size):
            for row in range(N):
                if row_has_ones[b, row]:  # Only expand rows that have 1s
                    B[b, row, new_starts[b, row]:new_ends[b, row] + 1] = 1

        return B
        
    def forward(self, number_mask, reps, expand_percentage = 0, get_context=False):
        '''
            mask - B, max_nums, N
            
            reps - B, N, D
        '''
        mask = number_mask
        
        mode = self.aggr_mode
        if(get_context):
            mask = self.get_context_mask(number_mask, K=5)
            mode = "mean"
        if(expand_percentage):
            mask = self.expand_window_vectorized(number_mask, expand_percentage)
        if(mode == "mean"):
            # print("shape", number_mask.shape, reps.shape)
            #FIXME
            fl_mask = mask.float()
            num_ones = fl_mask.sum(dim=-1, keepdim=True)
            norm_mask = fl_mask / num_ones.clamp(min=1)
            # norm_mask = mask.float()
            return torch.matmul(norm_mask, reps )
        
        if(mode == "token"):
            fl_mask = mask.float()
            return torch.matmul(fl_mask, reps )
        
        if(mode == "gru_l2r"):
            b, m, n, d = mask.size(0), mask.size(1), mask.size(2), reps.size(2)
            num_nored = torch.einsum("bmn,bnd->bmnd", mask, reps)
            num_nored = num_nored.view(b*m, n, -1)
            h0 = torch.zeros(2,b*m,d).to("cuda")
            num_red = self.gru(num_nored, h0)[0][:,-1,:]
            num_red = num_red.view(b,m,-1)
            return num_red
        
        if(mode == "gru_r2l"):
            b, m, n, d = mask.size(0), mask.size(1), mask.size(2), reps.size(2)
            num_nored = torch.einsum("bmn,bnd->bmnd", mask, reps)
            num_nored = num_nored.view(b*m, n, -1)
            num_nored = torch.flip(num_nored, dims=[1])
            h0 = torch.zeros(2,b*m,d).to("cuda")
            num_red = self.gru(num_nored, h0)[0][:,-1,:]
            num_red = num_red.view(b,m,-1)
            return num_red
            
        else:
            raise NotImplementedError
    
class NumberPredictionLoss(nn.Module):
    def __init__(self, frac_exp, output_dim, exp_range=NUM_RANGE):
        super(NumberPredictionLoss, self).__init__()
        self.exp_min, self.exp_max = exp_range
        self.exp_max_range = self.exp_max - self.exp_min + 1
        
        self.dim_exp = int(frac_exp*output_dim)
        self.dim_man = output_dim - self.dim_exp

        self.exp_predictor = nn.Linear(self.dim_exp, self.exp_max_range)
        self.mantissa_predictor = nn.Linear(self.dim_man, 1)
        self.exp_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.mantissa_loss_fn = nn.MSELoss(reduction="none")
        
    def forward(self, num_reps, numbers, mask):
        nonzero_mask = numbers != 0
        exp_reps, mantissa_reps = torch.split(num_reps, [self.dim_exp, self.dim_man], dim=-1)
        mantissa, exponents = scientific_notation(numbers)  # Decompose numbers

        exp_logits = self.exp_predictor(exp_reps)  # Shape: (batch, num_slots, exp_max_range)
        exp_labels = (exponents - self.exp_min).long()  # Shift exponent range to start from 0
        exp_loss = self.exp_loss_fn(exp_logits.view(-1, self.exp_max_range), exp_labels.view(-1))
        exp_loss = exp_loss.view_as(numbers)
        exp_loss = exp_loss * (nonzero_mask)  # Mask out invalid slots
        

        mantissas_logits = self.mantissa_predictor(mantissa_reps).squeeze(-1)  # Shape: (batch, num_slots)
        mantissa_loss = self.mantissa_loss_fn(mantissas_logits, mantissa) * nonzero_mask  # Mask out invalid slots

        total_loss = (exp_loss.sum() + mantissa_loss.sum()) / (nonzero_mask.float().sum() + 1e-6)

        return total_loss, exp_logits

    # @staticmethod
    # def decompose_to_scientific(numbers):
        
    #     exponents = torch.floor(torch.log10(torch.abs(numbers) + 1e-8))  # Exponents
    #     mantissas = numbers / (10 ** exponents)  # Mantissas
    #     exponents = exponents.clamp(min=-10, max=10)  # Clamp exponents to valid range
    #     return mantissas, exponents
    
class NumGPTEmbed(nn.Module):
    def __init__(self, frac_exp, output_dim, exp_range=NUM_RANGE):
        super(NumGPTEmbed, self).__init__()
        num_exponents = exp_range[1] - exp_range[0] + 1  # Total unique exponents
        self.exp_range = exp_range
        self.dim_exp = int(frac_exp*output_dim)
        self.dim_man = output_dim - self.dim_exp
        
        self.exponent_embedding = nn.Embedding(num_exponents, self.dim_exp)
        
    def mantissa_embedding(self, mantissas):
        factor = (10-(-10))/(self.dim_man-1)
        proto = torch.arange(0, self.dim_man, device="cuda", dtype=torch.float)*factor - 10
        
        # _max = self.exp_range[1]
        # _min = self.exp_range[0]
        # factor = (_max-_min)/(self.dim_man-1)
        # proto = torch.arange(0, self.dim_man, device="cuda", dtype=torch.float)*factor - _max
        
        mantissas = mantissas.unsqueeze(-1)
        diff = mantissas - proto
        embs = torch.exp(-(diff)**2)
        return embs
        
    def forward(self, numbers):
        mask = numbers != 0  # Shape: (batch, max_nums)
        mantissas, exponents = scientific_notation(numbers)
        
        mantissa_embeds = self.mantissa_embedding(mantissas)
        
        exponents = exponents - self.exp_range[0]  # Shift to start at 0
        
        exponent_embeds = self.exponent_embedding(exponents)
        
        number_embeds = torch.cat([exponent_embeds, mantissa_embeds], dim=-1)  # (batch_size, dim_man + dim_exp)
        number_embeds = number_embeds * mask.unsqueeze(-1)  # Broadcast mask to (batch, max_nums, dim)
        return number_embeds
    

def project_to_poincare_ball(embeddings, c=1.0):
    norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    max_norm = (1 - 1e-5) / (c ** 0.5)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return scale * embeddings

def poincare_asymmetric_score(q, d, transform, c=1.0, epsilon=1e-6):
    # q = q / torch.max(torch.norm(q, p=2, dim=-1, keepdim=True))
    # d = d / torch.max(torch.norm(d, p=2, dim=-1, keepdim=True))

    q_hyp = project_to_poincare_ball(q, c)
    d_transformed = transform(d) # Shape: [batch, seq_length, dim]
    d_transformed = project_to_poincare_ball(d_transformed, c)
    
    # print(torch.max(torch.norm(q_hyp, p=2, dim=-1)))  # Should be < 1
    # print(torch.max(torch.norm(d_hyp, p=2, dim=-1)))  # Should be < 1
    # Transform document embeddings with the asymmetric matrix A
    
    # Compute pairwise norms for queries and transformed documents
    norm_q = torch.norm(q_hyp, p=2, dim=2)  # Shape: [batch, seq_length, 1]
    norm_d = torch.norm(d_transformed, p=2, dim=2)  # Shape: [batch, seq_length, 1]
    
    # Expand norms for pairwise computation
    norm_q_expanded = norm_q.unsqueeze(2)  # Shape: [batch, seq_length, seq_length]
    norm_d_expanded = norm_d.unsqueeze(1)  # Shape: [batch, seq_length, seq_length]
    
    # Compute denominator with stability
    denominator = (1 - c * norm_q_expanded**2) * (1 - c * norm_d_expanded**2) + epsilon
    
    # Expand query and document embeddings for pairwise computation
    q_expanded = q_hyp.unsqueeze(2)  # Shape: [batch, seq_length, 1, dim]
    d_expanded = d_transformed.unsqueeze(1)  # Shape: [batch, 1, seq_length, dim]
    
    # Compute pairwise squared L2 distances between q and transformed d
    pairwise_diff = q_expanded - d_expanded  # Shape: [batch, seq_length, seq_length, dim]
    numerator = torch.sum(pairwise_diff ** 2, dim=-1)  # Shape: [batch, seq_length, seq_length]
    
    # Compute hyperbolic distance
    hyperbolic_dist = torch.log(1 + 2 * c * numerator / denominator + epsilon) / (c ** 0.5)  # Shape: [batch, seq_length, seq_length]
    # Return negative distance as the score
    # print("hype", hyperbolic_dist)
    return -hyperbolic_dist



class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)
    


class PairwiseScores_Cross(nn.Module):
    def __init__(self, dim, num_layers=1, out_dim=3, ):
        super(PairwiseScores_Cross, self).__init__()
        self.ffl = nn.Sequential(
            nn.Linear(2 * dim, 128),  # Hidden layer size (adjustable)
            nn.ReLU(),
            nn.Linear(128, out_dim)        # Output layer with 3 scores (greater, lesser, equal)
        )
        if(num_layers > 1):
           self.ffl = nn.Sequential(
                nn.Linear(2 * dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim)
            ) 
 
    def forward(self, query, doc, query_mask, doc_mask):
        """
        Args:
            query: Tensor of shape (batch, n1, dim), padded with zeros for invalid numbers
            doc: Tensor of shape (batch, n2, dim), padded with zeros for invalid numbers

        Returns:
            scores: Tensor of shape (batch, n1, n2, 3) with scores for valid pairs
        """
        batch, n1, dim = query.shape
        _, n2, _ = doc.shape


        # Expand and tile query and doc embeddings for pairwise comparison
        query_exp = query.unsqueeze(2).expand(-1, n1, n2, -1)  # (batch, n1, n2, dim)
        doc_exp = doc.unsqueeze(1).expand(-1, n1, n2, -1)      # (batch, n1, n2, dim)

        # Concatenate query and doc embeddings for each pair
        pair_emb = torch.cat([query_exp, doc_exp], dim=-1)  # (batch, n1, n2, 2 * dim)

        # Create a mask for valid pairs
        pair_mask = (query_mask.unsqueeze(2) & doc_mask.unsqueeze(1))  # (batch, n1, n2)

        # Flatten pair embeddings and mask to process only valid pairs
        pair_emb_flat = pair_emb.view(batch * n1 * n2, -1)  # (batch * n1 * n2, 2 * dim)
        pair_mask_flat = pair_mask.view(-1)  # (batch * n1 * n2)

        # Filter valid pairs
        valid_pair_emb = pair_emb_flat[pair_mask_flat]  # (num_valid_pairs, 2 * dim)

        # Compute scores for valid pairs using the feedforward layer
        valid_scores = self.ffl(valid_pair_emb)  # (num_valid_pairs, 3)

        # Create an output tensor for all pairs, initialize with zeros
        scores = torch.zeros(batch * n1 * n2, 3, device=query.device, dtype=valid_scores.dtype)  # (batch * n1 * n2, 3)

        # Populate scores for valid pairs
        scores[pair_mask_flat] = valid_scores

        # Reshape to the original pairwise shape
        scores = scores.view(batch, n1, n2, 3)  # (batch, n1, n2, 3)

        return scores

class PairwiseScores_multipred(nn.Module):
    def __init__(self, dim, num_layers=1, out_dim1=3,out_dim2=3 ):
        super(PairwiseScores_multipred, self).__init__()
        self.ffl = nn.Sequential(
            nn.Linear(2 * dim, 128),  # Hidden layer size (adjustable)
            nn.ReLU(),
        )
        if(num_layers > 1):
           self.ffl = nn.Sequential(
                nn.Linear(2 * dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            ) 
        self.outdim1 = out_dim1
        self.outdim2 = out_dim2
        self.pred1= nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim1)
        )
        self.pred2= nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim2)
        )
        
    def forward(self, query, doc, query_mask, doc_mask):
        batch, n1, dim = query.shape
        _, n2, _ = doc.shape


        # Expand and tile query and doc embeddings for pairwise comparison
        query_exp = query.unsqueeze(2).expand(-1, n1, n2, -1)  # (batch, n1, n2, dim)
        doc_exp = doc.unsqueeze(1).expand(-1, n1, n2, -1)      # (batch, n1, n2, dim)

        pair_emb = torch.cat([query_exp, doc_exp], dim=-1)  # (batch, n1, n2, 2 * dim)

        pair_mask = (query_mask.unsqueeze(2) & doc_mask.unsqueeze(1))  # (batch, n1, n2)

        pair_emb_flat = pair_emb.view(batch * n1 * n2, -1)  # (batch * n1 * n2, 2 * dim)
        pair_mask_flat = pair_mask.view(-1)  # (batch * n1 * n2)

        valid_pair_emb = pair_emb_flat[pair_mask_flat]  # (num_valid_pairs, 2 * dim)

        encoded = self.ffl(valid_pair_emb)
        
        valid_scores1 = self.pred1(encoded)
        scores1 = torch.zeros(batch * n1 * n2, self.outdim1, 
                              device=query.device, dtype=valid_scores1.dtype)
        scores1[pair_mask_flat] = valid_scores1
        scores1 = scores1.view(batch, n1, n2, self.outdim1)
        
        valid_scores2 = self.pred2(encoded)
        scores2 = torch.zeros(batch * n1 * n2, self.outdim2, 
                              device=query.device, dtype=valid_scores2.dtype)
        scores2[pair_mask_flat] = valid_scores2
        scores2 = scores2.view(batch, n1, n2, self.outdim2)

        return scores1, scores2
    

class PairwiseScores_multipred2(nn.Module):
    def __init__(self, dim, num_layers=1, out_dim1=3,out_dim2=3 ):
        super(PairwiseScores_multipred2, self).__init__()
        self.network = PairwiseScores_multipred(dim, num_layers, 1, 1)
        
    def forward(self, query, doc, query_mask, doc_mask):
        scores1 = self.network(query, doc, query_mask, doc_mask)
        scores2 = self.network(doc, query, doc_mask, query_mask)
        gr = 1/(1 + torch.exp(scores1-scores2))
        le = 1/(1 + torch.exp(scores2-scores1))
        eq = torch.exp(-torch.abs(gr-le))
        return torch.cat([gr,le, eq], dim=-1)
        
class PairwiseScores_Cls(nn.Module):
    def __init__(self, dim, num_layers=1, out_dim=3):
        super(PairwiseScores_Cls, self).__init__()
        self.ffl = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        if(num_layers > 1):
           self.ffl = nn.Sequential(
                nn.Linear(2 * dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim)
            ) 
 
    def forward(self, query, doc):
        """
        Args:
            query: Tensor of shape (batch1, dim)
            doc: Tensor of shape (batch2,dim)

        Returns:
            scores: Tensor of shape (batch1, batch2)
        """
        bs_q, bs_p = query.size(0), doc.size(0)

        query_exp = torch.Tensor.repeat(query, (1, bs_p)).view(-1, query.size(1))
        doc_exp = torch.Tensor.repeat(doc, (bs_q, 1))

        pair_emb = torch.cat([query_exp, doc_exp], dim=-1)  # (batch, 2 * dim)

        scores = self.ffl(pair_emb).squeeze(-1)  # (batch)

        scores = scores.view(bs_q, bs_p)

        return scores
    