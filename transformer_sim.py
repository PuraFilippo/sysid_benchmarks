"""
Implementation of the Encoder-Decoder Transformer models for multi-step simulation of dynamical systems.

Partially based on:
* nanoGPT https://github.com/karpathy/nanoGPT/
* The Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/
"""

import math
from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F


@dataclass
class Config:
    seq_len_ctx: int = 10_000
    seq_len_new: int = 128
    seq_len_patch: int = 400
    d_model_RNN: int = 128
    # d_model_patching: int =d_model_RNN
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_u: int = 1
    n_y: int = 1
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device_name: str = "cuda:1"

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x, mem):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_head, dropout=dropout, causal=False, bias=bias) # encoder is never causal
        
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_heads,
                                       dropout=dropout, causal=True, bias=bias)
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads,
                                         dropout=dropout, causal=False, bias=bias)
        self.ln_3 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), mem)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # final layer normalization
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)

    def forward(self, x, mem):
        for block in self.blocks:
            x = block(x, mem)
        x = self.ln_f(x)  # final layer normalization
        return x

import time


class TSTransformerBenchmark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.src = None
        self.mem = None
        patch_len = config.seq_len_ctx//config.seq_len_patch
        # self.lin_patch = torch.nn.Linear(patch_len*(config.n_u+config.n_y), config.d_model_patching)
        self.before_enc_wte = torch.nn.Linear(patch_len * (config.n_u + config.n_y), config.d_model_RNN)
        self.RNN = nn.RNN(config.n_u + config.n_y, config.d_model_RNN, num_layers=1,
                          batch_first=True)  # , bidirectional=True)
        self.encoder = TransformerEncoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)
        self.decoder = TransformerDecoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)

        self.encoder_wte = nn.Linear(config.d_model_RNN, config.n_embd)
        self.encoder_wte2 = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.encoder_wte_noRNN = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.encoder_wpe = PositionalEncoding(config.n_embd)
        self.decoder_wte1 = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.decoder_wte2 = nn.Linear(config.n_u, config.n_embd)
        self.decoder_wpe = PositionalEncoding(config.n_embd)  # could also be the same as encoder_wpe

        self.lm_head_mean = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?
        self.lm_head_logvar = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?

    # def embed_ctx(self, y, u,u_new,y_new,n_in, n_init = 0):
    def embed_ctx(self, y, u):
        ## UPDATE
        # introducing the patching approaches, here described the linear and the recurrent one, to process
        # context longer than 400 time-stamps

        max_len_enc = 400
        start_time1 = time.time()
        if u.shape[1] > max_len_enc:
            start = u.shape[1] % max_len_enc
            u = u[:, start:, :]
            y = y[:, start:, :]
            [B, T, nu] = u.shape
            n_init = T % max_len_enc
            ny = y.shape[2]
            patch_len = (T - n_init) // max_len_enc

            yu_patch = torch.cat((y[:, n_init:, :], u[:, n_init:, :]), dim=-1)
            # IF NO SKIP
            # ---
            # yu_patch = yu_patch.view(B,max_len_enc, patch_len*(nu + ny)) ## if linear patch
            yu_patch = yu_patch.view(B * max_len_enc, patch_len, nu + ny)  # if RNN patch
            # yu = self.lin_patch(yu_patch)
            _, hn = self.RNN(yu_patch)

            d_model = hn.shape[2]
            yu = hn[-1:].view(B, max_len_enc, d_model)
            tok_emb = self.encoder_wte(yu)

        else:
            yu = torch.cat((y, u), dim=-1)
            tok_emb = self.encoder_wte_noRNN(yu)
            end_time1 = time.time()

        src = self.encoder_wpe(tok_emb)
        return src

    def embed_new(self, u_new, y_new, n_in):
        ##UPDATE
        ## definition of two linear layer, one for process the query input and one for the initial conditions
        size_dec = y_new.size()
        assert n_in < size_dec[
            1], "the number of initial condition has to be less then the elements that enter in the decoder"

        # change and use the matrix for linearity in the encoder
        yu_new = torch.cat((u_new[:, :n_in, :], y_new[:, :n_in, :]), dim=-1)
        tok_emb_new1 = self.decoder_wte1(yu_new)
        tok_emb_new2 = self.decoder_wte2(u_new[:, n_in:, :])
        tok_emb_new = torch.cat((tok_emb_new1, tok_emb_new2), dim=1)
        tgt = self.decoder_wpe(tok_emb_new)
        return tgt

    def forward(self, y, u, u_new, y_new, n_in):
        if self.src is None:
            self.src = self.embed_ctx(y, u)  # perhaps dropout of this?
        tgt = self.embed_new(u_new, y_new, n_in)  # perhaps dropout of this?
        if self.mem is None:
            self.mem = self.encoder(self.src)

        ##UPDATE
        # probabilistic prediction of the query output, without considering the initial condition in the loss

        output = self.decoder(tgt, self.mem)[:, n_in:, :]

        y_mean = self.lm_head_mean(output)
        y_logvar = self.lm_head_logvar(output)
        y_std = torch.exp(y_logvar / 2)
        loss = None
        if y_new is not None:  # compute loss
            batch_y_dist = torch.distributions.Normal(y_mean, y_std)
            nll = -batch_y_dist.log_prob(y_new[:, n_in:, :])
            loss = torch.mean(nll)
            rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(y_new[:, n_in:, :], y_mean))
        return y_mean, y_std, loss, rmse_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True  # 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class TSTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_len = config.seq_len_ctx//config.seq_len_patch
        # self.lin_patch = torch.nn.Linear(patch_len*(config.n_u+config.n_y), config.d_model_patching)
        self.before_enc_wte = torch.nn.Linear(patch_len * (config.n_u + config.n_y), config.d_model_RNN)
        self.RNN = nn.RNN(config.n_u + config.n_y, config.d_model_RNN, num_layers=1, batch_first=True)#, bidirectional=True)
        self.encoder = TransformerEncoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)
        self.decoder = TransformerDecoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)

        self.encoder_wte = nn.Linear(config.d_model_RNN, config.n_embd)
        self.encoder_wte2 = nn.Linear(config.n_u+config.n_y, config.n_embd)
        self.encoder_wte_noRNN = nn.Linear(config.n_u+config.n_y, config.n_embd)
        self.encoder_wpe = PositionalEncoding(config.n_embd)
        self.decoder_wte1 = nn.Linear(config.n_u + config.n_y, config.n_embd) 
        self.decoder_wte2 = nn.Linear(config.n_u, config.n_embd)
        self.decoder_wpe = PositionalEncoding(config.n_embd) # could also be the same as encoder_wpe

        self.lm_head_mean = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?
        self.lm_head_logvar = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?

    # def embed_ctx(self, y, u,u_new,y_new,n_in, n_init = 0):
    def embed_ctx(self, y, u):
        ## UPDATE
        # introducing the patching approaches, here described the linear and the recurrent one, to process 
        # context longer than 400 time-stamps

        max_len_enc = 400
        start_time1 = time.time()
        if u.shape[1]>max_len_enc:
            start = u.shape[1]%max_len_enc
            u = u[:,start:,:]
            y = y[:,start:,:]
            [B,T,nu] = u.shape
            n_init = T%max_len_enc
            ny = y.shape[2]
            patch_len = (T-n_init)//max_len_enc
            
            yu_patch = torch.cat((y[:,n_init:,:], u[:,n_init:,:]), dim=-1)
            #IF NO SKIP
            #---
            # yu_patch = yu_patch.view(B,max_len_enc, patch_len*(nu + ny)) ## if linear patch
            yu_patch = yu_patch.view(B*max_len_enc, patch_len, nu + ny) # if RNN patch
            #yu = self.lin_patch(yu_patch)
            _, hn = self.RNN(yu_patch)
            
            d_model = hn.shape[2]
            yu = hn[-1:].view(B, max_len_enc, d_model)
            tok_emb = self.encoder_wte(yu)
            
        else:
            yu = torch.cat((y, u), dim=-1)
            tok_emb = self.encoder_wte_noRNN(yu)
            end_time1  = time.time()
            
        src = self.encoder_wpe(tok_emb)
        return src

    def embed_new(self, u_new,y_new,n_in):
        ##UPDATE
        ## definition of two linear layer, one for process the query input and one for the initial conditions
        size_dec  = y_new.size()
        assert n_in<size_dec[1], "the number of initial condition has to be less then the elements that enter in the decoder"

        # change and use the matrix for linearity in the encoder
        yu_new = torch.cat((u_new[:,:n_in,:],y_new[:,:n_in,:]), dim=-1)
        tok_emb_new1 = self.decoder_wte1(yu_new)
        tok_emb_new2 = self.decoder_wte2(u_new[:,n_in:,:])
        tok_emb_new = torch.cat((tok_emb_new1,tok_emb_new2), dim=1)
        tgt = self.decoder_wpe(tok_emb_new)
        return tgt

    def forward(self, y, u, u_new, y_new,n_in):
        src = self.embed_ctx(y, u)  # perhaps dropout of this?
        tgt = self.embed_new(u_new, y_new,n_in)  # perhaps dropout of this?
        mem = self.encoder(src)

        ##UPDATE
        # probabilistic prediction of the query output, without considering the initial condition in the loss

        output = self.decoder(tgt, mem)[:,n_in:,:]

        y_mean = self.lm_head_mean(output)
        y_logvar = self.lm_head_logvar(output)
        y_std = torch.exp(y_logvar/2)
        loss = None
        if y_new is not None: # compute loss
            batch_y_dist = torch.distributions.Normal(y_mean, y_std)
            nll = -batch_y_dist.log_prob(y_new[:,n_in:,:])
            loss = torch.mean(nll)
            rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(y_new[:,n_in:,:], y_mean))
        return y_mean, y_std,loss , rmse_loss
    

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
