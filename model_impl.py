import math
import inspect
from dataclasses import dataclass
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig


class LayerNormImpl(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        assert bias is False
        self.ndim = ndim
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return (
            (input - input.mean(dim=-1, keepdim=True))
            / torch.sqrt(
                input.var(dim=-1, keepdim=True) * (self.ndim - 1) / self.ndim + 1e-5
            )
            * self.weight
        )


class MLPImpl(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class CausalSelfAttentionImpl(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        assert self.n_embd == C

        n_head = self.n_embd // self.head_size

        q, k, v = self.c_attn.weight.split(self.n_embd, dim=0)
        assert q.shape == k.shape, k.shape == v.shape
        assert q.shape == torch.Size([self.n_embd, self.n_embd])
        qs = []
        ks = []
        vs = []
        for i in range(n_head):
            qs.append(q[i * self.head_size : (i + 1) * self.head_size])
            ks.append(k[i * self.head_size : (i + 1) * self.head_size])
            vs.append(v[i * self.head_size : (i + 1) * self.head_size])

        mimic_linear = lambda l, x: x @ l.transpose(0, 1)
        scores = []
        for i in range(n_head):
            query = mimic_linear(qs[i], x)
            assert query.shape == (B, T, self.head_size)
            # key = self.ks[i][x]  # (B, T, self.head_size)
            key = mimic_linear(ks[i], x)  # (B, T, self.head_size)
            assert key.shape == (B, T, self.head_size)
            affinity = query @ key.transpose(1, 2)
            assert affinity.shape == (B, T, T), affinity.shape
            musked_affinity = affinity * self.bias * (1.0 / math.sqrt(self.head_size))
            musked_affinity = torch.where(
                torch.isclose(self.bias, torch.tensor(0.0)),
                float("-inf"),
                musked_affinity,
            )
            _score = F.softmax(musked_affinity, dim=-1)  # (B, T, T)
            scores.append(_score.reshape(B, 1, T, T))
        scores = torch.concat(scores, dim=1)
        assert scores.shape == (B, n_head, T, T)
        scores = self.attn_dropout(scores)

        score_per_head = [scores[:, i] for i in range(n_head)]
        assert score_per_head[0].shape == (B, T, T)

        value_per_head = [mimic_linear(vs[i], x) for i in range(n_head)]
        assert value_per_head[0].shape == (B, T, self.head_size)
        agg_val_per_head = [
            score_per_head[i] @ value_per_head[i] for i in range(n_head)
        ]
        assert agg_val_per_head[0].shape == (B, T, self.head_size)

        output = torch.concat(agg_val_per_head, dim=2)
        assert output.shape == (B, T, C)
        return self.resid_dropout(self.c_proj(output))


class BlockImpl(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNormImpl(config.n_embd, config.bias)
        self.attn = CausalSelfAttentionImpl(config)
        self.ln_2 = LayerNormImpl(config.n_embd, config.bias)
        self.mlp = MLPImpl(config)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        return self.mlp(self.ln_2(x)) + x


class GPTImpl(nn.Module):
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert [(pn, p) for pn, p in param_dict.items() if not p.requires_grad] == []
        print("configure_optimizers: successful assert")
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(BlockImpl(config) for _ in range(config.n_layer)),
                ln_f=LayerNormImpl(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        assert len(idx.shape) == 2
        B, T = idx.shape
        assert targets is None or targets.shape == (B, T)

        text_embd = self.transformer.wte(idx)
        assert text_embd.shape == (B, T, self.config.n_embd)
        pos_embd = self.transformer.wpe(torch.arange(0, T))
        assert pos_embd.shape == (T, self.config.n_embd)
        total_embd = text_embd + pos_embd

        x = self.transformer.drop(total_embd)
        for i in range(self.config.n_layer):
            x = self.transformer.h[i](x)

        logits = self.lm_head(self.transformer.ln_f(x))
        assert logits.shape == (B, T, self.config.vocab_size)
        return logits, F.cross_entropy(logits.transpose(1, 2), targets, )
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


