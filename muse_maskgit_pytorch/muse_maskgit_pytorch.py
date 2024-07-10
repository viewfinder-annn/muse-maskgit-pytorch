import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from muse_maskgit_pytorch.attend import Attend

from tqdm.auto import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8,
        flash = True,
        dropout = 0.
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            scale = scale
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h = h, i = n)
            context_mask = F.pad(context_mask, (1, 0), value = True)

        out = self.attend(q, k, v, mask = context_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True, flash = flash),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context = None, context_mask = None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context = context, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)

class SelfTransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# transformer - it's all we need

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out = None,
        t5_name = DEFAULT_T5_NAME,
        self_cond = False,
        add_mask_id = False,
        vq_layers = 1,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        # self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = SelfTransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.vq_layers = vq_layers
        self.to_logits = nn.ModuleList([nn.Linear(dim, self.dim_out, bias=False) for _ in range(vq_layers)])

        # text conditioning
        # self.encode_text = partial(t5_encode_text, name = t5_name)
        # text_embed_dim = get_encoded_dim(t5_name)
        # self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)
        pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        self_cond_embed = None,
        cond_drop_prob = 0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        # new x: embedded input, shape: [b, n, dim]
        device, b, n = x.device, x.shape[0], x.shape[1]
        assert n <= self.seq_len

        # classifier free guidance

        # if cond_drop_prob > 0.:
        #     mask = prob_mask_like((b, 1), 1. - cond_drop_prob, device) # 用1 - cond_drop_prob是想要prob_mask_like函数输出False的概率为cond_drop_prob
        #     context_mask = context_mask & mask

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim = -2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value = True)

        # embed tokens

        # x = self.token_emb(x)
        
        x = x + self.pos_emb(torch.arange(n, device = device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x)

        logits = torch.stack([linear(embed) for linear in self.to_logits], dim = 1) # [b, vq_layers, n, dim_out]
        # print("logits.shape: ", logits.shape)
        # print("labels.shape: ", labels.shape)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        # labels: [b, vq_layers*n]
        labels = labels.reshape(b, self.vq_layers, n)
        loss = self._compute_cross_entropy(logits, labels, ignore_index)

        if not return_logits:
            return loss

        return loss, logits

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index
    ):
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.
        
        Adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py#L212

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            # print(f"{k}: logits_k.shape: {logits_k.shape}, targets_k.shape: {targets_k.shape}")
            q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=ignore_index)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        # return ce, ce_per_codebook
        return ce

# self critic wrapper

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim_out, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        logits, embeds = self.net.forward_with_cond_scale(x, *args, return_embed = True, **kwargs)
        return self.to_pred(logits)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        logits, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed = True, **kwargs)
        return self.to_pred(logits)

    def forward(self, x, *args, labels = None, **kwargs):
        logits_net, embeds = self.net(x, *args, return_embed = True, **kwargs)
        logits = self.to_pred(logits_net) # [b, vq_layers, n, 1]

        if not exists(labels):
            return logits

        logits = rearrange(logits, '... 1 -> ...') # [b, vq_layers, n]
        return F.binary_cross_entropy_with_logits(logits, labels)

# specialized transformers

class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id = True, **kwargs)

class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out = 1, **kwargs)

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# noise schedules

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

class AudioEncoder(nn.Module):
    def __init__(self, dim, input_dim, n_fft, hop_length, win_length, mlp_layers, mlp_activation=nn.ReLU(), transformer_layers=6):
        super().__init__()
        self.dim = dim
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.mlp_layers = mlp_layers

        # Define MLP
        self.input_dim = input_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_layers[0]),
            mlp_activation,
            nn.Linear(mlp_layers[0], mlp_layers[1])
        )

        # Define Transformer Blocks
        self.transformer_blocks = SelfTransformerBlocks(
            dim=dim,
            depth=transformer_layers,  # Example depth
            dim_head=64,
            heads=8,
            ff_mult=4,
            flash=True
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        # Compute STFT
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True)
        # Power-law compression
        X_mag = torch.abs(X)**0.3
        X_phase = torch.angle(X)
        X_compressed = torch.cat((X_mag, X_phase), dim=1)  # [b, 2*ceil((n - win_length) / hop_length + 1), 1 + n // hop_length]
        X_compressed = X_compressed[:, :, :-1] # [b, 2*ceil((n - win_length) / hop_length + 1), n // hop_length]
        # print("X_compressed.shape: ", X_compressed.shape)
        X_compressed = X_compressed.transpose(1, 2)
        # print("X_compressed.shape: ", X_compressed.shape)

        # Apply MLP
        mlp_output = self.mlp(X_compressed)

        # Reshape to [batch_size, seq_len, feature_dim] for transformer
        mlp_output = mlp_output.view(x.shape[0], -1, self.dim)  # Adjust according to actual sizes

        # Pass through Transformer
        embeddings = self.transformer_blocks(mlp_output)

        return embeddings

# main maskgit classes

@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vq_layers: int,
        vq_model: nn.Module,
        audio_encoder: AudioEncoder,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic = False,
        vae: Optional[VQGanVAE] = None,
        cond_drop_prob = 0.5,
        self_cond_prob = 0.9,
        no_mask_token_prob = 0.,
        critic_loss_weight = 1.
    ):
        super().__init__()
        self.vae = vae.copy_for_eval() if exists(vae) else None

        self.seq_len = seq_len
        self.vq_layers = vq_layers
        
        self.vq_model = vq_model

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob
        
        self.code_emb = nn.ModuleList([nn.Embedding(self.transformer.num_tokens + 1, self.transformer.dim) for _ in range(vq_layers)])
        self.audio_encoder = audio_encoder
        # print(self.audio_encoder)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        noisy_audios: torch.Tensor,
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        force_not_use_token_critic = False,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        cond_scale = 3,
        critic_noise_scale = 1
    ):
        # begin with all image token ids masked

        device = next(self.parameters()).device

        batch = noisy_audios.shape[0]

        # shape = (batch, self.vq_layers, self.seq_len)
        
        seq_len = self.vq_layers * self.seq_len
        shape = (batch, seq_len)

        ids = torch.full(shape, self.mask_id, dtype = torch.long, device = device)
        scores = torch.zeros(shape, dtype = torch.float32, device = device)

        starting_temperature = temperature

        cond_ids = None

        # text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        self_cond_embed = None

        # for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim = -1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            x = ids
            x = x.reshape(batch, self.vq_layers, self.seq_len)
        
            # encode x & noisy audio
            code_embeds = torch.sum(torch.stack([
                self.code_emb[i](x[:, i, :]) 
                for i in range(self.vq_layers)
            ], dim=1), dim=1) # [b, n, dim]
            audio_embeds = self.audio_encoder(noisy_audios) # [b, n, dim]
            
            x = code_embeds + audio_embeds

            logits, embed = demask_fn(
                x,
                self_cond_embed = self_cond_embed,
                conditioning_token_ids = cond_ids,
                cond_scale = cond_scale,
                return_embed = True
            )

            self_cond_embed = embed if self.self_cond else None
            
            logits = logits.reshape(batch, self.vq_layers * self.seq_len, -1)

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids == self.mask_id

            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    conditioning_token_ids = cond_ids,
                    cond_scale = cond_scale
                )

                scores = rearrange(scores, '... 1 -> ...')

                scores = scores + (uniform(scores.shape, device = device) - 0.5) * critic_noise_scale * (steps_until_x0 / timesteps)

            else:
                probs_without_temperature = logits.softmax(dim = -1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, '... 1 -> ...')

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert self.no_mask_token_prob > 0., 'without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token'

        # get ids

        ids = rearrange(ids, 'b (i j) -> b i j', i = self.vq_layers, j = self.seq_len)

        with torch.no_grad():
            audios = self.decode(ids)
        return ids, audios

    def encode(self, clean_audios: torch.Tensor):
        with torch.no_grad():
            x = self.vq_model.preprocess(clean_audios, 44100)
            z, codes, latents, _, _ = self.vq_model.encode(x)
        return codes

    def decode(self, codes: torch.Tensor):
        with torch.no_grad():
            z_q, _, _ = self.vq_model.quantizer.from_codes(codes)
            audios = self.vq_model.decode(z_q)
        return audios

    def forward(
        self,
        clean_audios: torch.Tensor,
        noisy_audios: torch.Tensor,
        ignore_index = -1,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        train_only_generator = False,
        sample_temperature = None
    ):
        '''
        audio_codes: [batch, audio_len]
        noisy_audios: [batch, audio_len]
        '''
        
        audio_codes = self.encode(clean_audios)
        
        # get some basic variables
        assert len(audio_codes.shape) == 3
        assert audio_codes.shape[1] == self.vq_layers
        assert audio_codes.shape[2] == self.seq_len
        ids = rearrange(audio_codes, 'b ... -> b (...)') # [b, vq_layers, n] -> [b, vq_layers*n]
        batch, seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)
        # print("ids.shape: ", ids.shape)

        # prepare mask

        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device = device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        mask_id = self.transformer.mask_id
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, mask_id, ids)
        
        x = x.reshape(batch, self.vq_layers, self.seq_len)
        
        # encode x & noisy audio
        code_embeds = torch.sum(torch.stack([
            self.code_emb[i](x[:, i, :]) 
            for i in range(self.vq_layers)
        ], dim=1), dim=1) # [b, n, dim]
        audio_embeds = self.audio_encoder(noisy_audios) # [b, n, dim]
        # print("code_embeds.shape: ", code_embeds.shape)
        # print("audio_embeds.shape: ", audio_embeds.shape)
        x = code_embeds + audio_embeds

        # get text embeddings

        if exists(texts):
            text_embeds = self.transformer.encode_text(texts)
            texts = None

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    text_embeds = text_embeds,
                    conditioning_token_ids = None,
                    cond_drop_prob = 0.,
                    return_embed = True
                )

                self_cond_embed.detach_()

        # get loss

        ce_loss, logits = self.transformer(
            x,
            text_embeds = text_embeds,
            self_cond_embed = self_cond_embed,
            conditioning_token_ids = None,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
            ignore_index = ignore_index,
            return_logits = True
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(logits, temperature = default(sample_temperature, random()))

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            text_embeds = text_embeds,
            conditioning_token_ids = None,
            labels = critic_labels,
            cond_drop_prob = cond_drop_prob
        )

        return ce_loss + self.critic_loss_weight * bce_loss

# final Muse class

@beartype
class Muse(nn.Module):
    def __init__(
        self,
        base: MaskGit,
        superres: MaskGit
    ):
        super().__init__()
        self.base_maskgit = base.eval()

        assert superres.resize_image_for_cond_image
        self.superres_maskgit = superres.eval()

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        cond_scale = 3.,
        temperature = 1.,
        timesteps = 18,
        superres_timesteps = None,
        return_lowres = False,
        return_pil_images = True
    ):
        lowres_image = self.base_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            temperature = temperature,
            timesteps = timesteps
        )

        superres_image = self.superres_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            cond_images = lowres_image,
            temperature = temperature,
            timesteps = default(superres_timesteps, timesteps)
        )
        
        if return_pil_images:
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            superres_image = list(map(T.ToPILImage(), superres_image))            

        if not return_lowres:
            return superres_image

        return superres_image, lowres_image
