import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


class gpt2_conv1d(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[:-1] + (self.bias.shape[0],)
        return torch.addmm(
            self.bias,
            x.reshape(-1, x.shape[-1]),
            self.weight,
        ).view(output_shape)


class mlp_block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.up = gpt2_conv1d(config.n_embd, config.n_embd * 4)
        self.down = gpt2_conv1d(config.n_embd * 4, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )
        return self.down(x)


class gpt2_attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.in_proj = gpt2_conv1d(config.n_embd, 3 * config.n_embd)
        self.out_proj = gpt2_conv1d(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.in_proj(x).chunk(3, dim=-1)
        batch_size, seq_len, _ = query.shape

        def split_heads(states: torch.Tensor) -> torch.Tensor:
            return states.view(
                batch_size, seq_len, self.n_head, self.head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        causal_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=x.device
        ).tril()
        allowed_mask = causal_mask[None, None] & attention_mask[:, None, None].bool()
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(x.shape)
        return self.out_proj(attended)


class gpt2_block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            normalized_shape=config.n_embd, eps=config.layer_norm_epsilon
        )
        self.atten = gpt2_attention(config)
        self.norm2 = nn.LayerNorm(
            normalized_shape=config.n_embd, eps=config.layer_norm_epsilon
        )
        self.ffn = mlp_block(config)

    def forward(self, x: torch.Tensor, atten_mask) -> torch.Tensor:
        normed = self.norm1(x)
        key_padding_mask = atten_mask == 0
        attn = self.atten(normed, atten_mask)
        attn = attn.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        x = x + attn
        normed = self.norm2(x)
        x = self.ffn(normed) + x
        return x


class gpt2_small(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.token_embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
        )
        self.position_embed = nn.Embedding(
            num_embeddings=config.n_positions,
            embedding_dim=config.n_embd,
        )
        self.blocks = nn.ModuleList(gpt2_block(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(
            normalized_shape=config.n_embd, eps=config.layer_norm_epsilon
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, atten_mask) -> torch.Tensor:
        pos = (atten_mask.cumsum(dim=1) - 1).clamp_min(0)
        embed = self.token_embed(x) + self.position_embed(pos)
        for block in self.blocks:
            embed = block(embed, atten_mask)
        embed_norm = self.norm(embed)
        logits = self.lm_head(embed_norm)
        return logits


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Load pretrained GPT-2 Small weights into manually implemented transformer
    blocks. Generate for the entire batch at once, choosing the highest-logit
    token for every unfinished sequence at each step. Stop each sequence at EOS
    or max_seq_length total tokens, including the prompt.

    Return newly generated text for each prompt and a tensor of pre-selection
    logits shaped (batch_size, decoding_steps, 50257). Fill logits with zero
    after a row has finished while other rows continue.
    """

    """ Initialization """

    config = GPT2Config()

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = gpt2_small(config).to(dtype=torch.float16)
    pre_model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        dtype=torch.float16,
    )

    """ Copy Weight """

    state_dict = pre_model.state_dict()

    with torch.no_grad():
        model.token_embed.weight.copy_(state_dict["transformer.wte.weight"])
        model.position_embed.weight.copy_(state_dict["transformer.wpe.weight"])

        for index, block in enumerate(model.blocks):
            prefix = f"transformer.h.{index}"
            block.norm1.weight.copy_(state_dict[f"{prefix}.ln_1.weight"])
            block.norm1.bias.copy_(state_dict[f"{prefix}.ln_1.bias"])
            block.atten.in_proj.weight.copy_(state_dict[f"{prefix}.attn.c_attn.weight"])
            block.atten.in_proj.bias.copy_(state_dict[f"{prefix}.attn.c_attn.bias"])
            block.atten.out_proj.weight.copy_(
                state_dict[f"{prefix}.attn.c_proj.weight"]
            )
            block.atten.out_proj.bias.copy_(state_dict[f"{prefix}.attn.c_proj.bias"])
            block.norm2.weight.copy_(state_dict[f"{prefix}.ln_2.weight"])
            block.norm2.bias.copy_(state_dict[f"{prefix}.ln_2.bias"])
            block.ffn.up.weight.copy_(state_dict[f"{prefix}.mlp.c_fc.weight"])
            block.ffn.up.bias.copy_(state_dict[f"{prefix}.mlp.c_fc.bias"])
            block.ffn.down.weight.copy_(state_dict[f"{prefix}.mlp.c_proj.weight"])
            block.ffn.down.bias.copy_(state_dict[f"{prefix}.mlp.c_proj.bias"])

        model.norm.weight.copy_(state_dict["transformer.ln_f.weight"])
        model.norm.bias.copy_(state_dict["transformer.ln_f.bias"])
        model.lm_head.weight.copy_(state_dict["lm_head.weight"])

    """ Forward """
    model.eval()
    encoded = tokenizer(
        input,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    atten_mask = encoded["attention_mask"]
    lengths = atten_mask.sum(dim=1)
    finished = lengths >= max_seq_length
    generated = [[] for _ in input]
    steps = []

    with torch.inference_mode():
        while not bool(finished.all()):
            logits = model(token_ids, atten_mask)[:, -1, :].clone()
            active = ~finished
            logits[~active] = 0
            next_ids = logits.argmax(dim=-1)
            next_ids[~active] = tokenizer.eos_token_id
            steps.append(logits)

            for index in range(len(input)):
                if active[index]:
                    token = int(next_ids[index])
                    generated[index].append(token)
                    lengths[index] += 1
                    if (
                        token == tokenizer.eos_token_id
                        or lengths[index] >= max_seq_length
                    ):
                        finished[index] = True

            token_ids = torch.cat((token_ids, next_ids[:, None]), dim=1)
            atten_mask = torch.cat(
                (atten_mask, active[:, None].to(atten_mask.dtype)), dim=1
            )

    out_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    logits = (
        torch.stack(steps, dim=1)
        if steps
        else torch.empty((len(input), 0, config.vocab_size))
    )

    return out_str, logits
