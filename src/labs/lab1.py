import torch
from torch import nn
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

class mlp_block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.up = nn.Linear(
            config.n_embd,
            config.n_embd * 4
        )
        self.activation = nn.GELU(approximate="tanh")
        self.down = nn.Linear(
            config.n_embd * 4, 
            config.n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.up(x))
        return self.down(x)

class gpt2_block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            normalized_shape=config.n_embd,
            eps=config.layer_norm_epsilon
        )
        self.atten = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=config.n_embd,
            eps=config.layer_norm_epsilon
        )
        self.ffn = mlp_block(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x, _ = self.atten(
            normed,
            normed,
            normed,
            is_causal=True, 
            need_weight=False
        ) + x
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
        self.blocks = nn.ModuleList(
            gpt2_block(config) for _ in range(config.n_layer)
        )
        self.norm = nn.LayerNorm(
            normalized_shape=config.n_embd, 
            eps=config.layer_norm_epsilon
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arrange(0, x.shape[1])
        embed = self.token_embed(x) + self.position_embed(x)
        for block in self.blocks:
            embed = block(embed)
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

    tokens = tokenizer(
        input,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

    model = gpt2_small(config)
    pre_model = GPT2LMHeadModel.from_pretrained("gpt2")

    """ Copy Weight """

    state_dict = pre_model.state_dict()

    # TODO: copy weight from pretrained gpt2 into gpt2_small class    

    """ Forward """

    raise NotImplementedError("Implement GPT-2 Small here")

if __name__ == "__main__":
    gpt2_complete(["hi"])