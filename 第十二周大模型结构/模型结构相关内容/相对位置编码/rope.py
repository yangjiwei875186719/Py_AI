import torch
import importlib
from einops import rearrange

apply_rotary_emb_func = None

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            self._rotary_pos_emb_cache = rearrange(emb, "n d -> 1 n 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    if apply_rotary_emb_func is not None:
        t_ = t.float()
        freqs = freqs.squeeze(0).squeeze(1)
        cos = freqs[:, : freqs.shape[-1] // 2].cos()
        sin = freqs[:, : freqs.shape[-1] // 2].sin()
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs.shape[-1]
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)

batch_size = 2
max_len = 5
num_head = 4
emb_size = 8
re = RotaryEmbedding(emb_size)
print(re.state_dict())
r = re.forward(max_len)[:,-max_len:, :, :]
print(r)
print(r.shape)

q = torch.rand((batch_size, max_len, num_head, emb_size))
x = apply_rotary_pos_emb(q, r)
print(x)
print(x.shape)