import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np

class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param('kernel', 
                        nn.initializers.ones, 
                        (self.dim,), 
                        self.param_dtype)

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight

# jax rotary embs
# jax doesn't have imaginary data types (or does it?), so gptj splits the real and imaginary parts into 2 different dimensions on the same tensor
# from gptj https://github.com/huggingface/transformers/blob/37e0974afcbccdc85da59d51b44e1437b6b3caea/src/transformers/models/gptj/modeling_flax_gptj.py#L109
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)

# pytorch rotary embs
# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis
