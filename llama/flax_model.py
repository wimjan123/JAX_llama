import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

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

# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
#     t = jnp.arange(end) # type: ignore
#     freqs = jnp.outer(t, freqs).float()  # type: ignore
#     freqs_cis = jnp.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis

# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis
