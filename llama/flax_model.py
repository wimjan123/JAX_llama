import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
from typing import Tuple, Optional
import math

class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel', 
            nn.initializers.ones, 
            (self.dim,), 
            self.param_dtype, 
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x.astype(self.dtype)).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(np.float32)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)

def apply_rotary_emb(
    xq: jnp.ndarray, 
    xk: jnp.ndarray, 
    freqs_cis: jnp.ndarray, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

class Attention(nn.Module):
    n_heads: int
    dim: int
    max_batch_size: int
    max_seq_len: int
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Dense(
            self.n_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.wk = nn.Dense(
            self.n_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.wv = nn.Dense(
            self.n_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.wo = nn.Dense(
            self.dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )

        self.cache_k = self.variable("cache", "cache_k", jnp.zeros, (self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim), self.dtype)
        self.cache_v = self.variable("cache", "cache_v", jnp.zeros, (self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim), self.dtype)

    def __call__(self, x: jnp.ndarray, start_pos: int, freqs_cis: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k.value = self.cache_k.value.at[:bsz, start_pos:(start_pos+seqlen)].set(xk)
        self.cache_v.value = self.cache_v.value.at[:bsz, start_pos:(start_pos+seqlen)].set(xv)

        keys = self.cache_k.value[:bsz, :(start_pos+seqlen)]
        values = self.cache_v.value[:bsz, :(start_pos+seqlen)]

        xq = jnp.transpose(xq, axes=(0, 2, 1, 3))
        keys = jnp.transpose(keys, axes=(0, 2, 1, 3))
        values = jnp.transpose(values, axes=(0, 2, 1, 3))
        scores = jnp.matmul(xq, jnp.transpose(keys, axes=(0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, slen, cache_len + slen)
        scores = jax.nn.softmax(scores.astype(self.dtype), axis=-1).astype(self.dtype)
        output = jnp.matmul(scores, values)  # (bs, n_heads, slen, head_dim)
        output = jnp.transpose(output, axes=(0, 2, 1, 3)).reshape(bsz, seqlen, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    multiple_of: int
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        hidden_dim = int(2 * self.hidden_dim / 3)
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.w1 = nn.Dense(
            hidden_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.w2 = nn.Dense(
            self.dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.w3 = nn.Dense(
            hidden_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    dim: int
    n_heads: int
    multiple_of: int
    norm_eps: float
    max_batch_size: int
    max_seq_len: int
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(
            self.n_heads, 
            self.dim, 
            self.max_batch_size, 
            self.max_seq_len, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )
        self.feed_forward = FeedForward(
            self.dim, 
            4 * self.dim, 
            self.multiple_of, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
        )
        self.attention_norm = RMSNorm(
            self.dim, 
            eps=self.norm_eps, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )
        self.ffn_norm = RMSNorm(
            self.dim, 
            eps=self.norm_eps, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )

    def __call__(self, x: jnp.ndarray, start_pos: int, freqs_cis: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    vocab_size: int
    n_layers: int
    dim: int
    n_heads: int
    multiple_of: int
    norm_eps: float
    max_batch_size: int
    max_seq_len: int
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.tok_embeddings = nn.Embed(
            self.vocab_size, 
            self.dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
        )
        self.layers = [
            TransformerBlock(
                self.dim, 
                self.n_heads, 
                self.multiple_of, 
                self.norm_eps, 
                self.max_batch_size, 
                self.max_seq_len, 
                dtype=self.dtype, 
                param_dtype=self.param_dtype, 
            ) for _ in range(self.n_layers)
        ]
        self.norm = RMSNorm(
            self.dim, 
            eps=self.norm_eps, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
        )
        self.output = nn.Dense(
            self.vocab_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
        )
        self.freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads, 
            self.max_seq_len * 2, 
        )

    def __call__(self, tokens: jnp.ndarray, start_pos: int) -> jnp.ndarray:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos:(start_pos+seqlen)]

        mask = None
        if seqlen > 1:
            mask = jnp.full((1, 1, seqlen, seqlen), float("-inf"), dtype=self.dtype)
            mask = jnp.triu(mask, k=start_pos+1).astype(self.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.astype(self.dtype)
