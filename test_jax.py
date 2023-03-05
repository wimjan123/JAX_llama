import jax
import jax.numpy as jnp
from llama import flax_model, model
import torch
import numpy as np
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Tuple, List, Tuple
import os
from flax.core.frozen_dict import unfreeze, freeze
from dataclasses import dataclass
import functools

## TODO:
# assert device

@dataclass
class ModelArgs:
    dim: int = 32
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 256
    multiple_of: int = 2
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_len: int = 64

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def test_RMSNorm(args: ModelArgs, total_tests: int, atol: float) -> np.ndarray:
    errs = []
    for test_n in range(total_tests):
        x = np.random.randn(args.max_batch_size, args.dim).astype(np.float32)

        jax_rms_norm = flax_model.RMSNorm(args.dim)
        jax_params = jax_rms_norm.init(jax.random.PRNGKey(0), jnp.ones(args.dim, dtype=jnp.float32))['params']
        jax_output = jax_rms_norm.apply({'params': jax_params}, jnp.asarray(x))
        jax_output = np.asarray(jax_output)

        torch_rms_norm = model.RMSNorm(args.dim)
        torch_output = torch_rms_norm(torch.tensor(x))
        torch_output = torch_output.detach().numpy()

        assert np.allclose(jax_output, torch_output, atol=atol), f"RMSNorm test {test_n} failed"
        errs.append(np.max(np.abs(jax_output - torch_output)))
    return np.asarray(errs, dtype=np.float32)

def test_precompute_freqs_cis(args: ModelArgs, atol: float) -> float:
    jax_freqs_cis = flax_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    jax_freqs_cis = np.asarray(jax_freqs_cis)

    torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    torch_freqs_cis = torch_freqs_cis.detach().numpy()

    assert np.allclose(jax_freqs_cis, torch_freqs_cis, atol=atol), f"precompute_freqs_cis test failed"
    return np.max(np.abs(jax_freqs_cis - torch_freqs_cis))

def test_compare_reshape_for_broadcast(args: ModelArgs, atol: float) -> float:
    x = np.zeros((args.max_batch_size, args.max_seq_len, args.dim // (args.n_heads * 2),), dtype=np.float32)
    
    jax_freqs_cis = flax_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    jax_output = flax_model.reshape_for_broadcast(jax_freqs_cis, jnp.asarray(x))
    jax_output = np.asarray(jax_output)

    torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    torch_output = model.reshape_for_broadcast(torch_freqs_cis, torch.tensor(x))
    torch_output = torch_output.detach().numpy()

    assert np.allclose(jax_output, torch_output, atol=atol), f"reshape_for_broadcast test failed"
    return np.max(np.abs(jax_output - torch_output))

def test_apply_roary_emb(args: ModelArgs, total_tests: int, atol: float) -> Tuple[np.ndarray]:
    errs0, errs1 = [], []
    for test_n in range(total_tests):
        xq = np.random.randn(args.max_batch_size, args.max_seq_len, args.n_heads, args.dim // args.n_heads).astype(np.float32)
        xk = np.random.randn(args.max_batch_size, args.max_seq_len, args.n_heads, args.dim // args.n_heads).astype(np.float32)

        jax_freqs_cis = flax_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        jax_output = flax_model.apply_rotary_emb(jnp.asarray(xq), jnp.asarray(xk), jax_freqs_cis)
        jax_output = (np.asarray(jax_output[0]), np.asarray(jax_output[1]))

        torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        torch_output = model.apply_rotary_emb(torch.tensor(xq), torch.tensor(xk), torch_freqs_cis)
        torch_output = (torch_output[0].detach().numpy(), torch_output[1].detach().numpy())

        assert np.allclose(jax_output[0], torch_output[0], atol=atol) and \
            np.allclose(jax_output[1], torch_output[1], atol=atol), f"apply_rotary_emb test {test_n} failed"
        errs0.append(np.max(np.abs(jax_output[0] - torch_output[0])))
        errs1.append(np.max(jax_output[1] - torch_output[1]))
    return np.asarray(errs0, dtype=np.float32), np.asarray(errs1, dtype=np.float32)

def test_Attention(args: ModelArgs, total_tests: int, atol: float) -> np.ndarray:
    errs = []
    for test_n in range(total_tests):
        x = np.random.randn(args.max_batch_size, args.max_seq_len, args.dim).astype(np.float32)
        wq = np.random.randn(args.dim, args.dim).astype(np.float32)
        wk = np.random.randn(args.dim, args.dim).astype(np.float32)
        wv = np.random.randn(args.dim, args.dim).astype(np.float32)
        wo = np.random.randn(args.dim, args.dim).astype(np.float32)

        jax_freqs_cis = flax_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        jax_attention = flax_model.Attention(
            n_heads=args.n_heads, 
            dim=args.dim, 
            max_batch_size=args.max_batch_size, 
            max_seq_len=args.max_seq_len, 
        )
        jax_state = jax_attention.init(
            jax.random.PRNGKey(0), 
            jnp.ones((args.max_batch_size, args.max_seq_len, args.dim), dtype=jnp.float32), 
            0, 
            jax_freqs_cis, 
            None, 
        )
        # load weights
        jax_state = unfreeze(jax_state)
        jax_state['params'] = {
            'wq': {'kernel': jnp.asarray(wq)}, 
            'wk': {'kernel': jnp.asarray(wk)}, 
            'wv': {'kernel': jnp.asarray(wv)}, 
            'wo': {'kernel': jnp.asarray(wo)}, 
        }
        jax_state = freeze(jax_state)
        # get output
        jax_output, _ = jax_attention.apply(
            jax_state, 
            jnp.asarray(x), 
            0, 
            jax_freqs_cis, 
            None, 
            mutable=['cache'], 
        )
        jax_output = np.asarray(jax_output)

        torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        torch_attention = model.Attention(
            model.ModelArgs(
                n_heads=args.n_heads, 
                dim=args.dim, 
                max_batch_size=args.max_batch_size, 
                max_seq_len=args.max_seq_len, 
            ), 
        )
        torch_attention.load_state_dict({
            "wo.weight": torch.tensor(wo.transpose()), 
            "wq.weight": torch.tensor(wq.transpose()), 
            "wv.weight": torch.tensor(wv.transpose()), 
            "wk.weight": torch.tensor(wk.transpose()), 
        }) # load weights, have to transpose because pytorch linear layers are reversed from Jax.
        torch_output = torch_attention(torch.tensor(x), 0, torch_freqs_cis, None)
        torch_output = torch_output.detach().numpy()

        assert np.allclose(jax_output, torch_output, atol=atol), f"Attention test {test_n} failed"
        errs.append(np.max(np.abs(jax_output - torch_output)))
    return np.asarray(errs, dtype=np.float32)

def test_feedForward(args: ModelArgs, total_tests: int, atol: float) -> List[float]:
    hidden_dim = args.dim * 4
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

    errs = []
    for test_n in range(total_tests):
        x = np.random.randn(args.max_batch_size, args.max_seq_len, args.dim).astype(np.float32)
        w1 = np.random.randn(args.dim, hidden_dim).astype(np.float32)
        w2 = np.random.randn(hidden_dim, args.dim).astype(np.float32)
        w3 = np.random.randn(args.dim, hidden_dim).astype(np.float32)

        jax_mlp = flax_model.FeedForward(
            dim=args.dim, 
            hidden_dim=args.dim*4, 
            multiple_of=args.multiple_of, 
        )
        jax_params = freeze({
            'w1': {'kernel': jnp.asarray(w1)}, 
            'w2': {'kernel': jnp.asarray(w2)}, 
            'w3': {'kernel': jnp.asarray(w3)}, 
        }) # load weights
        jax_output = jax_mlp.apply({'params': jax_params}, jnp.asarray(x))
        jax_output = np.asarray(jax_output)

        torch_mlp = model.FeedForward(
            dim=args.dim, 
            hidden_dim=args.dim*4, 
            multiple_of=args.multiple_of
        )
        torch_mlp.load_state_dict({
            "w1.weight": torch.tensor(w1.transpose()), 
            "w2.weight": torch.tensor(w2.transpose()), 
            "w3.weight": torch.tensor(w3.transpose()), 
        }) # load weights, have to transpose because pytorch linear layers are reversed from Jax.
        torch_output = torch_mlp(torch.tensor(x))
        torch_output = torch_output.detach().numpy()

        assert np.allclose(jax_output, torch_output, atol=atol), f"FeedForward test {test_n} failed"
        errs.append(np.max(np.abs(jax_output - torch_output)))
    return np.asarray(errs, dtype=np.float32)

def test_TransformerBlock(args: ModelArgs, total_tests: int, atol: float) -> np.ndarray:
    hidden_dim = args.dim * 4
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
    errs = []
    for test_n in range(total_tests):
        x = np.random.randn(args.max_batch_size, args.max_seq_len, args.dim).astype(np.float32)
        wq = np.random.randn(args.dim, args.dim).astype(np.float32)
        wk = np.random.randn(args.dim, args.dim).astype(np.float32)
        wv = np.random.randn(args.dim, args.dim).astype(np.float32)
        wo = np.random.randn(args.dim, args.dim).astype(np.float32)
        w1 = np.random.randn(args.dim, hidden_dim).astype(np.float32)
        w2 = np.random.randn(hidden_dim, args.dim).astype(np.float32)
        w3 = np.random.randn(args.dim, hidden_dim).astype(np.float32)
        attention_norm_scale = np.random.randn(args.dim).astype(np.float32)
        ffn_norm_scale = np.random.randn(args.dim).astype(np.float32)

        jax_freqs_cis = flax_model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        jax_transformer_block = flax_model.TransformerBlock(
            dim=args.dim, 
            n_heads=args.n_heads, 
            multiple_of=args.multiple_of, 
            norm_eps=args.norm_eps, 
            max_batch_size=args.max_batch_size, 
            max_seq_len=args.max_seq_len, 
        )
        jax_state = jax_transformer_block.init(
            jax.random.PRNGKey(0), 
            jnp.ones((args.max_batch_size, args.max_seq_len, args.dim), dtype=jnp.float32), 
            0, 
            jax_freqs_cis, 
            None, 
        )
        # load weights
        jax_state = unfreeze(jax_state)
        jax_state['params'] = {
            'attention': {
                'wq': {'kernel': jnp.asarray(wq)}, 
                'wk': {'kernel': jnp.asarray(wk)}, 
                'wv': {'kernel': jnp.asarray(wv)}, 
                'wo': {'kernel': jnp.asarray(wo)}, 
            }, 
            'feed_forward': {
                'w1': {'kernel': jnp.asarray(w1)}, 
                'w2': {'kernel': jnp.asarray(w2)}, 
                'w3': {'kernel': jnp.asarray(w3)}, 
            }, 
            'attention_norm': {'kernel': jnp.asarray(attention_norm_scale)}, 
            'ffn_norm': {'kernel': jnp.asarray(ffn_norm_scale)}, 
        }
        jax_state = freeze(jax_state)
        # get output
        jax_output, _ = jax_transformer_block.apply(
            jax_state, 
            jnp.asarray(x), 
            0, 
            jax_freqs_cis, 
            None, 
            mutable=['cache'], 
        )
        jax_output = np.asarray(jax_output)

        torch_freqs_cis = model.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        torch_transformer_block = model.TransformerBlock(
            0, 
            model.ModelArgs(
                dim=args.dim, 
                n_heads=args.n_heads, 
                multiple_of=args.multiple_of, 
                norm_eps=args.norm_eps, 
                max_batch_size=args.max_batch_size, 
                max_seq_len=args.max_seq_len, 
            ), 
        )
        torch_transformer_block.load_state_dict({
            "attention.wo.weight": torch.tensor(wo.transpose()), 
            "attention.wq.weight": torch.tensor(wq.transpose()), 
            "attention.wv.weight": torch.tensor(wv.transpose()), 
            "attention.wk.weight": torch.tensor(wk.transpose()), 
            "feed_forward.w1.weight": torch.tensor(w1.transpose()), 
            "feed_forward.w2.weight": torch.tensor(w2.transpose()), 
            "feed_forward.w3.weight": torch.tensor(w3.transpose()), 
            "attention_norm.weight": torch.tensor(attention_norm_scale), 
            "ffn_norm.weight": torch.tensor(ffn_norm_scale), 
        }) # load weights, have to transpose because pytorch linear layers are reversed from Jax.
        torch_output = torch_transformer_block(torch.tensor(x), 0, torch_freqs_cis, None)
        torch_output = torch_output.detach().numpy()

        assert np.allclose(jax_output, torch_output, atol=atol), f"TransformerBlock test {test_n} failed"
        errs.append(np.abs(jax_output - torch_output).max())
    return np.asarray(errs, dtype=np.float32)

def test_Transformer(args: ModelArgs, total_tests: int, atol: float) -> np.ndarray:
    hidden_dim = args.dim * 4
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
    errs = []
    for test_n in range(total_tests):
        x = np.random.randint(low=0, high=args.vocab_size, size=(args.max_batch_size, args.max_seq_len), dtype=np.int32)
        layer_weights = [
            {
                'attention': {
                    'wq': {'kernel': np.random.randn(args.dim, args.dim).astype(np.float32)}, 
                    'wk': {'kernel': np.random.randn(args.dim, args.dim).astype(np.float32)}, 
                    'wv': {'kernel': np.random.randn(args.dim, args.dim).astype(np.float32)}, 
                    'wo': {'kernel': np.random.randn(args.dim, args.dim).astype(np.float32)}, 
                }, 
                'feed_forward': {
                    'w1': {'kernel': np.random.randn(args.dim, hidden_dim).astype(np.float32)}, 
                    'w2': {'kernel': np.random.randn(hidden_dim, args.dim).astype(np.float32)}, 
                    'w3': {'kernel': np.random.randn(args.dim, hidden_dim).astype(np.float32)}, 
                }, 
                'attention_norm': {'kernel': np.random.randn(args.dim).astype(np.float32)}, 
                'ffn_norm': {'kernel': np.random.randn(args.dim).astype(np.float32)}, 
            } for _ in range(args.n_layers)
        ]
        tok_embeddings = np.random.randn(args.vocab_size, args.dim).astype(np.float32)
        norm = np.random.randn(args.dim).astype(np.float32)
        output = np.random.randn(args.dim, args.vocab_size).astype(np.float32)

        jax_transformer = flax_model.Transformer(
            vocab_size=args.vocab_size, 
            n_layers=args.n_layers, 
            dim=args.dim, 
            n_heads=args.n_heads, 
            multiple_of=args.multiple_of, 
            norm_eps=args.norm_eps, 
            max_batch_size=args.max_batch_size, 
            max_seq_len=args.max_seq_len, 
        )
        jax_state = jax_transformer.init(
            jax.random.PRNGKey(0), 
            jnp.ones((args.max_batch_size, args.max_seq_len), dtype=jnp.int32), 
            0, 
        )
        # load weights
        jax_state = unfreeze(jax_state)
        jax_state['params'] = {
            'tok_embeddings': {'embedding': jnp.asarray(tok_embeddings)}, 
            'norm': {'kernel': jnp.asarray(norm)}, 
            'output': {'kernel': jnp.asarray(output)}, 
            **{'layers_%d' % (i): layer_weights[i] for i in range(args.n_layers)}, 
        }
        jax_state = freeze(jax_state)
        # get output
        jax_output, _ = jax_transformer.apply(
            jax_state, 
            jnp.asarray(x), 
            0, 
            mutable=['cache'], 
        )
        jax_output = np.asarray(jax_output)

        torch_transformer = model.Transformer(
            model.ModelArgs(
                vocab_size=args.vocab_size, 
                n_layers=args.n_layers, 
                dim=args.dim, 
                n_heads=args.n_heads, 
                multiple_of=args.multiple_of, 
                norm_eps=args.norm_eps, 
                max_batch_size=args.max_batch_size, 
                max_seq_len=args.max_seq_len, 
            ), 
        )
        torch_layer_weight = lambda i: {
            "layers.%d.attention.wo.weight" % (i): torch.tensor(layer_weights[i]['attention']['wo']['kernel'].transpose()), 
            "layers.%d.attention.wq.weight" % (i): torch.tensor(layer_weights[i]['attention']['wq']['kernel'].transpose()), 
            "layers.%d.attention.wv.weight" % (i): torch.tensor(layer_weights[i]['attention']['wv']['kernel'].transpose()), 
            "layers.%d.attention.wk.weight" % (i): torch.tensor(layer_weights[i]['attention']['wk']['kernel'].transpose()), 
            "layers.%d.feed_forward.w1.weight" % (i): torch.tensor(layer_weights[i]['feed_forward']['w1']['kernel'].transpose()), 
            "layers.%d.feed_forward.w2.weight" % (i): torch.tensor(layer_weights[i]['feed_forward']['w2']['kernel'].transpose()), 
            "layers.%d.feed_forward.w3.weight" % (i): torch.tensor(layer_weights[i]['feed_forward']['w3']['kernel'].transpose()), 
            "layers.%d.attention_norm.weight" % (i): torch.tensor(layer_weights[i]['attention_norm']['kernel']), 
            "layers.%d.ffn_norm.weight" % (i): torch.tensor(layer_weights[i]['ffn_norm']['kernel']), 
        }
        torch_transformer.load_state_dict({
            "tok_embeddings.weight": torch.tensor(tok_embeddings), 
            "norm.weight": torch.tensor(norm), 
            "output.weight": torch.tensor(output.transpose()), 
            **functools.reduce(lambda x, y: {**x, **y}, [torch_layer_weight(i) for i in range(args.n_layers)]), 
        }) # load weights
        torch_output = torch_transformer(torch.tensor(x), 0)
        torch_output = torch_output.detach().numpy()

        assert np.allclose(jax_output, torch_output, atol=atol), f"Transformer test {test_n} failed"
        errs.append(np.max(np.abs(jax_output - torch_output)))
    return np.asarray(errs, dtype=np.float32)

if __name__ == "__main__":
    np.random.seed(0)
    setup_model_parallel()

    with torch.no_grad():
        print('='*10)
        print("[Testing RMSNorm]")
        errs = test_RMSNorm(ModelArgs(), 128, atol=1e-8)
        print("[Passed]")
        print("Max RMSNorm error: %f" % (np.max(errs)))
        print("Mean RMSNorm error: %f" % (np.mean(errs)))
        print("Median RMSNorm error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing precompute_freqs_cis]")
        errs = test_precompute_freqs_cis(ModelArgs(), atol=1e-8)
        print("[Passed]")
        print("Max precompute_freqs_cis error: %f" % (np.max(errs)))
        print("Mean precompute_freqs_cis error: %f" % (np.mean(errs)))
        print("Median precompute_freqs_cis error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing reshape_for_broadcast]")
        errs = test_compare_reshape_for_broadcast(ModelArgs(), atol=1e-8)
        print("[Passed]")
        print("Max reshape_for_broadcast error: %f" % (np.max(errs)))
        print("Mean reshape_for_broadcast error: %f" % (np.mean(errs)))
        print("Median reshape_for_broadcast error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing apply_rotary_emb]")
        errs0, errs1 = test_apply_roary_emb(ModelArgs(), 128, atol=1e-6)
        print("[Passed]")
        print("Max apply_rotary_emb error: %f, %f" % (np.max(errs0), np.max(errs1)))
        print("Mean apply_rotary_emb error: %f, %f" % (np.mean(errs0), np.mean(errs1)))
        print("Median apply_rotary_emb error: %f, %f" % (np.median(errs0), np.median(errs1)))
        print('='*10)

        print('='*10)
        print("[Testing Attention]")
        errs = test_Attention(ModelArgs(), 128, atol=1e-2)
        print("[Passed]")
        print("Max Attention error: %f" % (np.max(errs)))
        print("Mean Attention error: %f" % (np.mean(errs)))
        print("Median Attention error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing FeedForward]")
        errs = test_feedForward(ModelArgs(), 128, atol=1e-3)
        print("[Passed]")
        print("Max FeedForward error: %f" % (np.max(errs)))
        print("Mean FeedForward error: %f" % (np.mean(errs)))
        print("Median FeedForward error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing TransformerBlock]")
        errs = test_TransformerBlock(ModelArgs(), 128, atol=1e-1)
        print("[Passed]")
        print("Max TransformerBlock error: %f" % (np.max(errs)))
        print("Mean TransformerBlock error: %f" % (np.mean(errs)))
        print("Median TransformerBlock error: %f" % (np.median(errs)))
        print('='*10)

        print('='*10)
        print("[Testing Transformer]")
        errs = test_Transformer(ModelArgs(), 128, atol=1e-2)
        print("[Passed]")
        print("Max Transformer error: %f" % (np.max(errs)))
        print("Mean Transformer error: %f" % (np.mean(errs)))
        print("Median Transformer error: %f" % (np.median(errs)))
        print('='*10)
