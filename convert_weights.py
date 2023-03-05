from pathlib import Path
import torch
import fire
import json
import numpy as np
from jaxtyping import PyTree
import flax
import jax
import jax.numpy as jnp
import os

def convert_llama_weights(ckpt_dir: str, verbose: bool=False) -> PyTree[np.ndarray]:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpts = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        if verbose:
            print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            print('Loaded.')
        ckpts[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    jax_weights = {
        'tok_embeddings': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].numpy() for ckpt in ckpts], axis=1)}, 
        'norm': {'kernel': ckpts[0]['norm.weight'].numpy()}, 
        'output': {'kernel': np.concatenate([ckpt['output.weight'].numpy() for ckpt in ckpts], axis=0).transpose()}, 
        **{'layers_%d' % (layer): {
            'attention': {
                'wq': {'kernel': np.concatenate([ckpt['layers.%d.attention.wq.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                'wk': {'kernel': np.concatenate([ckpt['layers.%d.attention.wk.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                'wv': {'kernel': np.concatenate([ckpt['layers.%d.attention.wv.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                'wo': {'kernel': np.concatenate([ckpt['layers.%d.attention.wo.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()}, 
            }, 
            'feed_forward': {
                'w1': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w1.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
                'w2': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w2.weight' % (layer)].numpy() for ckpt in ckpts], axis=1).transpose()}, 
                'w3': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w3.weight' % (layer)].numpy() for ckpt in ckpts], axis=0).transpose()}, 
            }, 
            'attention_norm': {'kernel': ckpts[0]['layers.%d.attention_norm.weight' % (layer)].numpy()}, 
            'ffn_norm': {'kernel': ckpts[0]['layers.%d.ffn_norm.weight' % (layer)].numpy()}, 
        } for layer in range(params['n_layers'])}, 
    }
    return jax_weights, params

def main(ckpt_dir: str, out_dir: str, verbose: bool=False):
    if verbose:
        print(f'Converting {ckpt_dir} to {out_dir}')

    # get weights
    jax_weights, params = convert_llama_weights(ckpt_dir, verbose)
    with jax.default_device(jax.devices('cpu')[0]):
        jax_weights = jax.tree_map(lambda x: jnp.asarray(x), jax_weights)
    
    # save weights
    if verbose:
        print('Saving weights ...')
    out_dir = str(Path(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'model.msgpack'), 'wb') as f:
        f.write(flax.serialization.msgpack_serialize(jax_weights))
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    if verbose:
        print('Saved.')

if __name__ == '__main__':
    fire.Fire(main)
