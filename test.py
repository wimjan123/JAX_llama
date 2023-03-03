import jax
import jax.numpy as jnp
from llama import flax_model, model
import torch
import numpy as np

if __name__ == "__main__":

    # Compare RMSNorm

    jax_rms_norm = flax_model.RMSNorm(512)
    jax_params = jax_rms_norm.init(jax.random.PRNGKey(0), jnp.ones(512))['params']
    jax_output = jax_rms_norm.apply({'params': jax_params}, jnp.ones((1, 512)))

    torch_rms_norm = model.RMSNorm(512)
    torch_output = torch_rms_norm(torch.ones((1, 512)))

    jax_output = np.asarray(jax_output)
    torch_output = torch_output.detach().numpy()

    assert np.allclose(jax_output, torch_output)

    

    # import IPython; IPython.embed()
