import jax
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert "TPU" in device_type

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline

import time

dtype = jnp.bfloat16

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)

p_params = replicate(params)

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())

prompts = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Close-up photograph of young black woman against urban background, high quality, bokeh",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
] * 16
print(f"Running on batch {len(prompts)}")
prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)


p_generate = pmap(pipeline._generate)

start_time = time.time()
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()
print("--- %s seconds ---" % (time.time() - start_time))

print(images.shape)
