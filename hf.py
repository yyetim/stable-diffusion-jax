import time

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import random

from diffusers import FlaxStableDiffusionPipeline
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import pmap


num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind
print(f"Found {num_devices} JAX devices of type {device_type}.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", required=True, type=str, help="Accelerator Type")
    parser.add_argument("--batch_sizes", 
        required=True, help="Delimited batch sizes",
        type=lambda batches: [int(batch) for batch in batches.split(',')])
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    dtype = jnp.bfloat16

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-4-bf16",
        dtype=dtype,
    )

    p_params = replicate(params)

    def create_key(seed=0):
        return jax.random.PRNGKey(seed)

    rng = create_key(0)
    rng = jax.random.split(rng, jax.device_count())

    p_generate = pmap(pipeline._generate)

    base_prompts = [
        "Labrador in the style of Hokusai",
        "Painting of a squirrel skating in New York",
        "HAL-9000 in the style of Van Gogh",
        "Times Square under water, with fish and a dolphin swimming around",
        "Ancient Roman fresco showing a man working on his laptop",
        "Close-up photograph of young black woman against urban background, high quality, bokeh",
        "Armchair in the shape of an avocado",
        "Clown astronaut in space, with Earth in the background",
    ]

    print(f"Running warmup on batch {len(base_prompts)}")
    base_prompt_ids = pipeline.prepare_inputs(base_prompts)
    base_prompt_ids = shard(base_prompt_ids)

    start_time = time.time()
    images = p_generate(base_prompt_ids, p_params, rng)
    images = images.block_until_ready()
    print("--- %s seconds ---" % (time.time() - start_time))

    for batch in args.batch_sizes:
        prompts = [random.choice(base_prompts) for _ in range(batch)]
        print(f"Profiling with batch size {len(prompts)}")
        prompt_ids = pipeline.prepare_inputs(prompts)
        prompt_ids = shard(prompt_ids)

        start_time = time.time()
        with jax.profiler.trace(f"./traces/stable-diffusion/jax/{args.accelerator}/batch{len(prompts)}/"):
            images = p_generate(prompt_ids, p_params, rng)
            images = images.block_until_ready()
        print("--- %s seconds ---" % (time.time() - start_time))
        print(images.shape)

if __name__ == "__main__":
  main()
