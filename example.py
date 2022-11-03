import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import os
from PIL import Image
from transformers import CLIPTokenizer, FlaxCLIPTextModel, CLIPConfig
import tempfile
import time

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D,
    StableDiffusionSafetyCheckerModel,
)
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# convert diffusers checkpoint to jax
pt_path = "stable-diffusion-v1-4"
user = os.environ['USER']
fx_path = f"/home/{user}/stable-diffusion-v1-4-flax"
def model_path(model_name):
    return os.path.join(fx_path, model_name)

if not all([os.path.exists(model_path(m)) for m in ["unet", "vae"]]):
    print("All models not converted. Converting...")
    convert_diffusers_to_jax(pt_path, fx_path)


# inference with jax
dtype = jnp.bfloat16
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", _do_init=False, dtype=dtype
)
unet, unet_params = UNet2D.from_pretrained(model_path("unet"), _do_init=False, dtype=dtype)
vae, vae_params = AutoencoderKL.from_pretrained(model_path("vae"), _do_init=False, dtype=dtype)
# safety_model, safety_model_params = StableDiffusionSafetyCheckerModel.from_pretrained(f"{fx_path}/safety_model", _do_init=False, dtype=dtype)

config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler = PNDMScheduler()

# create inference state and replicate it across all TPU devices
inference_state = InferenceState(text_encoder_params=clip_params, unet_params=unet_params, vae_params=vae_params)
inference_state = replicate(inference_state)

# create pipeline
pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vae=vae)
sample = jax.pmap(pipe.sample, static_broadcasted_argnums=(4, 5))

def run_example(prompt, num_samples):
    print(f"Running for prompt: {prompt}")
    start = time.time()
    # prepare inputs
    input_ids = tokenizer(
        [prompt] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
    ).input_ids
    uncond_input_ids = tokenizer(
        [""] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
    ).input_ids
    prng_seed = jax.random.PRNGKey(42)

    # shard inputs and rng
    input_ids = shard(input_ids)
    uncond_input_ids = shard(uncond_input_ids)
    prng_seed = jax.random.split(prng_seed, jax.local_device_count())

    # pmap the sample function
    num_inference_steps = 50
    guidance_scale = 7.5
    print(f"tokenize inputs: {time.time() - start}")

    # sample images
    start = time.time()
    with jax.profiler.trace("./traces"):
        images = sample(
            input_ids,
            uncond_input_ids,
            prng_seed,
            inference_state,
            num_inference_steps,
            guidance_scale,
        )
        images.block_until_ready()
    print(f"sample time: {time.time() - start}")
    return images


# import jax.profiler
# jax.profiler.start_server(9999)
num_samples = 4
p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
images = run_example(p, num_samples)

p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
images = run_example(p, num_samples)

num_samples = 4
p = "A computer chip with wings flying above the clouds, epic, cinematic"
images = run_example(p, num_samples)
# jax.profiler.stop_server()

# convert images to PIL images
images = images / 2 + 0.5
images = jnp.clip(images, 0, 1)
images = (images * 255).round().astype("uint8")
images = np.asarray(images).reshape((num_samples, 512, 512, 3))

pil_images = [Image.fromarray(image) for image in images]

grid = image_grid(pil_images, rows=1, cols=num_samples)

temp_dir = tempfile.mkdtemp()
print(f"Outputting to directory: {temp_dir}")
grid.save(os.path.join(temp_dir, f"image.png"))
