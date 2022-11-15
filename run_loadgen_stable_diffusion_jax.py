import array
import tempfile
import time
import os
import pathlib

import argparse
import jax
import jax.numpy as jnp
import mlperf_loadgen as lg
import numpy as np
import pickle
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPConfig
from transformers import CLIPTokenizer
from transformers import FlaxCLIPTextModel

from stable_diffusion_jax import AutoencoderKL
from stable_diffusion_jax import InferenceState
from stable_diffusion_jax import PNDMScheduler
from stable_diffusion_jax import StableDiffusionPipeline
from stable_diffusion_jax import StableDiffusionSafetyCheckerModel
from stable_diffusion_jax import UNet2D
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server", "MultiStream"], default="MultiStream", help="Scenario")
    parser.add_argument("--n_chips", required=True, type=int, help="Number of TPU chips")
    parser.add_argument("--batch_sizes", 
        required=True, help="Delimited batch sizes",
        type=lambda batches: [int(batch) for batch in batches.split(',')])
    args = parser.parse_args()
    return args

class StableDiffusionJax():
  def __init__(self,
    count = 4,
    perf_count_override = None,
    cache_path='features.pickle'
  ):
    print("Initializing model...")
    self._count = count
    self._perf_count = perf_count_override or self._count

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

    clip_params = clip_model.to_bf16(clip_params)
    unet_params = unet.to_bf16(unet_params)
    vae_params = vae.to_bf16(vae_params)

    config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = PNDMScheduler()

    # create inference state and replicate it across all TPU devices
    inference_state = InferenceState(text_encoder_params=clip_params, unet_params=unet_params, vae_params=vae_params)
    self._inference_state = replicate(inference_state)

    # create pipeline
    pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=self.tokenizer, unet=unet, scheduler=scheduler, vae=vae)
    self.sample = jax.pmap(pipe.sample, static_broadcasted_argnums=(4, 5))

    # create inference params
    self._prng_seed = jax.random.PRNGKey(42)
    self._num_inference_steps = 50
    self._guidance_scale = 7.5

    # create data for load test
    print(f"Creating data of size {self._count}...")
    self._input_ids, self._uncond_input_ids = self.create_dataset(self._count, cache_path)

  def create_dataset(self, count, cache_path):
    if os.path.exists(cache_path):
      print("Loading input_ids and uncond_input_ids from '%s'..." % cache_path)
      with open(cache_path, 'rb') as cache_file:
        input_ids, uncond_input_ids = pickle.load(cache_file)
    else:
      # load prompts, filter out invalid, and sample
      dataset = load_dataset("succinctly/midjourney-prompts")
      test_data = dataset['test'].to_pandas()
      prompts = test_data[test_data['text'].str.len() > 10]
      sampled_prompts = prompts.sample(count).text

      # tokenize
      input_ids = self.tokenizer(
          list(sampled_prompts), padding="max_length", truncation=True, max_length=77, return_tensors="jax"
      ).input_ids
      uncond_input_ids = self.tokenizer(
          [""] * count, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
      ).input_ids

      print("Caching input_ids and uncond_input_ids at '%s'..." % cache_path)
      with open(cache_path, 'wb') as cache_file:
        pickle.dump((input_ids, uncond_input_ids), cache_file)
    return input_ids, uncond_input_ids

  def create_sut(self):
    return lg.ConstructSUT(
        self._issue_queries, self._flush_queries, self._process_latencies)

  def create_qsl(self):
    return lg.ConstructQSL(
        self._count, self._perf_count, self._load_query_samples, self._unload_query_samples)

  def _load_query_samples(self, sample_list):
    pass

  def _unload_query_samples(self, sample_list):
    pass

  def _flush_queries(self):
    pass

  def _process_latencies(self, latencies_ns):
    print(f"All latencies {latencies_ns}, len {len(latencies_ns)}")
    print("Average latency: ")
    print(np.mean(latencies_ns))
    print("Median latency: ")
    print(np.percentile(latencies_ns, 50))
    print("90 percentile latency: ")
    print(np.percentile(latencies_ns, 90))

  def _issue_queries(self, query_samples):
    # get data
    indices = [query_sample.index for query_sample in query_samples]
    queried_input_ids = jax.numpy.take(self._input_ids, jax.numpy.array(indices), axis = 0)
    queried_uncond_input_ids = jax.numpy.take(self._uncond_input_ids, jax.numpy.array(indices), axis = 0)

    # predict
    images = self.predict(queried_input_ids, queried_uncond_input_ids)

    # create and return response
    response = []
    for idx, sample in enumerate(query_samples):
      response_array = array.array("B", images[idx].tobytes())
      bi = response_array.buffer_info()
      response.append(lg.QuerySampleResponse(sample.id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)

  def predict(self, input_ids, uncond_input_ids):
    input_ids = shard(input_ids)
    uncond_input_ids = shard(uncond_input_ids)
    prng_seed = jax.random.split(self._prng_seed, jax.local_device_count())

    # pmap the sample function
    images = self.sample(
        input_ids,
        uncond_input_ids,
        prng_seed,
        self._inference_state,
        self._num_inference_steps,
        self._guidance_scale,
    )
    images.block_until_ready()
    return images

def main():
  args = get_args()

  m = StableDiffusionJax(count=4096)

  # warmup
  print("Starting warmup...")
  prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
  input_ids = m.tokenizer(
      [prompt] * args.n_chips, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
  ).input_ids
  uncond_input_ids = m.tokenizer(
      [""] * args.n_chips, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
  ).input_ids
  m.predict(input_ids, uncond_input_ids)
  print("Done with warmup!\n")

  batches = args.batch_sizes
  for batch in batch_sizes:
    print(f"Running loadgen on batch size {batch}")
    log_dir = f'./mlperf_log_outputs/jax/{args.accelerator}/batch{batch}'
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_query_count = 10
    settings.min_duration_ms = 10000

    if args.scenario == 'MultiStream':
      settings.scenario = lg.TestScenario.MultiStream
      settings.multi_stream_target_latency_ns = 100000000
      settings.multi_stream_samples_per_query = batch
      settings.multi_stream_max_async_queries = 2
    elif args.scenario == 'Offline':
      settings.scenario = lg.TestScenario.Offline
      settings.offline_expected_qps = 1000
    else:
      print(f"Scenario {args.scenario} not supported in JAX. Exiting.")
      return

    sut = m.create_sut()
    qsl = m.create_qsl()
    print(f"Starting {args.scenario} test")
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    print("Finish test")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == "__main__":
  main()
