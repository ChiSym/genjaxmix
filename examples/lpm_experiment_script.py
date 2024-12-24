import jax
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
from genspn.io import load_huggingface, split_data
from genspn.distributions import make_trace
import jax.numpy as jnp
from genspn.smc import smc
from genspn.distributions import MixtureModel, logpdf
from functools import partial
from pathlib import Path
from huggingface_hub import login, HfApi
from tqdm import tqdm
import os
import numpy as np
import polars as pl
import time


def run_experiment(max_clusters, gibbs_iters, alpha, d, schema, train_data, test_data, valid_data, key):
    key, subkey = jax.random.split(key)
    trace = make_trace(subkey, alpha, d, schema, train_data, max_clusters)

    key, subkey = jax.random.split(key)
    start = time.time()
    trace, sum_logprobs = smc(subkey, trace, test_data, max_clusters, train_data, gibbs_iters, max_clusters)
    time_elapsed = time.time() - start

    idx = jnp.argmax(sum_logprobs)
    print(f"idx: {idx}")
    cluster = trace.cluster[idx]

    mixture_model = MixtureModel(
        pi=cluster.pi/jnp.sum(cluster.pi), 
        f=cluster.f[:max_clusters])

    return time_elapsed, sum_logprobs, trace, jax.vmap(logpdf, in_axes=(None, 0))(mixture_model, (valid_data,))

def run_experiment_wrapper(seed, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_replicates)
    schema, (train_data, valid_data) = load_huggingface(dataset_name)

    train_data, test_data = split_data(train_data, .1, seed=seed)

    # reduce train size
    if "sydt" in dataset_name:
        train_data, _ = split_data(train_data, .9, seed=seed)
    # elif "covertype" in dataset_name:
    #     pass
    # elif "kdd" in dataset_name:
    #     train_data, _ = split_data(train_data, .1, seed=seed)

    partial_run_exp = partial(run_experiment, max_clusters, gibbs_iters, alpha, d,
        schema, train_data, test_data, valid_data)

    # logprobs = jax.vmap(partial_run_exp, in_axes=(0, None, None, None))(
    #     keys, train_data, test_data, valid_data)

    # logprobs = jax.vmap(partial_run_exp)(keys)
    time_elapsed, sum_logprobs, trace, logprobs = zip(*[partial_run_exp(k) for k in keys])

    import ipdb; ipdb.set_trace()

    logprobs = jnp.stack(logprobs)

    data_id = jnp.arange(logprobs.shape[1])
    data_id = jnp.tile(data_id, (n_replicates, 1))

    replicate = jnp.arange(logprobs.shape[0])
    replicate = jnp.tile(replicate, (logprobs.shape[1], 1)).T

    df = pl.DataFrame({
        "data_id": np.array(data_id.flatten()),
        "replicate": np.array(replicate.flatten()),
        "logprob": np.array(logprobs.flatten())
    })

    path = Path("../results_no_rejuvenation", dataset_name, "held-out-likelihood.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

login(token=os.environ.get("HF_KEY"))

api = HfApi()
dataset = api.dataset_info("Large-Population-Model/model-building-evaluation")
config = dataset.card_data['configs']


dataset_names = [c['data_files'][0]['path'].rpartition('/')[0] for c in config]
n_replicates = 1
seed = 1234
max_clusters = 300
alpha = 2
d = .1
gibbs_iters = 20

for dataset_name in tqdm(dataset_names):
    print(dataset_name)
    run_experiment_wrapper(seed, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d)
