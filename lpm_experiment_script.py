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
import pickle
import json
from functools import partial


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

    logpdf_fn = partial(logpdf, mixture_model)
    logprobs = jax.lax.map(logpdf_fn, valid_data, batch_size=1000)

    return mixture_model, time_elapsed, sum_logprobs, trace, logprobs

def run_experiment_wrapper(seed, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_replicates)
    schema, (train_data, valid_data) = load_huggingface(dataset_name)

    train_data, test_data = split_data(train_data, .1, seed=seed)

    # reduce train size
    if "sydt" in dataset_name:
        train_data, _ = split_data(train_data, .9, seed=seed)
    elif "covertype" in dataset_name:
        train_data, _ = split_data(train_data, .9, seed=seed)
    elif "kdd" in dataset_name:
        train_data, _ = split_data(train_data, .9, seed=seed)

    partial_run_exp = partial(run_experiment, max_clusters, gibbs_iters, alpha, d,
        schema, train_data, test_data, valid_data)

    # logprobs = jax.vmap(partial_run_exp, in_axes=(0, None, None, None))(
    #     keys, train_data, test_data, valid_data)

    # logprobs = jax.vmap(partial_run_exp)(keys)
    mixture_model, time_elapsed, sum_logprobs, trace, logprobs = zip(*[partial_run_exp(k) for k in keys])

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

    path = Path("../results", dataset_name, "held-out-likelihood-missing.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    with open(f"../results/{dataset_name}/model-missing.pkl", "wb") as f:
        pickle.dump(mixture_model, f)

    with open(f"../results/{dataset_name}/schema-missing.json", "w") as f:
        for k, v in schema['var_metadata'].items():
            if 'breaks' in v:
                # remove breaks from schema, can't save jnp.array
                # to json and converting to list is lossy
                v['breaks'] = []
        json.dump(schema, f)

login(token=os.environ.get("HF_KEY"))

api = HfApi()
dataset = api.dataset_info("Large-Population-Model/model-building-evaluation")
config = dataset.card_data['configs']


dataset_names = [c['data_files'][0]['path'].rpartition('/')[0] for c in config]
dataset_names = [d for d in dataset_names if ("CTGAN" in d) or ('lpm' in d)]
dataset_names = [d for d in dataset_names if "CTGAN" in d][1:]
# dataset_names = [d for d in dataset_names if ("sydt" in d) or ('covertype' in d)]
print(dataset_names)
# dataset_names = dataset_names[-2:]
n_replicates = 1
seed = 1234
max_clusters = 500
alpha = 2
d = .1
gibbs_iters = 20

for dataset_name in tqdm(dataset_names):
    print(dataset_name)
    run_experiment_wrapper(seed, dataset_name, n_replicates, max_clusters, gibbs_iters, alpha, d)
