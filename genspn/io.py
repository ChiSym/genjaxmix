import polars as pl
import polars.selectors as cs
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Num
from plum import dispatch
import os
from safetensors import safe_open
from safetensors.flax import save_file
from joblib import Memory
from genspn.histogram import histogram

memory = Memory("cachedir")


def dataframe_to_arrays(df: pl.DataFrame):
    schema = make_schema(df)
    categorical_df = df.select(schema["types"]["categorical"])
    numerical_df = df.select(schema["types"]["piecewise_uniform"])

    # def normalize(col: pl.Expr):
    #     return (col - schema["var_metadata"][col.name]["mean"]) / schema["var_metadata"][col.name]["std"]

    # numerical_df = numerical_df.with_columns(
    #     pl.all().map_batches(normalize)
    # )

    numerical_array  = None if numerical_df.is_empty() else jnp.array(numerical_df.to_numpy())
    categorical_arrays, schema = (None, schema) if categorical_df.is_empty() else categorical_df_to_integer(categorical_df, schema)

    return schema, (numerical_array, *categorical_arrays)


def categorical_df_to_integer(df: pl.DataFrame, schema: dict):
    def cast_to_categorical(col: pl.Expr):
        return col.cast(pl.Enum(schema["var_metadata"][col.name]["levels"]))

    df = df.with_columns(pl.all().map_batches(cast_to_categorical))

    array = df.with_columns(pl.all().to_physical()).to_numpy()
    array = jnp.array(array)

    all_n_categories = np.nanmax(array, axis=0)
    dtypes = [get_dtype(n_categories) for n_categories in all_n_categories]
    unique_dtypes = list(set(dtypes))

    arrays = []
    for dtype in unique_dtypes:
        idxs = np.where(np.array(dtypes) == dtype)[0]
        arrays.append(jnp.nan_to_num(array[:, idxs], nan=jnp.iinfo(dtype).max).astype(dtype))

    dtype_idxs = [unique_dtypes.index(dtype) for dtype in dtypes]
    schema["var_metadata"]["categorical_precisions"] = dtype_idxs

    return arrays, schema

def get_dtype(n_categories):
    match n_categories:
        case n_categories if n_categories < jnp.iinfo(jnp.uint4).max:
            dtype = jnp.uint8  # uint4 currently not supported by jax
        case n_categories if n_categories < jnp.iinfo(jnp.uint8).max:
            dtype = jnp.uint8
        case n_categories if n_categories < jnp.iinfo(jnp.uint16).max:
            dtype = jnp.uint16
        case n_categories if n_categories < jnp.iinfo(jnp.uint32).max:
            dtype = jnp.uint32
        case n_categories if n_categories < jnp.iinfo(jnp.uint64).max:
            dtype = jnp.uint64
        case _:
            raise ValueError(n_categories)

    return dtype

def load_huggingface(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-num-missing.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet"
    }
    # train_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}")
    # test_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}")

    train_df = pl.read_parquet(f"../model-building-evaluation/{splits['train']}")
    test_df = pl.read_parquet(f"../model-building-evaluation/{splits['test']}")

    df = pl.concat((train_df, test_df))
    schema, (numerical_array, categorical_array) = dataframe_to_arrays(df)

    n_train = len(train_df)

    if numerical_array is None:
        return schema, (categorical_array[:n_train], categorical_array[n_train:])
    elif categorical_array is None:
        return schema, (numerical_array[:n_train], numerical_array[n_train:])
    else:
        return schema, ((numerical_array[:n_train], categorical_array[:n_train]), (numerical_array[n_train:], categorical_array[n_train:]))


def _get_indices(n, seed: int):
    """"Create a random permutation of indices using the provided seed."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


@dispatch
def split_data(data: tuple[Float[Array, "n n_c"], Integer[Array, "n n_d"]], test_ratio: float = 0.2, seed: int = 42):
    # Unpack the train_data tuple
    data_numerical, data_categorical = data

    # Calculate the number of samples for the train set
    n_samples = data_numerical.shape[0]
    n_train = int((1 - test_ratio) * n_samples)

    # Create a random permutation of indices
    indices = _get_indices(n_samples, seed)

    # Split the numerical data
    train_numerical, test_numerical = data_numerical[indices[:n_train]], data_numerical[indices[n_train:]]

    # Split the categorical data
    train_categorical, test_categorical = data_categorical[indices[:n_train]], data_categorical[indices[n_train:]]

    # Recombine the split data into tuples
    train_data = (train_numerical, train_categorical)
    test_data = (test_numerical, test_categorical)

    return train_data, test_data


@dispatch
def split_data(data: Float[Array, "n n_c"] | Integer[Array, "n n_d"], test_ratio: float = 0.2, seed: int = 42):
    # Calculate the number of samples for the train set (80% of the data)
    n_samples = data.shape[0]
    n_train = int((1 - test_ratio) * n_samples)

    # Create a random permutation of indices
    indices = _get_indices(n_samples, seed)

    # Split the numerical data
    train_data, test_data = data[indices[:n_train]], data[indices[n_train:]]

    return train_data, test_data

# @memory.cache
def get_breaks(arr: Float[Array, "n"]):
    # return bayesian_blocks(arr, fitness='events')
    breaks, _ = histogram(arr, max_bins=20)
    return breaks

def make_schema(df: pl.DataFrame):
    schema = {
        "types":{
            "piecewise_uniform": [],
            "normal": [],
            "categorical": []
        },
        "var_metadata":{}
    }
    key = jax.random.PRNGKey(1234)
    for c in df.columns:
        if df[c].dtype == pl.Utf8:
            schema["types"]["categorical"].append(c)
            schema["var_metadata"][c] = {"levels": df[c].drop_nulls().unique().sort().to_list()}
        elif df[c].dtype == pl.Float64:
            schema["types"]["piecewise_uniform"].append(c)
            breaks = get_breaks(df[c].drop_nulls().to_numpy())
            schema["var_metadata"][c] = {"breaks": breaks}
            if jnp.diff(breaks).min() == 0.:
                import ipdb; ipdb.set_trace()
        else:
            raise ValueError(c)
    return schema


def _assert_keys_mixture(mixture_parameters):
    possible_keys_non_log_prob = {"cluster_weights", "mu", "sigma", "number_different_categorical_types"}
    for key in mixture_parameters.keys():
        assert (key in possible_keys_non_log_prob) or key.startswith("logprob")

def serialize(mixture_parameters, path):
    _assert_keys_mixture(mixture_parameters)
    save_file(mixture_parameters, path)

def deserialize(path):
    mixture_parameters = {}
    with safe_open(path, framework="flax", device="cpu") as f:
        for key in f.keys():
           mixture_parameters[key] = f.get_tensor(key)
    _assert_keys_mixture(mixture_parameters)
    return mixture_parameters
