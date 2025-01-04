from genjax import gen, categorical, uniform, normal
from genjax import ChoiceMapBuilder
from functools import reduce
import polars as pl
import jax
import numpy as np

@gen
def call_func(type_name, params):
    match type_name:
        case "piecewise_uniform":
            return piecewise_uniform.vmap(in_axes=(0,0))(params["breaks"], params["logp"]) @ type_name
        case "normal":
            return normal.vmap(in_axes=(0,0))(params["mu"], params["sigma"]) @ type_name
        case "categorical":
            return categorical.vmap()(params["logp"]) @ type_name
        case _:
            raise ValueError(f"Unknown type: {type_name}")

@gen
def piecewise_uniform(breaks, logp):
    idx = categorical(logp) @ "bin"
    return uniform(breaks[idx], breaks[idx + 1]) @ "val"

@gen
def mixture_model(pi, cluster_dicts):
    idx = categorical(pi) @ "c"

    cluster_dict = jax.tree.map(lambda x: x[idx], cluster_dicts)
    return cluster.inline(cluster_dict)

@gen
def cluster(cluster_dict):
    return [
        call_func.inline(type_name, params)
        for type_name, params in cluster_dict.items()

    ]

def simulate(key, mm_dict, N=10000):
    # method to efficiently simulate from mixture model
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N)
    idxs = jax.vmap(jax.random.categorical, in_axes=(0, None))(keys, mm_dict["pi"])

    def get_dict(idx, f):
        return jax.tree.map(lambda x: x[idx], f)

    dicts = jax.vmap(get_dict, in_axes=(0, None))(idxs, mm_dict["f"])

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N)
    return jax.jit(jax.vmap(cluster.simulate, in_axes=(0, 0)))(keys, (dicts, ))

def trace_to_polars(tr, schema):
    choices = tr.get_choices()
    dfs = []
    for k, col_names in schema["types"].items():
        if col_names:
            match k:
                case "piecewise_uniform":
                    values = choices.get_submap(k)['val']
                    df = pl.from_numpy(np.array(values), col_names)
                case "categorical":
                    values = choices[k]
                    df = pl.from_numpy(np.array(values), col_names)

                    # TODO do this more efficiently
                    for col in col_names:
                        df = df.with_columns(pl.col(col).cast(pl.Enum(schema["var_metadata"][col]["levels"])).name.keep())
                case "normal":
                    values = choices[k]
                    df = pl.from_numpy(np.array(values), col_names)
                case _:
                    raise ValueError(f"Unknown key: {k}")
            dfs.append(df)

    return pl.concat(dfs, how="horizontal")

def make_chm(constraints, schema):
    chm_list = []
    for k, v in constraints.items():
        k_type = [schema["types"][k_type] for k_type in schema["types"] if k in schema["types"][k_type]][0]
        k_idx = schema["types"][k_type].index(k)
        match k_type:
            case "categorical":
                level_idx = schema["var_metadata"][k]["levels"].index(v)
                chm = ChoiceMapBuilder["categorical", k_idx].set(level_idx)
            case "normal":
                chm = ChoiceMapBuilder["normal", k_idx].set(v)
            case "piecewise_uniform":
                chm = ChoiceMapBuilder["piecewise_uniform", "val", k_idx].set(v)
            case _:
                raise ValueError(f"Unknown variable type: {k_type}")
        chm_list.append(chm)
    return reduce(lambda x, y: x ^ y, chm_list)