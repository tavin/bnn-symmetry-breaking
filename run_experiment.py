from dataclasses import dataclass
from datetime import datetime
from os import path
import pickle
import time

import click
import jax
import numpyro
from numpyro.infer import MCMC, NUTS

import bnn
from dataset import Data, Split, load, normalize


@dataclass
class Trial:
    prng_key: int
    dataset: str
    split: Split
    model: str
    hidden_dim: int
    num_warmup: int
    num_samples: int
    elapsed_time: float
    mcmc: MCMC


@click.command()
@click.option('--outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True), required=True)
@click.option('--dataset', type=str, required=True)
@click.option('--model', type=str, required=True)
@click.option('--hidden_dim', type=int, required=True)
@click.option('--num_warmup', type=int, default=100)
@click.option('--num_samples', type=int, default=100)
@click.option('--num_trials', type=int, default=100)
@click.option('--skip_trials', type=int, default=0)
@click.option('--task_index', type=int, default=0)
@click.option('--task_count', type=int, default=1)
def main(
    outdir: str,
    dataset: str,
    model: str,
    hidden_dim: int,
    num_warmup: int,
    num_samples: int,
    num_trials: int,
    skip_trials: int,
    task_index: int,
    task_count: int,
):
    f_it = '0' + str(len(str(num_trials-1))) + 'd'
    for it in range(skip_trials, num_trials):
        if it % task_count != task_index:
            continue
        outfile = f'{outdir}/{model}.{dataset}.{format(it, f_it)}'
        if path.exists(outfile):
            print(outfile, '✓')
        else:
            print(outfile, '@', datetime.now().replace(microsecond=0))
            trial = run_trial(it, dataset, model, hidden_dim, num_warmup, num_samples)
            with open(outfile, 'wb') as fp:
                pickle.dump(trial, fp)


def run_trial(prng_key, dataset, model, hidden_dim, num_warmup, num_samples) -> Trial:
    mcmc_key, split_key, _next = jax.random.split(jax.random.PRNGKey(prng_key), 3)
    split: Split = Split(0.90, jax.random.randint(split_key, (), 0, 2**32).item())
    data: Data = load(getattr(numpyro.examples.datasets, dataset), split)
    x, *_ = normalize(data.x_train)
    y, *_ = normalize(data.y_train)
    kernel = NUTS(getattr(bnn, model), init_strategy=bnn.init_strategy)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(mcmc_key, x, y, hidden_dim=hidden_dim)
    elapsed_time = time.perf_counter() - t0
    return Trial(prng_key=prng_key, dataset=dataset, split=split, model=model, hidden_dim=hidden_dim,
                 num_warmup=num_warmup, num_samples=num_samples, elapsed_time=elapsed_time, mcmc=mcmc)


if __name__ == '__main__':
    numpyro.util.enable_x64()  # else there will be numerical exceptions
    main()

