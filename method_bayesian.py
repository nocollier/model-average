from typing import Any, Callable, Dict

import numpy as np
from drs import drs
from numpy.typing import NDArray


def prior_uniform_drs(n_samples: int, n_models: int):
    """Return uniform prior using the Dirichlet-rescale algorithm.

    Parameters
    ----------
    n_samples
        The number of samples to generate
    n_models
        The number of models (components)

    See Also
    --------
    https://github.com/dgdguk/drs

    """
    prior = np.array(
        [
            drs(n_models, 1.0, np.ones(n_models), np.zeros(n_models))
            for i in range(n_samples)
        ]
    )
    return prior


def prior_uniform_elias(
    n_samples: int, n_models: int, threshold: float = 0.95, max_power: int = 20
):
    """Return uniform prior using Elias' method.

    Produces uniform samples by rescaling and applies a power until a sample of at least
    `threshold` is seen. This is to make sure we test if one single model is dominant to
    all others.

    """

    def _normalize(a):
        return a / a.sum(axis=1)[:, np.newaxis]

    prior = np.random.uniform(size=(n_samples, n_models))
    pwr = 1
    while _normalize(prior**pwr).max() < threshold and pwr < max_power:
        pwr += 1
    return _normalize(prior**pwr)


def compute_posterior_samples(
    inputs: Dict[str, Dict[str, Any]], priors: Dict[str, NDArray], likelihood: Callable
):
    """Return

    Parameters
    ----------
    inputs
        A nested dictionary of modeled inputs to blend where the top level consists of
        model keys and nested levels are variables.
    priors
        A dictionary of whose keys represent the models.
    likelihood
        A function which returns a scalar value given a dictionary of mean inputs.

    Returns
    -------
    posterior
        An array of the mean weighted by the priors and mapped through the likelihood.

    """
    n_samples = min([len(priors[p]) for p in priors])
    models = inputs.keys()
    variables = list(set([v for _, m in inputs.items() for v in m]))
    post = []
    for i in range(n_samples):
        mean = {}
        for variable in variables:
            mean[variable] = []
            for model in models:
                mean[variable].append(inputs[model][variable] * priors[model][i])
            mean[variable] = sum(mean[variable])
        post.append(likelihood(mean))
    return np.array(post)


if __name__ == "__main__":
    # define a likelihood function
    def rmse(mean):
        obs = {"tas": 1.29, "pr": 2.61}  # <-- data fabricated to favor CESM2
        return np.sqrt(sum([(mean[v] - obs[v]) ** 2 for v in mean]))

    # setup the model inputs, normally these would be xarray dataarrays but anything
    # that can be scaled by a float and added together will work.
    inputs = {
        "CESM2": {"tas": 1.3, "pr": 2.6},
        "E3SM": {"tas": 1.1, "pr": 2.8},
    }
    models = list(inputs.keys())

    # generate a prior distribution and assign values to each model in a dictionary.
    samples = prior_uniform_drs(10000, len(models))
    priors = {model: samples[:, col] for col, model in enumerate(models)}

    # call the function to generate the posterior samples
    posterior = compute_posterior_samples(inputs, priors, rmse)

    # choose weights to be the mean of the best 5%
    best_indices = np.where(posterior < np.percentile(posterior, 5))
    weights = {m: p[best_indices].mean() for m, p in priors.items()}
    print(weights)
