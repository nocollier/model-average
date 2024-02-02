from enum import Enum
from typing import Any, Callable, Dict, Literal, Union

import matplotlib.pyplot as plt
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


def prior_uniform_power(
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


class Prior(Enum):
    DRS = prior_uniform_drs
    POWER = prior_uniform_power


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

    Note
    ----
    This function is the most flexible but does require that all data be loaded into
    memory prior to calling the function. We will need a version of this where the
    inputs are a dictionary of model objects and a list of variables.

    """
    n_samples = min([len(priors[p]) for p in priors])
    models = inputs.keys()
    variables = list(set([v for _, m in inputs.items() for v in m]))
    post = []
    for i in range(n_samples):  # we could do this in parallel
        mean = {}
        for variable in variables:
            mean[variable] = []
            for model in models:
                mean[variable].append(inputs[model][variable] * priors[model][i])
            mean[variable] = sum(mean[variable])
        post.append(likelihood(mean))
    return np.array(post)


def compute_model_weights(
    inputs: Dict[str, Dict[str, Any]],
    likelihood: Callable,
    prior: Literal[Prior.DRS, Prior.POWER] = Prior.DRS,
    number_samples: int = 100,
    best_percentile: float = 5.0,
    plot_distributions: Union[str, None] = None,
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
    weights
        An array of the mean weighted by the priors and mapped through the likelihood.

    """
    models = list(inputs.keys())
    nmodels = len(models)

    # generate a prior distribution and assign values to each model in a dictionary.
    samples = prior(number_samples, nmodels)
    priors = {model: samples[:, col] for col, model in enumerate(models)}

    # call the function to generate the posterior samples
    posterior = compute_posterior_samples(inputs, priors, likelihood)

    # choose weights to be the mean of the best 5%
    best_indices = np.where(posterior < np.percentile(posterior, best_percentile))
    weights = {m: p[best_indices].mean() for m, p in priors.items()}
    assert np.allclose(sum([w for _, w in weights.items()]), 1)

    # optionally plot prior/posterior
    if plot_distributions is not None:
        fig, axs = plt.subplots(
            figsize=(10, nmodels * 3), nrows=nmodels, ncols=2, tight_layout=True
        )
        for i, m in enumerate(models):
            axs[i, 0].hist(priors[m])
            axs[i, 0].set_title(f"Prior {m}")
            axs[i, 0].set_xlim(0, 1)
            axs[i, 1].hist(priors[m][best_indices])
            axs[i, 1].set_title(f"Posterior {m}")
            axs[i, 1].set_xlim(0, 1)
        fig.savefig(plot_distributions)
        plt.close()

    return weights


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("https://www.ilamb.org/CMIP6/historical/scalar_database.csv")
    df = df[
        (df.ScalarName == "Overall Score")
        & (df.Region == "global")
        & (df.AnalysisType == "MeanState")
        & (df.Model != "MeanCMIP6")
    ]

    # define a likelihood function, 1 - weighted mean overall score
    def likelihood(mean):
        like = 1 - np.sqrt(
            sum(
                [
                    mean[val] ** 2
                    for val in [
                        "GrossPrimaryProductivity",
                        "Evapotranspiration",
                        "Precipitation",
                    ]
                ]
            )
        )
        return like

    # setup the model inputs, normally these would be xarray dataarrays but anything
    # that can be scaled by a float and added together will work.
    models = df.Model.unique()
    inputs = {model: {} for model in models}
    for var, source in zip(
        ["GrossPrimaryProductivity", "Evapotranspiration", "Precipitation"],
        ["FLUXCOM", "GLEAMv3.3a", "GPCPv2.3"],
    ):
        q = df[(df.Variable == var) & (df.Source == source)]
        for model in models:
            inputs[model][var] = float(q[q.Model == model]["Data"].values)

    print(inputs)
    w = compute_model_weights(
        inputs,
        likelihood,
        prior=Prior.POWER,
        number_samples=1000,
        plot_distributions="dist.png",
    )
    print(w)
