Title: Bayesian Regression in PYMC3 using MCMC & Variational Inference
Date: 2018-11-07
Tags: machine-learning, probabilistic-programming, python, pymc3

![jpeg]({filename}/images/data_science/mcmc_vi_pymc3/pymc3_logo.jpg)

Conducting a Bayesian data analysis - e.g. estimating a Bayesian linear regression model - will usually require some form of Probabilistic Programming Language (PPL), unless analytical approaches (e.g. based on conjugate prior models), are appropriate for the task at hand. More often than not, PPLs implement Markov Chain Monte Carlo (MCMC) algorithms that allow one to draw samples and make inferences from the posterior distribution implied by the choice of model - the likelihood and prior distributions for its parameters - conditional on the observed data.

MCMC algorithms are, generally speaking, computationally expensive and do not scale very easily. For example, it is not as easy to distribute the execution of these algorithms over a cluster of machines, when compared to the optimisation algorithms used for training deep neural networks (e.g. stochastic gradient descent).

Over the past few years, however, a new class of algorithms for inferring Bayesian models has been developed, that do **not** rely heavily on computationally expensive random sampling. These algorithms are referred to as Variational Inference (VI) algorithms and have been shown to be successful with the potential to scale to 'large' datasets.

My preferred PPL is [PYMC3](https://docs.pymc.io) and offers a choice of both MCMC and VI algorithms for inferring models in Bayesian data analysis. This blog post is based on a Jupyter notebook located in [this GitHub repository](https://github.com/AlexIoannides/pymc3-advi-hmc-demo), whose purpose is to demonstrate using PYMC3, how MCMC and VI can both be used to perform a simple linear regression, and to make a basic comparison of their results.

## A (very) Quick Introduction to Bayesian Data Analysis

Like statistical data analysis more broadly, the main aim of Bayesian Data Analysis (BDA) is to infer unknown parameters for models of observed data, in order to test hypotheses about the physical processes that lead to the observations. Bayesian data analysis deviates from traditional statistics - on a practical level - when it comes to the explicit assimilation of prior knowledge regarding the uncertainty of the model parameters, into the statistical inference process and overall analysis workflow. To this end, BDA focuses on the posterior distribution,

$$
p(\Theta | X) = \frac{p(X | \Theta) \cdot p(\Theta)}{p(X)}
$$

Where,

- $\Theta$ is the vector of unknown model parameters, that we wish to estimate; 
- $X$ is the vector of observed data;
- $p(X | \Theta)$ is the likelihood function that models the probability of observing the data for a fixed choice of parameters; and,
- $p(\Theta)$ is the prior distribution of the model parameters.

For an **excellent** (inspirational) introduction to practical BDA, take a look at [Statistical Rethinking by Richard McElreath](https://xcelab.net/rm/statistical-rethinking/), or for a more theoretical treatment try [Bayesian Data Analysis by Gelman & co.](http://www.stat.columbia.edu/~gelman/book/).

This notebook is concerned with demonstrating and comparing two separate approaches for inferring the posterior distribution, $p(\Theta | X)$, for a linear regression model.

## Imports and Global Settings

Before we get going in earnest, we follow the convention of declaring all imports at the top of the notebook.

```python
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano
import warnings
from numpy.random import binomial, randn, uniform
from sklearn.model_selection import train_test_split
```

And then notebook-wide (global) settings that enable in-line plotting, configure Seaborn for visualisation and to explicitly ignore warnings (e.g. NumPy deprecations).

```python
%matplotlib inline

sns.set()
warnings.filterwarnings('ignore')
```

## Create Synthetic Data

We will assume that there is a dependent variable (or labelled data) $\tilde{y}$, that is a linear function of independent variables (or feature data), $x$ and $c$. In this instance, $x$ is a positive real number and $c$ denotes membership to one of two categories that occur with equal likelihood. We express this model mathematically, as follows,

$$
\tilde{y} = \alpha_{c} + \beta_{c} \cdot x + \sigma \cdot \tilde{\epsilon}
$$

where $\tilde{\epsilon} \sim N(0, 1)$, $\sigma$ is the standard deviation of the noise in the data and $c \in \{0, 1\}$ denotes the category. We start by defining our *a priori* choices for the model parameters.

```python
alpha_0 = 1
alpha_1 = 1.25

beta_0 = 1
beta_1 = 1.25

sigma = 0.75
```

We then use these to generate some random samples that we store in a DataFrame and visualise using the Seaborn package.

```python
n_samples = 1000

category = binomial(n=1, p=0.5, size=n_samples)
x = uniform(low=0, high=10, size=n_samples)

y = ((1 - category) * alpha_0 + category * alpha_1
     + ((1 - category) * beta_0 + category * beta_1) * x
     + sigma * randn(n_samples))

model_data = pd.DataFrame({'y': y, 'x': x, 'category': category})

display(model_data.head())
_ = sns.relplot(x='x', y='y', hue='category', data=model_data)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>x</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.429483</td>
      <td>2.487456</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.987868</td>
      <td>5.801619</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.340802</td>
      <td>3.046879</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.826015</td>
      <td>6.172437</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.659304</td>
      <td>9.829751</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_9_1.png)

## Split Data into Training and Test Sets

One of the advantages of generating synthetic data is that we can ensure we have enough data to be able to partition it into two sets - one for training models and one for testing models. We use a helper function from the Scikit-Learn package for this task and make use of stratified sampling to ensure that we have a balanced representation of each category in both training and test datasets.

```python
train, test = train_test_split(
    model_data, test_size=0.2, stratify=model_data.category)
```

We will be using the [PYMC3](https://docs.pymc.io) package for building and estimating our Bayesian regression models, which in-turn uses the Theano package as a computational 'back-end' (in much the same way that the Keras package for deep learning uses TensorFlow as back-end). Consequently, we will have to interact with Theano if we want to have the ability to swap between training and test data (which we do). As such, we will explicitly define 'shared' tensors for all of our model variables.

```python
y_tensor = theano.shared(train.y.values.astype('float64'))
x_tensor = theano.shared(train.x.values.astype('float64'))
cat_tensor = theano.shared(train.category.values.astype('int64'))
```

## Define Bayesian Regression Model

Now we move on to define the model that we want to estimate (i.e. our hypothesis regarding the data), irrespective of how we will perform the inference. We will assume full knowledge of the data-generating model we defined above and define conservative regularising priors for each of the model parameters.

```python
with pm.Model() as model:
    alpha_prior = pm.HalfNormal('alpha', sd=2, shape=2)
    beta_prior = pm.Normal('beta', mu=0, sd=2, shape=2)
    sigma_prior = pm.HalfNormal('sigma', sd=2, shape=1)
    mu_likelihood = alpha_prior[cat_tensor] + beta_prior[cat_tensor] * x_tensor
    y_likelihood = pm.Normal('y', mu=mu_likelihood, sd=sigma_prior, observed=y_tensor)
```

## Model Inference Using MCMC (HMC)

We will make use of the default MCMC method in PYMC3's `sample` function, which is Hamiltonian Monte Carlo (HMC). Those interested in the precise details of the HMC algorithm are directed to the [excellent paper Michael Betancourt](https://arxiv.org/abs/1701.02434). Briefly, MCMC algorithms work by defining multi-dimensional Markovian stochastic processes, that when simulated (using Monte Carlo methods), will eventually converge to a state where successive simulations will be equivalent to drawing random samples from the posterior distribution of the model we wish to estimate.

The posterior distribution has one dimension for each model parameter, so we can then use the distribution of samples for each parameter to infer the range of possible values and/or compute point estimates (e.g. by taking the mean of all samples).

For the purposes of this demonstration, we sample two chains in parallel (as we have two CPU cores available for doing so and this effectively doubles the number of samples), allow 1,000 steps for each chain to converge to its steady-state and then sample for a further 5,000 steps - i.e. generate 5,000 samples from the posterior distribution, assuming that the chain has converged after 1,000 samples.

```python
with model:
    hmc_trace = pm.sample(draws=5000, tune=1000, cores=2)
```

Now let's take a look at what we can infer from the HMC samples of the posterior distribution.

```python
pm.traceplot(hmc_trace)
pm.summary(hmc_trace)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta__0</th>
      <td>1.002347</td>
      <td>0.013061</td>
      <td>0.000159</td>
      <td>0.977161</td>
      <td>1.028955</td>
      <td>5741.410305</td>
      <td>0.999903</td>
    </tr>
    <tr>
      <th>beta__1</th>
      <td>1.250504</td>
      <td>0.012084</td>
      <td>0.000172</td>
      <td>1.226709</td>
      <td>1.273830</td>
      <td>5293.506143</td>
      <td>1.000090</td>
    </tr>
    <tr>
      <th>alpha__0</th>
      <td>0.989984</td>
      <td>0.073328</td>
      <td>0.000902</td>
      <td>0.850417</td>
      <td>1.141318</td>
      <td>5661.466167</td>
      <td>0.999900</td>
    </tr>
    <tr>
      <th>alpha__1</th>
      <td>1.204203</td>
      <td>0.069373</td>
      <td>0.000900</td>
      <td>1.069428</td>
      <td>1.339139</td>
      <td>5514.158012</td>
      <td>1.000004</td>
    </tr>
    <tr>
      <th>sigma__0</th>
      <td>0.734316</td>
      <td>0.017956</td>
      <td>0.000168</td>
      <td>0.698726</td>
      <td>0.768540</td>
      <td>8925.864908</td>
      <td>1.000337</td>
    </tr>
  </tbody>
</table>
</div>

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_19_1.png)

Firstly, note that `Rhat` values (the Gelman Rubin statistic) converging to 1 implies chain convergence for the marginal parameter distributions, while `n_eff` describes the effective number of samples after autocorrelations in the chains have been accounted for. We can see from the `mean` (point) estimate of each parameter that HMC has done a reasonable job of estimating our original parameters.

## Model Inference using Variational Inference (mini-batch ADVI)

Variational Inference (VI) takes a completely different approach to inference. Briefly, VI is a name for a class of algorithms that seek to fit a chosen class of functions to approximate the posterior distribution, effectively turning inference into an optimisation problem. In this instance VI minimises the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) (a measure of the 'similarity' between two densities), between the approximated posterior density and the actual posterior density. An excellent review of VI can be found in the [paper by Blei & co.](https://arxiv.org/abs/1601.00670).

Just to make things more complicated (and for this description to be complete), the KL divergence is actually minimised, by maximising the Evidence Lower BOund (ELBO), which is equal to the negative of the KL divergence up to a constant term - a constant that is computationally infeasible to compute, which is why, technically, we are optimising ELBO and not the KL divergence, albeit to achieve the same end-goal.

We are going to make use of PYMC3's Auto-Differentiation Variational Inference (ADVI) algorithm (full details in the paper by [Kucukelbir & co.](https://arxiv.org/abs/1603.00788)), which is capable of computing a VI for any differentiable posterior distribution (i.e. any model with continuous prior distributions). In order to achieve this very clever feat (the paper is well-worth a read), the algorithm first maps the posterior into a space where all prior distributions have the same support, such that they can be well approximated by fitting a spherical n-dimensional Gaussian distribution within this space - this is referred to as the 'Gaussian mean-field approximation'. Note, that due to the initial transformation, this is **not** the same as approximating the posterior distribution using an n-dimensional Normal distribution. The parameters of these Gaussian parameters are then chosen to maximise the ELBO using gradient ascent - i.e. using high-performance auto-differentiation techniques in numerical computing back-ends such as Theano, TensorFlow, etc..

The assumption of a spherical Gaussian distribution does, however, imply no dependency (i.e. zero correlations) between parameter distributions. One of the advantages of HMC over ADVI, is that these correlations, which can lead to under-estimated variances in the parameter distributions, are included. ADVI gives these up in the name of computational efficiency (i.e. speed and scale of data). This simplifying assumption can be dropped, however, and PYMC3 does offer the option to use 'full-rank' Gaussians, but I have not used this in anger (yet).

We also take the opportunity to make use of PYMC3's ability to compute ADVI using 'batched' data, analogous to how Stochastic Gradient Descent (SGD) is used to optimise loss functions in deep-neural networks, which further facilitates model training at scale thanks to the reliance on auto-differentiation and batched data, which can also be distributed across CPU (or GPUs).

In order to enable mini-batch ADVI, we first have to setup the mini-batches (we use batches of 100 samples).


```python
map_tensor_batch = {y_tensor: pm.Minibatch(train.y.values, 100),
                    x_tensor: pm.Minibatch(train.x.values, 100),
                    cat_tensor: pm.Minibatch(train.category.values, 100)}
```

We then compute the variational inference using 30,000 iterations (for the gradient ascent of the ELBO). We use the `more_replacements` key-word argument to swap-out the original Theano tensors with the batched versions defined above.

```python
with model:
    advi_fit = pm.fit(method=pm.ADVI(), n=30000,
                      more_replacements=map_tensor_batch)
```

Before we take a look at the parameters, let's make sure the ADVI fit has converged by plotting ELBO as a function of the number of iterations.

```python
advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(advi_fit.hist),
     'n': np.arange(advi_fit.hist.shape[0])})

_ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
```

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_27_0.png)

In order to be able to look at what we can infer from posterior distribution we have fit with ADVI, we first have to draw some samples from it, before summarising like we did with HMC inference.

```python
advi_trace = advi_fit.sample(10000)
pm.traceplot(advi_trace)
pm.summary(advi_trace)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta__0</th>
      <td>1.000717</td>
      <td>0.022073</td>
      <td>0.000220</td>
      <td>0.957703</td>
      <td>1.044096</td>
    </tr>
    <tr>
      <th>beta__1</th>
      <td>1.250904</td>
      <td>0.020917</td>
      <td>0.000206</td>
      <td>1.209715</td>
      <td>1.292017</td>
    </tr>
    <tr>
      <th>alpha__0</th>
      <td>0.984404</td>
      <td>0.122010</td>
      <td>0.001109</td>
      <td>0.755816</td>
      <td>1.230404</td>
    </tr>
    <tr>
      <th>alpha__1</th>
      <td>1.192829</td>
      <td>0.120833</td>
      <td>0.001146</td>
      <td>0.966362</td>
      <td>1.433906</td>
    </tr>
    <tr>
      <th>sigma__0</th>
      <td>0.760702</td>
      <td>0.060009</td>
      <td>0.000569</td>
      <td>0.649582</td>
      <td>0.883380</td>
    </tr>
  </tbody>
</table>
</div>

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_29_1.png)

Not bad! The mean estimates are comparable, but we note that the standard deviations appear to be larger than those estimated with HMC.

## Comparing Predictions

Let's move on to comparing the inference algorithms on the practical task of making predictions on our test dataset. We start by swapping the test data into our Theano variables.

```python
y_tensor.set_value(test.y.values)
x_tensor.set_value(test.x.values)
cat_tensor.set_value(test.category.values.astype('int64'))
```

And then drawing posterior-predictive samples for each new data-point, for which we use the mean as the point estimate to use for comparison.

```python
hmc_posterior_pred = pm.sample_ppc(hmc_trace, 1000, model)
hmc_predictions = np.mean(hmc_posterior_pred['y'], axis=0)

advi_posterior_pred = pm.sample_ppc(advi_trace, 1000, model)
advi_predictions = np.mean(advi_posterior_pred['y'], axis=0)

prediction_data = pd.DataFrame(
    {'HMC': hmc_predictions, 
     'ADVI': advi_predictions, 
     'actual': test.y,
     'error_HMC': hmc_predictions - test.y, 
     'error_ADVI': advi_predictions - test.y})

_ = sns.lmplot(y='ADVI', x='HMC', data=prediction_data,
               line_kws={'color': 'red', 'alpha': 0.5})
```

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_34_1.png)

As we might expect, given the parameter estimates, the two models generate similar predictions. 

To begin to get an insight into the differences between HMC and ADVI, we look at the inferred dependency structure between the samples of `alpha_0` and `beta_0`, for both HMC and VI, starting with HMC.

```python
param_samples_HMC = pd.DataFrame(
    {'alpha_0': hmc_trace.get_values('alpha')[:, 0], 
     'beta_0': hmc_trace.get_values('beta')[:, 0]})

_ = sns.scatterplot(x='alpha_0', y='beta_0', data=param_samples_HMC).set_title('HMC')
```

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_36_0.png)

And again for ADVI.

```python
param_samples_ADVI = pd.DataFrame(
    {'alpha_0': advi_trace.get_values('alpha')[:, 0], 
     'beta_0': advi_trace.get_values('beta')[:, 0]})

_ = sns.scatterplot(x='alpha_0', y='beta_0', data=param_samples_ADVI).set_title('ADVI')
```

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_38_0.png)

We can see clearly the impact of ADVI's assumption of n-dimensional spherical Gaussians, manifest in the inference!

Finally, let's compare predictions with the actual data.

```python
RMSE = np.sqrt(np.mean(prediction_data.error_ADVI ** 2))

print(f'RMSE for ADVI predictions = {RMSE:.3f}')

_ = sns.lmplot(y='ADVI', x='actual', data=prediction_data, 
               line_kws={'color': 'red', 'alpha': 0.5})
```

    RMSE for ADVI predictions = 0.746

![png]({filename}/images/data_science/mcmc_vi_pymc3/output_40_1.png)

Which is what one might expect, given the data generating model.

## Conclusions

MCMC and VI present two very different approaches for drawing inferences from Bayesian models. Despite these differences, their high-level output for a simplistic (but not entirely trivial) regression problem, based on synthetic data, is comparable regardless of the approximations used within ADVI. This is important to note, because general purpose VI algorithms such as ADVI have the potential to work at scale - on large volumes of data in a distributed computing environment (see the references embedded above, for case studies).
