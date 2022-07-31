from functools import partial

from jax import jit, random
from jax import nn as jnn, numpy as jnp
import numpyro
from numpyro.distributions import Gamma, Normal
from numpyro.distributions import constraints, Distribution, ImproperUniform
from numpyro.infer.initialization import init_to_uniform


def Improper(support, shape):
    return ImproperUniform(support, batch_shape=tuple(shape), event_shape=())


# noinspection PyAbstractClass
class StdNormal(Distribution):

    support = constraints.real

    def __init__(self, shape=()):
        super().__init__(batch_shape=tuple(shape))

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return random.normal(key, sample_shape + self.batch_shape)

    @staticmethod
    @jit
    def log_prob(value):
        return - jnp.log(2*jnp.pi)/2. - jnp.square(value)/2.

    @staticmethod
    @jit
    def log_prob_of_square(value):
        return - jnp.log(2*jnp.pi)/2. - value/2.


def init_strategy(site=None):
    if site is None:
        return partial(init_strategy)
    init = init_to_uniform(site, radius=1.)
    # rescale strategy here
    return init


def sorted_layer(w):
    r = jnp.square(w).sum(axis=0)
    s = jnp.cumsum(r)
    return w * jnp.sqrt(s / r)


def efficient_model(x, y=None, *, hidden_dim):

    scale_nn = 1. / jnp.sqrt(numpyro.sample('prec_nn', Gamma(1.)))

    b1 = scale_nn * numpyro.sample('nn_l1_bias', StdNormal([hidden_dim]))
    w1 = numpyro.sample('nn_l1_unscaled', Improper(constraints.real, (x.shape[-1], hidden_dim)))
    r1 = jnp.square(w1).sum(axis=-2)
    s1 = jnp.cumsum(r1, axis=-1)
    numpyro.factor('nn_l1_factor', StdNormal.log_prob_of_square(s1))
    w1 = w1 * jnp.sqrt(s1 / r1) * scale_nn

    b2 = scale_nn * numpyro.sample('nn_l2_bias', StdNormal([hidden_dim]))
    w2 = numpyro.sample('nn_l2_unscaled', Improper(constraints.real, (hidden_dim, hidden_dim)))
    r2 = jnp.square(w2).sum(axis=-2)
    s2 = jnp.cumsum(r2, axis=-1)
    numpyro.factor('nn_l2_factor', StdNormal.log_prob_of_square(s2))
    w2 = w2 * jnp.sqrt(s2 / r2) * scale_nn

    b3 = scale_nn * numpyro.sample('nn_l3_bias', StdNormal())
    w3 = scale_nn * numpyro.sample('nn_l3_weight', StdNormal([hidden_dim]))

    _model(x, y, b1, w1, b2, w2, b3, w3)


def improper_model(x, y=None, *, hidden_dim):

    log_prob = Normal().log_prob
    scale_nn = 1. / jnp.sqrt(numpyro.sample('prec_nn', Gamma(1.)))

    b1 = numpyro.sample('nn_l1_bias', ImproperUniform(constraints.real, (), (hidden_dim,)))
    w1 = numpyro.sample('nn_l1_unscaled', ImproperUniform(constraints.real, (), (x.shape[-1], hidden_dim)))
    w1 = numpyro.deterministic('nn_l1_weight', sorted_layer(w1))
    numpyro.factor('nn_l1_bias_factor', log_prob(b1))
    numpyro.factor('nn_l1_weight_factor', log_prob(w1))
    b1 = scale_nn * b1
    w1 = scale_nn * w1

    b2 = numpyro.sample('nn_l2_bias', ImproperUniform(constraints.real, (), (hidden_dim,)))
    w2 = numpyro.sample('nn_l2_unscaled', ImproperUniform(constraints.real, (), (hidden_dim, hidden_dim)))
    w2 = numpyro.deterministic('nn_l2_weight', sorted_layer(w2))
    numpyro.factor('nn_l2_bias_factor', log_prob(b2))
    numpyro.factor('nn_l2_weight_factor', log_prob(w2))
    b2 = scale_nn * b2
    w2 = scale_nn * w2

    b3 = numpyro.sample('nn_l3_bias', ImproperUniform(constraints.real, (), ()))
    w3 = numpyro.sample('nn_l3_weight', ImproperUniform(constraints.real, (), (hidden_dim,)))
    numpyro.factor('nn_l3_bias_factor', log_prob(b3))
    numpyro.factor('nn_l3_weight_factor', log_prob(w3))
    b3 = scale_nn * b3
    w3 = scale_nn * w3

    _model(x, y, b1, w1, b2, w2, b3, w3)


def dumb_model(x, y=None, *, hidden_dim):

    log_prob = Normal().log_prob
    scale_nn = 1. / jnp.sqrt(numpyro.sample('prec_nn', Gamma(1.)))

    b1 = numpyro.sample('nn_l1_bias', ImproperUniform(constraints.real, (), (hidden_dim,)))
    w1 = numpyro.sample('nn_l1_unscaled', ImproperUniform(constraints.real, (), (x.shape[-1], hidden_dim)))
    w1 = numpyro.deterministic('nn_l1_weight', w1)
    numpyro.factor('nn_l1_bias_factor', log_prob(b1))
    numpyro.factor('nn_l1_weight_factor', log_prob(w1))
    b1 = scale_nn * b1
    w1 = scale_nn * w1

    b2 = numpyro.sample('nn_l2_bias', ImproperUniform(constraints.real, (), (hidden_dim,)))
    w2 = numpyro.sample('nn_l2_unscaled', ImproperUniform(constraints.real, (), (hidden_dim, hidden_dim)))
    w2 = numpyro.deterministic('nn_l2_weight', w2)
    numpyro.factor('nn_l2_bias_factor', log_prob(b2))
    numpyro.factor('nn_l2_weight_factor', log_prob(w2))
    b2 = scale_nn * b2
    w2 = scale_nn * w2

    b3 = numpyro.sample('nn_l3_bias', ImproperUniform(constraints.real, (), ()))
    w3 = numpyro.sample('nn_l3_weight', ImproperUniform(constraints.real, (), (hidden_dim,)))
    numpyro.factor('nn_l3_bias_factor', log_prob(b3))
    numpyro.factor('nn_l3_weight_factor', log_prob(w3))
    b3 = scale_nn * b3
    w3 = scale_nn * w3

    _model(x, y, b1, w1, b2, w2, b3, w3)


def reference_model(x, y=None, *, hidden_dim):

    scale_nn = 1. / jnp.sqrt(numpyro.sample('prec_nn', Gamma(1.)))

    b1 = scale_nn * numpyro.sample('nn_l1_bias', Normal(jnp.zeros([hidden_dim])))
    w1 = scale_nn * numpyro.sample('nn_l1_weight', Normal(jnp.zeros([x.shape[-1], hidden_dim])))

    b2 = scale_nn * numpyro.sample('nn_l2_bias', Normal(jnp.zeros([hidden_dim])))
    w2 = scale_nn * numpyro.sample('nn_l2_weight', Normal(jnp.zeros([hidden_dim, hidden_dim])))

    b3 = scale_nn * numpyro.sample('nn_l3_bias', Normal(0))
    w3 = scale_nn * numpyro.sample('nn_l3_weight', Normal(jnp.zeros([hidden_dim])))

    _model(x, y, b1, w1, b2, w2, b3, w3)


def _model(x0, y, b1, w1, b2, w2, b3, w3):
    x1 = jnn.relu(x0 @ w1 + b1)
    x2 = jnn.relu(x1 @ w2 + b2)
    x3 = x2 @ w3 + b3
    scale_obs = 1. / jnp.sqrt(numpyro.sample('prec_obs', Gamma(30.)))
    with numpyro.plate('data', len(x0)):
        numpyro.sample('y', Normal(x3, scale_obs), obs=y)
    numpyro.deterministic('stat_l1_norm', jnp.sqrt(jnp.square(w1).sum(axis=0)))
    numpyro.deterministic('stat_l2_norm', jnp.sqrt(jnp.square(w2).sum(axis=0)))
    numpyro.deterministic('stat_l3_norm', jnp.sqrt(jnp.square(w3).sum()))
