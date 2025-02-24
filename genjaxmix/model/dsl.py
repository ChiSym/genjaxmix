from jax import Array
import jax
from plum import dispatch
from abc import abstractmethod, ABC


class Node(ABC):
    shape: list

    def __init__(self):
        self.check_dimensions()

    @abstractmethod
    def parents(self):
        pass

    @abstractmethod
    def initialize(self, key):
        pass

    def check_dimensions(self):
        inputs = self.parents()
        for arg in inputs:
            if len(arg.shape) != 2:
                raise ValueError("All parameters must be 2D arrays")

        shapes = [arg.shape for arg in inputs]
        if not all(shape == self.shape for shape in shapes):
            raise ValueError(
                "All parameters must have the same shape. Received shapes: ", shapes
            )


class Bernoulli(Node):
    def __init__(self, p):
        # if len(p.shape) != 2:
        #     raise ValueError("p must be a 2D array")

        if is_constant(p):
            p = wrap_constant(p)

        self.p = p
        self.shape = p.shape

        super().__init__()

    def parents(self):
        return [self.p]

    def initialize(self, key):
        return jax.random.bernoulli(key, shape=self.shape, p=self.p.value)


class Beta(Node):
    def __init__(self, alpha, beta):
        if len(alpha.shape) != 2:
            raise ValueError("alpha must be a 2D array")
        if len(beta.shape) != 2:
            raise ValueError("beta must be a 2D array")
        if alpha.shape != beta.shape:
            raise ValueError("alpha and beta must have the same shape")

        if is_constant(alpha):
            alpha = wrap_constant(alpha)
        if is_constant(beta):
            beta = wrap_constant(beta)

        self.alpha = alpha
        self.beta = beta
        self.shape = alpha.shape

    def parents(self):
        return [self.alpha, self.beta]

    def initialize(self, key):
        return jax.random.beta(key, self.alpha.value, self.beta.value, shape=self.shape)


class Categorical(Node):
    pass


class Constant(Node):
    def __init__(self, value):
        self.value = value
        self.shape = value.shape

    def parents(self):
        return []

    def initialize(self, key):
        return self.value

    def __repr__(self):
        return f"Constant({self.value})"

    def sample(self, key):
        return self.value


class Dirichlet(Node):
    pass


class Exponential(Node):
    def __init__(self, rate):
        if is_constant(rate):
            rate = wrap_constant(rate)

        self.rate = rate
        self.shape = rate.shape

        super().__init__()

    def parents(self):
        return [self.rate]

    def initialize(self, key):
        if isinstance(self.rate, Constant):
            return jax.random.exponential(key, shape=self.shape) / self.rate.value

    def sample(self, key, rate):
        return jax.random.exponential(key, shape=self.shape) / rate


class Gamma(Node):
    @dispatch
    def __init__(self, alpha, beta):
        if is_constant(alpha):
            alpha = wrap_constant(alpha)
        if is_constant(beta):
            beta = wrap_constant(beta)

        self.alpha = alpha
        self.beta = beta
        self.shape = alpha.shape

        super().__init__()

    @dispatch
    def __init__(self, alpha_and_beta: Node):  # noqa: F811
        self.alpha = alpha_and_beta
        self.beta = alpha_and_beta

    def parents(self):
        return [self.alpha, self.beta]

    def initialize(self, key):
        if isinstance(self.alpha, Constant):
            return jax.random.gamma(key, self.alpha.value) * self.beta.value
        else:
            raise NotImplementedError("WIP: Need to process in topological order")


class InverseGamma(Node):
    def __init__(self, alpha, beta):
        if len(alpha.shape) != 2:
            raise ValueError("alpha must be a 2D array")
        if len(beta.shape) != 2:
            raise ValueError("beta must be a 2D array")
        if alpha.shape != beta.shape:
            raise ValueError("alpha and beta must have the same shape")

        if is_constant(alpha):
            alpha = wrap_constant(alpha)
        if is_constant(beta):
            beta = wrap_constant(beta)

        self.alpha = alpha
        self.beta = beta
        self.shape = alpha.shape

    def parents(self):
        return [self.alpha, self.beta]

    def initialize(self, key):
        if isinstance(self.alpha, Constant):
            return jax.random.gamma(key, self.alpha.value) * self.beta.value
        else:
            raise NotImplementedError("WIP: Need to process in topological order")


class Normal(Node):
    fused: bool

    @dispatch
    def __init__(self, mu, sigma):
        if is_constant(mu):
            mu = wrap_constant(mu)

        if is_constant(sigma):
            sigma = wrap_constant(sigma)

        self.mu = mu
        self.sigma = sigma
        self.shape = mu.shape
        self.is_fused = False

        super().__init__()

    @dispatch
    def __init__(self, mu_and_sigma: Node):  # noqa: F811
        if is_constant(mu_and_sigma):
            raise NotImplementedError("mu_and_sigma must be a 3D array")

        self.mu = mu_and_sigma
        self.sigma = mu_and_sigma
        self.is_fused = True

        super().__init__()

    def parents(self):
        return [self.mu, self.sigma]

    def initialize(self, key):
        if not self.is_fused:
            return jax.random.normal(key, self.mu.shape)

    def __repr__(self):
        return f"Normal({self.mu}, {self.sigma})"

    def sample(self, key, mu, sigma):
        return jax.random.normal(key, shape=self.shape) * sigma + mu


class NormalInverseGamma(Node):
    def __init__(self, alpha, beta, mu, sigma):
        if is_constant(alpha):
            alpha = wrap_constant(alpha)

        if is_constant(beta):
            beta = wrap_constant(beta)

        if is_constant(mu):
            mu = wrap_constant(mu)

        if is_constant(sigma):
            sigma = wrap_constant(sigma)

        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.shape = mu.shape

        super().__init__()

    def parents(self):
        return [self.alpha, self.beta, self.mu, self.sigma]

    def initialize(self, key):
        raise NotImplementedError()


class Poisson(Node):
    def __init__(self, rate):
        if is_constant(rate):
            rate = wrap_constant(rate)

        self.rate = rate
        self.shape = rate.shape

        super().__init__()

    def parents(self):
        return [self.rate]

    def initialize(self, key):
        return jax.random.poisson(key, self.rate.value, shape=self.shape)


class Pareto(Node):
    def __init__(self, concentration, scale):
        if is_constant(concentration):
            concentration = wrap_constant(concentration)

        if is_constant(scale):
            scale = wrap_constant(scale)

        self.concentration = concentration
        self.scale = scale
        self.shape = concentration.shape

        super().__init__()

    def parents(self):
        return [self.concentration, self.scale]

    def initialize(self, key):
        return (
            jax.random.pareto(
                key, self.concentration.value, shape=self.concentration.shape
            )
            * self.scale.value
        )


class Uniform(Node):
    def __init__(self, a, b):
        if is_constant(a):
            a = wrap_constant(a)
        if is_constant(b):
            b = wrap_constant(b)

        self.a = a
        self.b = b
        self.shape = a.shape

        super().__init__()

    def parents(self):
        return [self.a, self.b]

    def initialize(self, key):
        return jax.random.uniform(
            key, self.a.shape, minval=self.a.value, maxval=self.b.value
        )


class Weibull(Node):
    def __init__(self, concentration, scale):
        if len(concentration.shape) != 2:
            raise ValueError("concentration must be a 2D array")
        if len(scale.shape) != 2:
            raise ValueError("scale must be a 2D array")
        if concentration.shape != scale.shape:
            raise ValueError("concentration and scale must have the same shape")

        if is_constant(concentration):
            concentration = wrap_constant(concentration)

        if is_constant(scale):
            scale = wrap_constant(scale)

        self.concentration = concentration
        self.scale = scale
        self.shape = concentration.shape

    def parents(self):
        return [self.concentration, self.scale]

    def initialize(self, key):
        raise NotImplementedError()


def is_constant(obj):
    return (
        isinstance(obj, Array)
        or isinstance(obj, int)
        or isinstance(obj, float)
        or isinstance(obj, Constant)
    )


def wrap_constant(obj):
    if isinstance(obj, Constant):
        return obj
    return Constant(obj)
