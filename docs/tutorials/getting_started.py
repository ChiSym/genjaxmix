# %% [markdown]
# # Test

# %% [markdown]
# This tutorial introduces Dirichlet Process Mixture Models and explores how to cluster using a simple example.

# %% [markdown]
# # Dataset
# First we need a dataset. In this tutorial, we will synthetically generate points on on the 2D plane and form small clusters that we expect to later detect during inference. We will generate data using a [Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) in pure JAX and later feed in this dataset to GenJAXMix.
#

# %%
import jax
import matplotlib.pyplot as plt

key = jax.random.key(0)
K_max = 5
N = 1000

# Generate cluster proportions
betas = jax.random.beta(key, 1.0, 1.0, shape=(K_max, ))
beta_not = jax.lax.cumprod(1 - betas[:-1])
beta_not = jax.numpy.concatenate([jax.numpy.array([1.0]), beta_not])
weights = betas * beta_not 

# Generate cluster centers
cluster_centers = jax.random.normal(key, (K_max, 2)) 


# %% [markdown]
# We follow [the stick-breaking process](https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process) description for Dirichlet Process Mixture Models. First we generate the stick lengths which represents the proportion of data points expected to belong to each cluster. We then generate the cluster centers - here we use sample the clusters using normal distributions.

# %%

# Sample cluster assignments
cluster_assignments = jax.random.categorical(key, jax.numpy.log(weights), shape=(N,))

# Generate data points
data_points = cluster_centers[cluster_assignments] + jax.random.normal(key, (N, 2))/5

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data_points[:, 0], data_points[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
plt.title('Generated Data Points and Cluster Centers')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# %% [markdown]
# Here we generated 100 data points for five clusters. 

# %% [markdown]
# # Model
# We will now define a Dirichlet Process Mixture Model using GenJAXMix to cluster this dataset.

# %%