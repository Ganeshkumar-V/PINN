import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

class Poisson1D(eqx.Module):
    NN: eqx.Module
    SourceFunction: lambda x: x

    # Initialize the Neural Network
    def __init__(self, key, nNodes, nhiddenLayer, Source):
        key, skey = jax.random.split(key)
        self.NN = eqx.nn.MLP(
            in_size=1,
            out_size="scalar",
            width_size=nNodes,
            depth=nhiddenLayer,
            activation=jax.nn.sigmoid,
            key=skey,
            dtype=jax.numpy.float32)
        self.SourceFunction = Source

    # Return the value of the Neural Network -- u(x)
    def __call__(self, x):
        return self.NN(jnp.array([x]))

    # Return residual of the Poisson equation
    def PDE(self, X):
        u = self.__call__(X)

        # Find the partial derivatives
        u_x = jax.grad(lambda x: self(x), argnums=0)
        u_xx = jax.grad(u_x, argnums=0)

        # Return the residual of the Poisson equation
        return u_xx(X) + self.SourceFunction(X)

    # Plot the solution contour
    def plot(self):
        # Plot the solution contour
        x = jnp.linspace(0, 1, 50)

        # Evaluate the Poisson network for all x values
        usol = jnp.zeros_like(x)
        for i in range(x.shape[0]):
            usol = usol.at[i].set(self(x[i]))

        # Plot the contour
        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Plot
        ax.plot(x, usol, label='u(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.set_title('Plot of Poisson Equation Solution')
        ax.legend()

        plt.tight_layout()
        plt.show()
