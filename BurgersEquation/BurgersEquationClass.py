import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

class Burgers1D(eqx.Module):
    NN: eqx.Module
    nu: float

    # Initialize the Neural Network
    def __init__(self, key, nNodes, nhiddenLayer, nu=0.01):
        key, skey = jax.random.split(key)
        self.NN = eqx.nn.MLP(
            in_size=2,
            out_size="scalar",
            width_size=nNodes,
            depth=nhiddenLayer,
            activation=jax.nn.tanh,
            key=skey,
            dtype=jax.numpy.float32)
        self.nu = nu
    
    # Return the value of the Neural Network
    def __call__(self, x, t):
        return self.NN(jnp.array([x, t]))

    # Return residual of the PDE
    def PDE(self, X, T):
        u = self(X, T)

        # Find the partial derivatives
        u_x, u_t = jax.jacrev(self, argnums=(0, 1))(X, T)
        u_xx = jax.jacrev(jax.jacrev(self, argnums=0), argnums=0)(X, T)

        # Return the pde loss
        return u_t + u*u_x - self.nu*u_xx
    
    # Plot the solution contour
    def plot(self):
        # Plot the solution contour
        x = jnp.linspace(-1, 1, 50)
        t = jnp.linspace(0, 1, 50)

        # Create meshgrid for x and t
        X, T = jnp.meshgrid(x, t)

        # Evaluate the Burgers network for all (x, t) pairs
        usol = jnp.zeros_like(X)
        for i in range(x.shape[0]):
            usol = usol.at[i, :].set(self(X[i, :], T[i, :]))

        # Plot the contour
        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Contour plot
        contour = ax.contourf(T, X, usol, levels=50, cmap='rainbow')
        fig.colorbar(contour, ax=ax, label='u(x,t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_title('Contour Plot of Burgers Equation Solution')

        plt.tight_layout()
        plt.show() 
