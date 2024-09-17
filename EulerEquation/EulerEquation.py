import jax
import jax.numpy as jnp
import equinox as eqx

class Euler1D(eqx.Module):
    gamma: float = 1.4
    NN: eqx.Module

    # Initialize the Neural Network
    def __init__(self, key, nNodes, nhiddenLayer, gamma=1.4):
        key, skey = jax.random.split(key)
        self.NN = eqx.nn.MLP(
            in_size=2,   # Input (x, t)
            out_size=3,  # Output (rho, rhoU, E)
            width_size=nNodes,
            depth=nhiddenLayer,
            activation=jax.nn.tanh,
            key=skey,
            dtype=jax.numpy.float32)
        self.gamma = gamma
    
    # Return the value of the Neural Network
    def __call__(self, x, t):
        return self.NN(jnp.array([x, t]))
    
    # Find Flux for conserved variable (Q)
    def flux(self, Q):
        rho, rho_u, E = Q
        u = rho_u / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        return jnp.array([rho_u, rho_u * u + p, u * (E + p)])
    
    # Convert conserved variable to primitive varible
    def conservedToPrimitive(self, Q):
        rho, rho_u, E = Q
        u = rho_u / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * u**2)
        return jnp.array([rho, u, p])
    
    # Convert primitive variable to conserved varible
    def primitiveToConserved(self, W):
        rho, u, p = W
        rho_u = rho*u
        E = rho*(p/(rho*(self.gamma - 1.0)) + 0.5*u**2)
        return jnp.array([rho, rho_u, E])

    # Euler Equation in terms of conserved variable --- returns residual
    def PDE(self, X, T):
        Q = self(X, T)

        Q_t = jax.jacfwd(self, argnums=1)(X, T)
        Q_x = jax.jacfwd(self, argnums=0)(X, T)
        A = jax.jacfwd(self.flux)(Q)

        return Q_t + A@Q_x 
     
