import numpy as np
import nengo

theta = 0.5
q = 6

A = np.zeros((q, q))
B = np.zeros((q, 1))
for i in range(q):
    B[i] = (-1.)**i * (2*i+1)
    for j in range(q):
        A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
A = A / theta
B = B / theta

def target(t):
    if 0 <= t < 2.0:
        return 1
    elif 2.0 <= t < 4.0:
        return -1
    else:
        return 0

# Create an array of size (4000,1)
target_values = np.zeros((4000,1))

# Set the first 2000 values to 1
target_values[:2000] = 1

# Set the next 2000 values to -1
target_values[2000:] = -1

with nengo.Network() as model:
    stim = nengo.Node(lambda t: np.sin(2*np.pi*t) if t<2 else np.sin(2*np.pi*t*2))
    lmu = nengo.Ensemble(n_neurons=1000, dimensions=q)
    # nengo.Connection(stim, ens, synapse=0.1)
    nengo.Connection(lmu, lmu, transform=A, synapse=0.1)
    nengo.Connection(stim, lmu, transform=B, synapse=0.1)
    p = nengo.Probe(lmu, synapse=0.01)
    ens2 = nengo.Ensemble(n_neurons=1000, dimensions=q)
    nengo.Connection(lmu, ens2, eval_points=x_values, function=target_values)
with nengo.Simulator(model) as sim:
    sim.run(4)

