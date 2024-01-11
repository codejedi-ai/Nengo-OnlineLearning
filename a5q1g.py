import nengo

# import matplotlib.pyplot as plt

# 

# Create the model
def choice(x):
    if x[0] > 0.9:
        return 1
    elif x[0] < -0.9:
        return -1
    else:
        return 0
model = nengo.Network()

with model:

    # Create an ensemble with 50 neurons

    accum = nengo.Ensemble(n_neurons=50, dimensions=1)
    output = nengo.Ensemble(n_neurons=50, dimensions=1, intercepts = nengo.dists.Uniform(0.4, 0.9))
    # acc.noise = nengo.processes.WhiteSignal(period=10, high=100, rms=1)

    # Create an input node with a constant value of 0.1

    input_node = nengo.Node(output=[0])
    # Connect the input node to the ensemble
    nengo.Connection(input_node, accum)

    # Connect the ensemble back to itself with a synapse of 0.1
    nengo.Connection(accum, accum, synapse=0.1)
    nengo.Connection(accum, output, function=choice)
    probe = nengo.Probe(output, synapse=0.01)
    probe_accum = nengo.Probe(accum, synapse=0.01)