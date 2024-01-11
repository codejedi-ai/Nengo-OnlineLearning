import nengo

import numpy as np
T = 30.0
learning_rate = 0.5e-4
f=lambda x: [x ** 2,-x]
with nengo.Network() as model:
    nd_stim = nengo.Node(nengo.processes.WhiteSignal(
        period=T, high=1.0, rms=0.5))
    ens_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_y = nengo.Ensemble(n_neurons=100, dimensions=2)

    ens_err = nengo.Ensemble(n_neurons=100, dimensions=2)
    ens_tar = nengo.Ensemble(n_neurons=100, dimensions=2)

    nengo.Connection(nd_stim, ens_x)
    nengo.Connection(nd_stim, ens_tar, function=f)
    nengo.Connection(ens_tar, ens_err, transform=-1.0)
    nengo.Connection(ens_y,   ens_err, transform= 1.0)

    con = nengo.Connection(ens_x, ens_y,
                           learning_rule_type=nengo.PES(learning_rate=learning_rate),
                           transform=np.zeros((2, 1)))
    nengo.Connection(ens_err, con.learning_rule)

    p_x = nengo.Probe(ens_x, synapse=10e-3)
    p_y = nengo.Probe(ens_y, synapse=10e-3)
    p_err = nengo.Probe(ens_err, synapse=10e-3)
    p_tar = nengo.Probe(ens_tar, synapse=10e-3)
