'''
Boltzmann machines differ from Hopfield network mainly in how neurons are updated.
Boltzmann machines r representations of and sampling mechanisms for the Boltzmann distributions
defined by weights and threshold.

The neurons are divided into:
    - visible neurons, which receive data points as input
    - hidden neurons, which are not fixed by data points
(Note that Hopfield networks only have visible neurons.)

We hope the true distribution underlying a given data sample is approximated well by
the probability distribution represented by the Boltzmann machine on its visible neurons.

There r two phases when training:
    - Positive phase: visible neurons r fixed to a randomly chosen data point;
                      only hidden neurons r updated until thermal equilibrium is reached.
    - Negative phase: All neurons r updated until thermal equilibrium is reached.

Unfortunately, this procedure is impractical unless the networks r very small.
(The larger the network, the more update steps need to be carried out to
obtain sufficiently reliable statistics for the (paired) neurons activation.)

Restricted Boltzmann machine has an efficient training way:
    - Restriction consists in using a bipartite graph instead of a fully connected graph.

So we will only implement restricted Boltzmann machine.
'''