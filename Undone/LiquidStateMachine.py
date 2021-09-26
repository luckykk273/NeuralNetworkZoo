"""
The code refers to the github below:
https://github.com/baiyuzi/LSM_python/blob/master/LSM_LIB.py
The discussion in the paper below has many similarities with the code:
https://www.researchgate.net/publication/333528093_Energy-efficient_FPGA_Spiking_Neural_Accelerators_with_Supervised_and_Unsupervised_Spike-timing-dependent-Plasticity#pf12

There is also C# version below:
https://bitbucket.org/Hananel/liquid-state-machine/src/master/

LSM consists of two major parts:
    - Reservoir: There r a number of spiking neurons which randomly (and not fully) connect to each other in.
    - Readout layer: It receives reservoir responses as inputs for classification.

Both inputs and outputs of a LSM are streams of data in continuous time.
These inputs and outputs r modelled mathematically as function u(t) and y(t).
"""

"""
To be continued...
"""
