import sys, os, time
import numpy as np
import showcase1_ageing as utils
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import logger

def get_connectivity(scaling_factor):
        #SC = np.loadtxt('/home/jwangbay/scratch/2023modelling/new/data/vab/sub_00010_SC_7NW100parc.txt')
        #SC = np.loadtxt('/home/jwangbay/scratch/2023modelling/new/scripts/matrix_thresh.txt')
        SC = np.loadtxt('/home/jwangbay/scratch/may_modelling/2023modelling/all_data/SC_thresh75le35/sub-CC_sc_TVBSchaeferTian220_75consensus_le35yo.txt')
        #SC = self.load_sc(subject)
        SC = SC / scaling_factor
        conn = connectivity.Connectivity(
                weights = SC,
                tract_lengths=np.ones_like(SC),
                centres = np.zeros(np.shape(SC)[0]),
                speed = np.r_[np.Inf]
        )
        conn.compute_region_labels()
 #       logger.warning("Placeholder region names!")
        return conn

my_magic = 1
my_noise = 0.03
my_G = 2.45

mpr = models.MontbrioPazoRoxin()



utils.phase_plane_interactive(
    mpr,
    integrators.HeunStochastic(
        dt=0.01, 
        noise=noise.Additive(nsig=np.r_[0.03])
    )
)

dt      = 0.01
nsigma  = my_noise
G       = my_G
sim_len = 30e3

sim = simulator.Simulator(
    connectivity = get_connectivity(my_magic),
    model = models.MontbrioPazoRoxin(
        eta   = np.r_[-4.6],
        J     = np.r_[14.5],
        Delta = np.r_[0.7],
        tau   = np.r_[1.],
    ),
    coupling = coupling.Linear(a=np.r_[G]),
    integrator = integrators.HeunStochastic(
        dt = 0.01,
        noise = noise.Additive(nsig=np.r_[nsigma, nsigma*2])
    ),
    monitors = [monitors.TemporalAverage(period=0.1)]
).configure()


# run time ~ 2 minutes
runner = NbMPRBackend()
(tavg_t, tavg_d), = runner.run_sim(sim, simulation_length=sim_len)
tavg_t *= 10

bold_t, bold_d = utils.tavg_to_bold(tavg_t, tavg_d, tavg_period=1.)


# cut the initial transient (10s)
bold_t = bold_t[2:]
bold_d = bold_d[2:]


ax = utils.plot_ts_stack(tavg_d[60*1000:120*1000:10,0,:,0], x=tavg_t[60*1000:120*1000:10]/1000., width=20)
ax.set(xlabel='time [s]');


plt.savefig("ts.png")
