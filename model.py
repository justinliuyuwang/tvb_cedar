import sys, os, time
import numpy as np
import showcase1_ageing as utils
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import resource
import time
import fcntl

import psutil



def get_connectivity(scaling_factor,subject):
        
        
        SC = np.loadtxt('/home/jwangbay/scratch/tvb/data/SC_thresh75le35/'+subject+'_sc_TVBSchaeferTian220_75consensus_le35yo.txt')

        
        SC = SC / scaling_factor
        conn = connectivity.Connectivity(
                weights = SC,
                tract_lengths=np.ones_like(SC),
                centres = np.zeros(np.shape(SC)[0]),
                speed = np.r_[np.Inf]
        )
        conn.compute_region_labels()
 
        return conn

def process_sub(my_magic,my_noise,my_G,subject,FCD_file):
    start_time = time.time()

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
        connectivity = get_connectivity(my_magic,subject),
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

    FCD, _ = utils.compute_fcd(bold_d[:,0,:,0], win_len=40)
    FCD_VAR_OV_vect= np.var(np.triu(FCD, k=40))

 



    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time

    np.save(FCD_file, FCD)

    return([FCD_VAR_OV_vect,time_taken])

