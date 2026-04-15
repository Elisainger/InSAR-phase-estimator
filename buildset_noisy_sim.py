#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 02 12:30:04 2019

@author: subhayanmukherjee
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), 'InSAR-Simulator'))
from simulator_2d import IfgSim

# settings
sim_cnt = 80 * 3
ifgsize = [1000, 1000]

output_dir = 'simtdset'
noisy_dir = path.join(output_dir, 'noisy')
clean_dir = path.join(output_dir, 'clean')

# create folders
os.makedirs(noisy_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

for loop_idx in range(sim_cnt):

    noisy_fname = path.join(noisy_dir, f'{loop_idx}.npy')
    clean_fname = path.join(clean_dir, f'{loop_idx}.npy')

    sim = IfgSim(width=ifgsize[1], height=ifgsize[0], rayleigh_scale=1.0)

    sim.add_n_buildings(width_range=[10,100], height_range=[1,40], depth_factor=0.35, nps=25)
    sim.add_n_gauss_bubbles(sigma_range=[10,150], amp_range=[-4.5,4.5], nps=110)
    sim.add_n_amp_stripes(thickness=9, nps=5)
    sim.add_n_amp_stripes(thickness=3, nps=50)

    sim.update(sigma=0.5)

    # save both
    np.save(noisy_fname, sim.ifg_noisy)
    np.save(clean_fname, sim.ifg)

    print('Saved pair #' + str(loop_idx))
