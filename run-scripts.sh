#!/bin/bash

set -x

cd 01_plasma_physics
cd ..

cd 02_envelope_equations
cd ..

cd 03_current_limits
cd ..

cd 04_halo_formation
python script_particle_core.py
cd ..

cd 05_collisions
cd ..