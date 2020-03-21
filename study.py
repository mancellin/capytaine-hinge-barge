#!/usr/bin/env python
# coding: utf-8

import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
import logging
from hinge_barge import HingeBarge, RectangleParams

logging.basicConfig(level=logging.INFO)

print(cpt.__version__)

resolution = 10

body_1_params = RectangleParams(
    name="body_1",
    shape=(0.68, 0.4, 0.1),            # dimensions of the parallelepiped
    geometric_center=(0.0, 0.0, 0.0),  # center of the parallelepiped
    center_of_mass=(0.0, 0.0, -0.025), # center of mass
    resolution=resolution,             # in panels per meter
)

body_2_params = RectangleParams(
    name="body_2",
    shape=(0.68, 0.4, 0.1),
    geometric_center=(0.0, 0.0, 0.0),
    center_of_mass=(0.0, 0.0, 0.0),
    resolution=resolution,
)

damping_plate_params = RectangleParams(
    name="damping_plate",
    shape=(0.48, 0.4, 0.01),
    geometric_center=(0.0, 0.0, -0.227),
    center_of_mass=(0.0, 0.0, -0.227),
    resolution=resolution,
)

body_3_params = RectangleParams(
    name="body_3",
    shape=(1.00, 0.4, 0.10),
    geometric_center=(0.0, 0.0, 0.0),
    center_of_mass=(0.0, 0.0, -0.025),
    resolution=resolution,
)


hb = HingeBarge(
    bodies=[[body_1_params], [body_2_params, damping_plate_params], [body_3_params]],
    dofs_names=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw", "body_1__Relative_Pitch", "body_3__Relative_Pitch"],
    distance_between_bodies=0.06,
)
# hb.show()

hb.compute_hydrostatics(rho_water=1025.0)
print(hb.mass)
print(hb.hydrostatic_stiffness)

hb.compute_hydrodynamics(
    omega_range=np.linspace(2, 10, 40),
    wave_direction_range=[0.0],
    rho_water=1025.0,
)

hb.compute_motion(
    wave_amplitude=0.02,
    pto={
        'body_1__Relative_Pitch': 10,
        'body_3__Relative_Pitch': 16,
    },
    viscous_dissipation=None,
)
power = hb.compute_power()

power.plot()
plt.show()
