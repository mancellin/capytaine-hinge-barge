#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.linalg import block_diag
import xarray as xr
import pandas as pd
import capytaine as cpt

class HingeBarge(cpt.FloatingBody):
    def __init__(self, *, bodies, distance_between_bodies, center=(0, 0, 0), name=None):
        self.components = [body.copy() for body in bodies]
        self.nb_components = len(self.components)
        self.nb_hinges = len(self.components) - 1
        self.distance_between_bodies = distance_between_bodies

        # MERGING BODIES
        def width_of_body(body):
            return body.mesh.vertices[:, 0].max() - body.mesh.vertices[:, 0].min()
        widths = np.array([width_of_body(bd) for bd in self.components])
        total_width = sum(widths) + self.nb_hinges*self.distance_between_bodies

        x_corner_of_bodies = (
            np.cumsum(np.concatenate([[0], widths[:-1]]))
            + np.arange(0, self.nb_hinges+1)*self.distance_between_bodies
            + center[0] - total_width/2
        )
        x_center_of_bodies = x_corner_of_bodies + widths/2
        x_positions_of_hinges = x_corner_of_bodies[1:] - self.distance_between_bodies/2


        for i, comp in enumerate(self.components):
            comp.name = f"body_{i}"

            # Reset dofs
            comp.dofs = {}
            comp.add_all_rigid_body_dofs()

            # Position all bodies side by side
            comp.translate_x(x_corner_of_bodies[i] - comp.mesh.vertices[:, 0].min())

        merged_components = sum(self.components[1:], self.components[0])

        # MERGING DOFS
        self.P = np.zeros((6 + self.nb_hinges, 6*self.nb_components))
        actual_dofs = {}
        for i_dof, dof in enumerate(["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]):
            for j in range(self.nb_components):
                self.P[i_dof, 6*j + i_dof] = 1.0
                if dof == "Pitch":  # Also add some heave to the bodies far from the center
                    self.P[i_dof, 6*j+2] = center[0] - x_center_of_bodies[j]
                elif dof == "Yaw":  # Also add some sway to the bodies far from the center
                    self.P[i_dof, 6*j+1] = x_center_of_bodies[j] - center[0]

            actual_dofs[dof] = sum(c * m for c, m in zip(self.P[i_dof, :], merged_components.dofs.values()))

        for i_hinge in range(self.nb_hinges):
            i_dof = 6 + i_hinge
            for j in range(self.nb_components):
                self.P[i_dof, 6*j + 2] = abs(x_positions_of_hinges[i_hinge] - x_center_of_bodies[j])   # Heave of body j
                self.P[i_dof, 6*j + 4] = 1.0 if x_center_of_bodies[j] < x_positions_of_hinges[i_hinge] else -1.0  # Pitch of body j

            actual_dofs[f"Hinge_{i_hinge}"] = sum(c * m for c, m in zip(self.P[i_dof, :], merged_components.dofs.values()))


        # REST OF THE INITIALIZATION
        super().__init__(mesh=merged_components.mesh, dofs=actual_dofs, name=name)

    def compute_mass_matrix(self, rho_water=1000.0):
        try:
            comp_mass_matrices = [comp.rigid_body_mass_matrix(rho_water) for comp in self.components]
        except:
            raise Exception("Could not compute the mass matrix of the components of the hinge-barge.")
        self.mass = self.add_dofs_labels_to_matrix(self.P @ block_diag(*comp_mass_matrices) @ self.P.T)

    def compute_hydrostatic_stiffness(self, rho_water=1000.0, g=9.81):
        try:
            comp_hs = [comp.rigid_body_hydrostatic_stiffness(rho_water, g) for comp in self.components]
        except:
            raise Exception("Could not compute the mass matrix of the components of the hinge-barge.")
        self.hydrostatic_stiffness = self.add_dofs_labels_to_matrix(self.P @ block_diag(*comp_hs) @ self.P.T)

    def compute_hydrostatics(self, rho_water=1000.0, g=9.81):
        self.compute_mass_matrix(rho_water)
        self.compute_hydrostatic_stiffness(rho_water, g)

    def pto_dissipation_matrix(self, pto):
        return self.add_dofs_labels_to_matrix(
            np.diag([pto[dof_name] if dof_name in pto else 0.0 for dof_name in self.dofs.keys()])
        )





if __name__ == "__main__":
    bd1 = cpt.RectangularParallelepiped()
    bd2 = cpt.RectangularParallelepiped()
    bd3 = cpt.RectangularParallelepiped()
    # bd1.show()

    hb = HingeBarge(bodies=[bd1, bd2, bd3], distance_between_bodies=0.1)
    hb.compute_hydrostatics(rho_water=1025.0, g=9.81)
    hb.keep_immersed_part()
    # hb.animate({"Hinge_0": 0.1}, loop_duration=1.0)
    # hb.animate({"Hinge_1": 0.1}, loop_duration=1.0)
    # hb.animate({"Hinge_0": 0.1, "Hinge_1": -0.1}, loop_duration=1.0)

    test_matrix = xr.Dataset(coords={
        'omega': np.linspace(1.0, 5.0, 50),
        'wave_direction': [0.0],
        'radiating_dof': list(hb.dofs),
        'rho': 1025.0,
    })
    dataset = cpt.BEMSolver().fill_dataset(test_matrix, [hb])
    # print(dataset)

    pto = {"Hinge_0": 6, "Hinge_1": 1}
    rao = cpt.post_pro.rao(
        dataset,
        wave_direction=0.0,
        dissipation=hb.pto_dissipation_matrix(pto),
    )
    wave_amplitude = 0.1
    motion = wave_amplitude * rao

    # hb.animate(0.1 * rao.sel(omega=1.75), loop_duration=1.0)

    power_per_dof = 0.5 * hb.pto_dissipation_matrix(pto) @ (np.square(np.abs(1j * dataset.coords['omega'] * motion)))
    power = power_per_dof.sum(dim="influenced_dof")
    power.plot()

    import matplotlib.pyplot as plt
    plt.show()
