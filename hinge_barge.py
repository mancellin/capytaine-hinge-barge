#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
from typing import Sequence

import numpy as np
import xarray as xr
import pandas as pd
from scipy.linalg import block_diag

import capytaine as cpt
from meshmagick.hydrostatics import Hydrostatics


FOUR_DOFS_NAMES = ['Heave', 'Pitch', 'body_1__Relative_Pitch', 'body_3__Relative_Pitch']
RHO_WATER = 1025.0

RectangleParams = namedtuple("RectangleParams", ["name", "shape", "geometric_center", "center_of_mass", "resolution"])

DEFAULT_BODY_1 = RectangleParams(name="body_1", shape=(0.68, 0.4, 0.10), geometric_center=(0.0, 0.0, 0.0), center_of_mass=(0.0, 0.0, -0.025), resolution=10)
DEFAULT_BODY_2 = RectangleParams(name="body_2", shape=(0.28, 0.4, 0.15), geometric_center=(0.0, 0.0, 0.0), center_of_mass=(0.0, 0.0, -0.025), resolution=10)
DEFAULT_BODY_3 = RectangleParams(name="body_3", shape=(1.00, 0.4, 0.10), geometric_center=(0.0, 0.0, 0.0), center_of_mass=(0.0, 0.0, -0.025), resolution=10)
DEFAULT_DAMPING_PLATE = RectangleParams(name="damping_plate", shape=(0.48, 0.4, 0.01), geometric_center=(0.0, 0.0, -0.227), center_of_mass=(0.0, 0.0, -0.227), resolution=10)


class HingeBarge(cpt.FloatingBody):
    """The purpose of this class is to generate a capytaine.FloatingBody
    that represents a hinge-barge with an arbitrary number of bodies.
    Each body can be composed of several rectangular parallelepiped moving together.
    """

    def __init__(
            self,
            bodies=((DEFAULT_BODY_1,), (DEFAULT_BODY_2, DEFAULT_DAMPING_PLATE), (DEFAULT_BODY_3,)),
            center=(0.88, 0, 0),
            distance_between_bodies=0.06, clever_mesh=False,
            dofs_names=FOUR_DOFS_NAMES, pto=None,
    ):
        """Initialize the mesh and properties of a hinge-barge."""
        self.bodies_params = bodies
        self.center = np.array(center)
        self.distance_between_bodies = distance_between_bodies

        # Generate the meshes of the individual bodies
        self._generate_meshes()

        # Merge the bodies together
        tmp_barge = cpt.FloatingBody.join_bodies(*self.bodies)

        actual_dofs = {}
        self.transformation_matrix = []
        for dof_name in dofs_names:
            coeffs = self._global_dof_from_individual_dofs(dof_name)
            self.transformation_matrix.append(coeffs)
            actual_dofs[dof_name] = sum([coef * motion for coef, motion in zip(coeffs, tmp_barge.dofs.values())])
        self.transformation_matrix = np.array(self.transformation_matrix)

        cpt.FloatingBody.__init__(self, mesh=tmp_barge.mesh, dofs=actual_dofs, name="hinge_barge")

        # Position and clipping
        self.translate(center)
        self.keep_immersed_part()

        # Compute static properties
        self._compute_mass()
        self._compute_hydrostatic_stiffness()
        self._compute_pto(pto)

    @staticmethod
    def _generate_mesh_of_a_parallelepiped(params: RectangleParams):
        """Generate the mesh of a single parallelepiped using capytaine mesh generator."""
        resolution = params.resolution*np.array(params.shape)
        resolution = resolution.astype(np.int)
        resolution[resolution < 1] = 1  # There should be at least one panel in each direction.

        body = cpt.RectangularParallelepiped(
            size=params.shape,
            resolution=resolution,
            center=params.geometric_center,
            # reflection_symmetry=params.clever_mesh,
            name=params.name,
        )
        body.center_of_mass = np.asarray(params.center_of_mass)
        body.rotation_center = body.geometric_center

        return body

    def _generate_meshes(self):
        """Create the mesh of the bodies one after the other."""
        self.bodies = []
        self.individual_hydrostatic_stiffness = []
        x_position_of_next_body = 0.0

        for body_params in self.bodies_params:
            main_subbody = self._generate_mesh_of_a_parallelepiped(body_params[0])

            body = main_subbody.copy()
            for subbody_params in body_params[1:]:
                body += self._generate_mesh_of_a_parallelepiped(subbody_params)

            body.name = main_subbody.name
            body.center_of_mass = main_subbody.center_of_mass
            body.geometric_center = main_subbody.geometric_center

            body.add_all_rigid_body_dofs()

            body.translate_x(x_position_of_next_body + body_params[0].shape[0]/2)
            x_position_of_next_body += body_params[0].shape[0] + self.distance_between_bodies

            self.bodies.append(body)
            self.individual_hydrostatic_stiffness.append(
                Hydrostatics(body.mesh.merged().to_meshmagick()).hs_data['stiffness_matrix']
            )

    def _global_dof_from_individual_dofs(self, dof_name):
        """All the dofs of the hinge-barge can be seen as linear combination
        of the rigid body dofs of the individual bodies composing the barge.
        This function returns the coefficients used to sum the dofs of the
        bodies to get a dof of the full barge.

        Parameters
        ----------
        dof_name: str
            The name of the dof of the full barge.
            Accepted: 'Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'
                      'body_1__Relative_Pitch', 'body_3__Relative_Pitch'

        Returns
        -------
        np.ndarray
            array of 6*nb_bodies coefficients
        """
        vector = np.zeros(6*len(self.bodies), dtype=np.float64)
        rigid_body_dofs = ['surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']

        if dof_name.lower() in rigid_body_dofs[:4]:
            vector[rigid_body_dofs.index(dof_name.lower())::6] = 1.0
            return vector

        elif dof_name.lower() == 'pitch':
            vector[rigid_body_dofs.index(dof_name.lower())::6] = 1.0
            for i_body, body in enumerate(self.bodies):
                vector[6*i_body+2] = -body.geometric_center[0] + self.center[0]
            return vector

        elif dof_name.lower() == 'yaw':
            vector[rigid_body_dofs.index(dof_name.lower())::6] = 1.0
            for i_body, body in enumerate(self.bodies):
                vector[6*i_body+1] = body.geometric_center[0] - self.center[0]
            return vector

        # Specific dofs for the three-body hinge-barge
        elif dof_name.lower() == 'body_1__relative_pitch':
            a = self.bodies_params[0][0].shape[0]/2 + self.distance_between_bodies/2
            return np.array([0, 0, a, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        elif dof_name.lower() == 'body_3__relative_pitch':
            c = self.bodies_params[2][0].shape[0]/2 + self.distance_between_bodies/2
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -c, 0, 1, 0])

        else:
            raise ValueError(f"Unrecognized dof name: {dof_name}")

    def _transformation_matrix(self):
        """Matrix of the coefficients discussed in the function above."""
        return np.array([self._global_dof_from_individual_dofs(dof) for dof in self.dofs])

    def _compute_mass(self):
        """Compute the mass matrix for the hinge-barge by combining the mass matrices
        of the individual parallelepiped composing the full hinge-barge."""

        def mass_matrix(params):
            """Mass matrix for a single rectangular parallelepiped.
            The mass is computed from the displaced mass of water
            because the body is assumed to be floating in equilibrium.
            """
            # TODO: Test for all center_of_mass and all dofs.
            draught = params.shape[2]/2
            mass = RHO_WATER * params.shape[0] * params.shape[1] * draught
            return np.diag([
                mass, mass, mass,
                (mass * (params.shape[1]**2 + params.shape[2]**2))/12 + mass*(params.center_of_mass[2]**2), # Roll
                (mass * (params.shape[0]**2 + params.shape[2]**2))/12 + mass*(params.center_of_mass[2]**2), # Pitch
                (mass * (params.shape[0]**2 + params.shape[1]**2))/12,                   # Yaw
            ])
        blocks = [sum(mass_matrix(body) for body in body_group) for body_group in self.bodies_params]
        P = self._transformation_matrix()
        self.mass = self.add_dofs_labels_to_matrix(P @ block_diag(*blocks) @ P.T)

    def _compute_hydrostatic_stiffness(self):
        """Compute the hydrostatic stiffness for the hinge-barge by combining the hydrostatic stiffnesses
        of the individual parallelepiped composing the full hinge-barge."""
        individual_hs = [np.zeros((6, 6)) for _ in range(len(self.bodies))]
        for i in range(len(self.bodies)):
            individual_hs[i][2:5, 2:5] = self.individual_hydrostatic_stiffness[i]
        P = self._transformation_matrix()
        self.hydrostatic_stiffness = self.add_dofs_labels_to_matrix(P @ block_diag(*individual_hs) @ P.T)

    def _compute_pto(self, pto=None):
        if pto is None:
            pto = {
                'body_1__Relative_Pitch': 10,
                'body_3__Relative_Pitch': 16,
            }
        non_pto_dofs = set(self.dofs) - set(pto.keys())
        pto.update({dof: 0.0 for dof in non_pto_dofs})
        self.pto = xr.DataArray(pd.Series(pto), dims=["radiating_dof"])
        self.Kpto = self.add_dofs_labels_to_matrix(
            np.diag([self.pto.sel(radiating_dof=name) for name in self.dofs.keys()])
        )


    def compute_hydrodynamics(self,
                              omega_range=np.linspace(2, 10, 40),
                              wave_direction_range=[np.pi],
                              ):
        dataset = xr.Dataset(coords={
            'omega': omega_range,
            'wave_direction': wave_direction_range,
            'radiating_dof': list(self.dofs),
            'rho': RHO_WATER,
        })
        self.dataset = cpt.Nemoh(linear_solver="gmres").fill_dataset(dataset, [self])
        self.dataset['PTO_matrix'] = self.Kpto
        return self.dataset

    def compute_motion(self, wave_amplitude=0.02, viscous_dissipation=None):
        if viscous_dissipation is None:
            viscous_dissipation = np.zeros(self.Kpto.shape)

        rao = [cpt.post_pro.rao(
            self.dataset, wave_direction=wave_direction, dissipation=self.dataset['PTO_matrix'] + viscous_dissipation
        ) for wave_direction in self.dataset.coords["wave_direction"]]

        self.dataset['RAO'] = xr.concat(rao, dim='wave_direction')
        self.dataset['motion'] = wave_amplitude * self.dataset['RAO']
        return self.dataset['motion']

    def compute_power(self):
        power_per_dof = 0.5 * self.dataset['PTO_matrix'].dot(np.square(np.abs(1j * self.dataset.coords['omega'] * self.dataset['motion'])))
        self.dataset['power'] = power_per_dof.sum(dim="influenced_dof")
        return self.dataset['power']


        # if plot_style is None:
        #     plot_style = {}
        # self.dataset.attrs['plot_style'] = plot_style


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    hb = HingeBarge(dofs_names=FOUR_DOFS_NAMES + ["Surge"])
    # print(hb.mass)
    # print(hb.hydrostatic_stiffness)
    # print(hb.Kpto)
    hb.compute_hydrodynamics()
    print(hb.dataset)
    # print(hb.animate_dof("body_1__relative_pitch", 1.0))

