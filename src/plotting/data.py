"""Read saved quantities and evaluate molecular angular moments."""

from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from ..results import text_attribute


def angular_weights(theta, phi):
    """
    Construct normalised solid angle quadrature weights.

    # Arguments:
    - theta::np.ndarray: The polar grid, in radians.
    - phi::np.ndarray: The closed azimuthal grid, in radians.

    # Returns:
    - np.ndarray: Weights with shape (n_theta, n_phi).
    """
    # Trapezoid weights along each angle, with half steps at the ends.
    weights_theta = np.empty_like(theta)
    weights_phi = np.empty_like(phi)
    for grid, weights in ((theta, weights_theta), (phi, weights_phi)):
        weights[0] = (grid[1] - grid[0]) / 2
        weights[-1] = (grid[-1] - grid[-2]) / 2
        weights[1:-1] = (grid[2:] - grid[:-2]) / 2

    # The sin(theta) of the solid angle, pinned to zero at the poles where it vanishes.
    sin_theta = np.sin(theta)
    sin_theta[[0, -1]] = 0
    weights = (weights_theta * sin_theta)[:, None] * weights_phi[None, :]
    if weights.sum() <= 0:
        raise ValueError('Angular averages need polar samples between the poles.')
    return weights / weights.sum()


def group_transitions(energies, tolerance):
    """
    Group neighbouring transition energies, including chains within the tolerance.

    # Arguments:
    - energies::dict: Original transition indices and energies, in eV.
    - tolerance::float: The neighbouring-energy tolerance, in eV.

    # Returns:
    - list[tuple]: The grouped original indices, ordered by energy.
    """
    # Walk up in energy, starting a new group whenever the gap to the previous state is too big.
    # Comparing to the previous state rather than the first chains states that are each close.
    groups = []
    previous = None
    for s in sorted(energies, key=lambda s: (energies[s], s)):
        gap = None if previous is None else energies[s] - energies[previous]
        if gap is None or (gap > tolerance and not np.isclose(gap, tolerance, rtol=1e-12, atol=0)):
            groups.append([])
        groups[-1].append(s)
        previous = s
    return [tuple(group) for group in groups]


class MolecularData:
    """Read one molecular file, keeping Julia/Python array-order conversion here."""

    def __init__(self, path, tolerance=0.01, sampling=None):
        """
        Read the shared coordinates and transition metadata.

        # Arguments:
        - path::Path: The grouped molecular result file.
        - tolerance::float: The neighbouring-energy tolerance, in eV.
        - sampling::dict or None: Cartesian/Fourier angular sampling sizes.
        """
        self.path = Path(path)
        self.sampling = sampling or {'n_q': 251, 'n_theta': 97, 'n_phi': 97}
        self._amplitudes = {}
        self._coefficients = {}
        self._densities = {}
        self._moments = {}
        with h5py.File(self.path) as results:
            if not results.attrs.get('completed', False):
                raise ValueError(f'{path}: the molecular calculation is incomplete.')
            self.mode = text_attribute(results, 'mode')
            self.grid = {name: np.asarray(values, dtype=float) for name, values in results['grid'].items()}
            self.energies = {int(s): float(group['Delta_E'][()]) for s, group in results['transitions'].items()}
            self.available = {int(s): set(group) for s, group in results['transitions'].items()}
        if not self.energies:
            raise ValueError(f'{path}: no transitions were saved.')
        self.groups = group_transitions(self.energies, tolerance)

    @cached_property
    def angular_grid(self):
        """
        Get the saved spherical grid or a fully covered Cartesian/Fourier angular grid.

        # Returns:
        - tuple: The q, theta and phi coordinates.
        """
        # The spherical method already has the angles.
        if self.mode == 'spherical':
            return self.grid['q'], self.grid['theta'], self.grid['phi']

        # The Cartesian and Fourier grids do not, so build one out to the largest |q| that the box
        # fully encloses, which is the smallest half width of its three axes.
        axes = [self.grid[name] for name in ('q_x', 'q_y', 'q_z')]
        if any(grid[0] >= 0 or grid[-1] <= 0 for grid in axes):
            raise ValueError('Angular averages need a Cartesian grid enclosing the origin.')
        q_max = min(min(-grid[0], grid[-1]) for grid in axes)
        return (np.linspace(0, q_max, self.sampling['n_q']),
                np.linspace(0, np.pi, self.sampling['n_theta']),
                np.linspace(0, 2*np.pi, self.sampling['n_phi']))

    def require(self, states, quantity):
        """
        Require a saved quantity for the selected transitions.

        # Arguments:
        - states::iterable: The transition indices.
        - quantity::str: The required dataset.
        """
        for s in states:
            if s not in self.available or quantity not in self.available[s]:
                raise ValueError(f'{self.path}: transition {s} has no saved {quantity}.')

    def amplitude(self, s):
        """
        Read a complex amplitude in (theta, phi, q) or (q_x, q_y, q_z) order.

        # Arguments:
        - s::int: The original transition index.

        # Returns:
        - np.ndarray: The amplitude in the plotting coordinate order.
        """
        self.require([s], 'f_s')
        if s not in self._amplitudes:
            with h5py.File(self.path) as results:
                values = np.asarray(results[f'transitions/{s}/f_s'])
            # HDF5 hands back the Julia axes reversed, so put them in the order the plots want.
            order = (1, 0, 2) if self.mode == 'spherical' else (2, 1, 0)
            self._amplitudes[s] = values.transpose(order)
        return self._amplitudes[s]

    def density(self, s):
        """
        Read a real transition density in (x, y, z) order, in Angstrom^-3.

        # Arguments:
        - s::int: The original transition index.

        # Returns:
        - np.ndarray: The density in the plotting coordinate order.
        """
        self.require([s], 'transition_density')
        if s not in self._densities:
            with h5py.File(self.path) as results:
                values = np.asarray(results[f'transitions/{s}/transition_density'], dtype=float)
            # HDF5 hands back the Julia axes reversed, so put them in the order the plots want.
            self._densities[s] = values.transpose(2, 1, 0)
        return self._densities[s]

    def density_plane(self, s, plane):
        """
        Sample the transition density on a plane through the origin, in Angstrom^-3.

        # Arguments:
        - s::int: The transition index.
        - plane::str: xy, xz or yz.

        # Returns:
        - tuple: Horizontal coordinates, vertical coordinates and the plane values.
        """
        # The density lives on the real space box the FFT used, so the axes are positions.
        horizontal, vertical = (self.grid[axis] for axis in plane)
        first, second = np.meshgrid(horizontal, vertical)
        points = np.zeros((*first.shape, 3))
        for values, axis in zip((first, second), plane):
            points[..., 'xyz'.index(axis)] = values

        # Every point of the plane is inside the box, so unlike the spherical form factor there are
        # no corners to mask off.
        coordinates = tuple(self.grid[name] for name in ('x', 'y', 'z'))
        sampled = RegularGridInterpolator(coordinates, self.density(s), bounds_error=True)(points)
        return horizontal, vertical, sampled

    def coefficients(self, states):
        """
        Sum real intensity coefficients over the selected transitions.

        # Arguments:
        - states::iterable: The original transition indices.

        # Returns:
        - np.ndarray: Coefficients in (q, spherical harmonic key) order.
        """
        self.require(states, 'f_sq_lm')
        for s in states:
            if s not in self._coefficients:
                with h5py.File(self.path) as results:
                    self._coefficients[s] = np.asarray(results[f'transitions/{s}/f_sq_lm'], dtype=float).T
        return sum(self._coefficients[s] for s in states)

    def sample(self, s, points):
        """
        Interpolate one amplitude at Cartesian momentum points.

        # Arguments:
        - s::int: The transition index.
        - points::np.ndarray: Cartesian momenta, with final dimension three, in keV.

        # Returns:
        - np.ndarray: The sampled complex amplitude.
        """
        # The spherical amplitude is stored against angles, so turn the points into (theta, phi, q).
        if self.mode == 'spherical':
            radius = np.linalg.norm(points, axis=-1)
            theta = np.arccos(np.clip(np.divide(points[..., 2], radius,
                out=np.zeros_like(radius), where=radius > 0), -1, 1))
            phi = np.mod(np.arctan2(points[..., 1], points[..., 0]), 2*np.pi)
            coordinates = (self.grid['theta'], self.grid['phi'], self.grid['q'])
            directions = np.stack([theta, phi, radius], axis=-1)
        else:
            coordinates = tuple(self.grid[name] for name in ('q_x', 'q_y', 'q_z'))
            directions = points
        return RegularGridInterpolator(coordinates, self.amplitude(s), bounds_error=True)(directions)

    def moments(self, states):
        """
        Calculate the angular mean and absolute standard deviation after summing intensities.

        # Arguments:
        - states::tuple: The original indices in one transition group.

        # Returns:
        - tuple: The mean and standard deviation on the angular q grid.
        """
        states = tuple(states)
        if states not in self._moments:
            # Sum the intensities of the group first, then take the moments of the total.
            q, theta, phi = self.angular_grid
            if self.mode == 'spherical':
                intensity = sum(np.abs(self.amplitude(s))**2 for s in states)
            else:
                # Sample the Cartesian amplitude on a spherical grid instead.
                th, ph, radius = np.meshgrid(theta, phi, q, indexing='ij')
                points = np.stack([radius*np.sin(th)*np.cos(ph), radius*np.sin(th)*np.sin(ph),
                                   radius*np.cos(th)], axis=-1)
                # Roundoff at the inscribed sphere must stay inside the saved Cartesian grid.
                for axis, name in enumerate(('q_x', 'q_y', 'q_z')):
                    points[..., axis] = np.clip(points[..., axis], self.grid[name][0], self.grid[name][-1])
                intensity = sum(np.abs(self.sample(s, points))**2 for s in states)
            weights = angular_weights(theta, phi)
            mean = np.einsum('tpq,tp->q', intensity, weights)
            variance = np.einsum('tpq,tp->q', (intensity - mean[None, None, :])**2, weights)
            self._moments[states] = mean, np.sqrt(variance)
        return self._moments[states]

    def ranked_groups(self):
        """
        Rank transition groups by their momentum-integrated angular average intensity.

        # Returns:
        - list[tuple]: The groups in descending response order.
        """
        strengths = {}
        for states in self.groups:
            # The l = 0 coefficient is the angular average up to the 1/sqrt(4π) of Y_00, so use it
            # when it was saved rather than resampling the angles.
            if all('f_sq_lm' in self.available[s] for s in states):
                mean = self.coefficients(states)[:, 0] / np.sqrt(4*np.pi)
                q = self.grid['q']
            else:
                mean = self.moments(states)[0]
                q = self.angular_grid[0]
            strengths[states] = float(trapezoid(mean, q))
        return sorted(self.groups, key=lambda states: (-strengths[states], states))

    def select_groups(self, selection='dominant', count=None):
        """
        Expand selected indices to complete energy groups without duplicating a group.

        # Arguments:
        - selection::str or list: all, dominant, or original transition indices.
        - count::int or None: The maximum number of dominant groups.

        # Returns:
        - list[tuple]: The selected groups.
        """
        if selection == 'dominant':
            return self.ranked_groups()[:count]
        if selection == 'all':
            return self.groups
        if not isinstance(selection, list) or any(s not in self.energies for s in selection):
            raise ValueError(f'{self.path}: invalid transition selection {selection}.')
        return [group for group in self.groups if set(group).intersection(selection)]

    def plane(self, states, plane, quantity='intensity'):
        """
        Sample a zero-coordinate plane, summing intensities only for grouped states.

        # Arguments:
        - states::tuple: The original transition indices.
        - plane::str: xy, xz or yz.
        - quantity::str: intensity, real or imag.

        # Returns:
        - tuple: Horizontal coordinates, vertical coordinates and the plane values.
        """
        if quantity != 'intensity' and len(states) != 1:
            raise ValueError('Real and imaginary form factors require an individual transition.')
        # The spherical grid has no Cartesian axes, so lay one out over the range it covers.
        if self.mode == 'spherical':
            q_max = float(self.grid['q'][-1])
            horizontal = vertical = np.linspace(-q_max, q_max, 401)
        else:
            horizontal, vertical = (self.grid['q_' + axis] for axis in plane)

        # Build the Cartesian points of the plane, leaving the third coordinate at zero.
        first, second = np.meshgrid(horizontal, vertical)
        points = np.zeros((*first.shape, 3))
        for values, axis in zip((first, second), plane):
            points[..., 'xyz'.index(axis)] = values

        # The corners of a square plane fall outside the sphere the data covers, so mask them off.
        radius = np.linalg.norm(points, axis=-1)
        if self.mode == 'spherical':
            mask = radius > q_max
            points[mask] = 0
        else:
            mask = np.zeros(first.shape, dtype=bool)
        values = np.zeros(first.shape)
        for s in states:
            amplitude = self.sample(s, points)
            values += np.abs(amplitude)**2 if quantity == 'intensity' else amplitude.real if quantity == 'real' else amplitude.imag
        return horizontal, vertical, np.ma.masked_where(mask, values)


class CrystalData:
    """Read the current crystal structure function output without altering its normalisation."""

    def __init__(self, path):
        """
        Read the saved momentum and energy grids.

        # Arguments:
        - path::Path: The current crystal form factor file.
        """
        self.path = Path(path)
        with h5py.File(path) as results:
            if 'S_lm' not in results and 'f_sq_lm' not in results:
                raise ValueError(f'{path}: no structure function or crystal intensity coefficients were saved.')
            self.q = np.asarray(results['q_grid'], dtype=float)
            self.energy = np.asarray(results['E_grid_eV'], dtype=float) if 'E_grid_eV' in results else None
            self.n_modes = results['S_lm' if 'S_lm' in results else 'f_sq_lm'].shape[0]
            # Which quantities the run was asked to save, so a figure needing one can be skipped.
            self.saved = set(results)

    def mode(self, mode):
        """
        Read one saved spherical harmonic in (E, q) order, in Angstrom^-3 eV^-1.

        # Arguments:
        - mode::tuple: The indices (l, m).

        # Returns:
        - np.ndarray: The signed structure function coefficient.
        """
        # The spherical harmonics are stored against the key l^2 + l + m.
        l, m = mode
        key = l*l + l + m
        if l < 0 or abs(m) > l or key >= self.n_modes:
            raise ValueError(f'Harmonic ({l},{m}) is not available.')
        with h5py.File(self.path) as results:
            if 'S_lm' not in results:
                raise ValueError(f'{self.path}: no structure function was saved.')
            return np.asarray(results['S_lm'][key], dtype=float).T

    def summed_coefficients(self, name):
        """
        Sum one saved set of intensity coefficients over states, including 2 pi / V_cell.

        # Arguments:
        - name::str: The saved dataset, either f_sq_lm or f_sq_lm_incoherent.

        # Returns:
        - np.ndarray: The summed coefficients in (q, spherical harmonic key) order, in Angstrom^-3.
        """
        with h5py.File(self.path) as results:
            if name not in results or 'V_cell_A3' not in results:
                raise ValueError(f'{self.path}: {name} or the cell volume were not saved.')
            coefficients = results[name]
            if coefficients.ndim != 3 or coefficients.shape[:2] != (self.n_modes, len(self.q)):
                raise ValueError(f'{self.path}: {name} does not match the saved grid.')
            volume = float(results['V_cell_A3'][()])
            if not np.isfinite(volume) or volume <= 0:
                raise ValueError(f'{self.path}: the unit cell volume must be positive.')
            # Sum over the states, one spherical harmonic at a time to keep the read small.
            summed = np.empty((len(self.q), self.n_modes))
            for key in range(self.n_modes):
                summed[:, key] = np.sum(coefficients[key], axis=1, dtype=float)

        # The 2π/V_cell is what makes this the structure function rather than the bare sum.
        return summed * (2*np.pi/volume)

    @cached_property
    def state_sums(self):
        """
        Sum the saved crystal intensity coefficients over states, including 2 pi / V_cell.

        # Returns:
        - np.ndarray: The summed coefficients in (q, spherical harmonic key) order, in Angstrom^-3.
        """
        return self.summed_coefficients('f_sq_lm')

    @cached_property
    def incoherent_sums(self):
        """
        Sum the saved incoherent reference over the monomer transitions, including 2 pi / V_cell.

        # Returns:
        - np.ndarray: The summed coefficients in (q, spherical harmonic key) order, in Angstrom^-3.
        """
        return self.summed_coefficients('f_sq_lm_incoherent')

    @cached_property
    def angular_mean(self):
        """
        Get the angular average structure function, in Angstrom^-3 eV^-1.

        # Returns:
        - np.ndarray: S_00 divided by sqrt(4 pi), in (E, q) order.
        """
        return self.mode((0, 0)) / np.sqrt(4*np.pi)

    @cached_property
    def integrated_mean(self):
        """
        Get the energy-integrated angular average structure function, in Angstrom^-3.

        # Returns:
        - np.ndarray: The angular average energy integral on the q grid.
        """
        return self.integrals[:, 0] / np.sqrt(4*np.pi)

    @cached_property
    def integrals(self):
        """
        Integrate each coefficient over the saved uniform energy grid.

        # Returns:
        - np.ndarray: The coefficients in (q, spherical harmonic key) order, in Angstrom^-3.
        """
        if self.energy is None or len(self.energy) < 2:
            raise ValueError(f'{self.path}: no structure function energy grid was saved.')
        # Average over the full grid because Float32 rounds neighbouring energy bins.
        step = (self.energy[-1] - self.energy[0]) / (len(self.energy) - 1)
        if not np.allclose(np.diff(self.energy), step, rtol=1e-3):
            raise ValueError('The saved structure function energy grid must be uniform.')
        # A uniform grid makes the integral a sum times the step, one harmonic at a time.
        integrated = np.empty((len(self.q), self.n_modes))
        with h5py.File(self.path) as results:
            for key in range(self.n_modes):
                integrated[:, key] = np.sum(results['S_lm'][key], axis=1, dtype=float) * step
        return integrated

    def ranked_modes(self, count, coefficients=None):
        """
        Rank non-monopole modes by the absolute area of their energy integral.

        # Arguments:
        - count::int: The number of modes.
        - coefficients::np.ndarray or None: The integrated coefficients to rank, or None for the structure function.

        # Returns:
        - list[tuple]: The indices (l, m), in descending response order.
        """
        coefficients = self.integrals if coefficients is None else coefficients
        areas = trapezoid(np.abs(coefficients), self.q, axis=0)

        # Skip key 0, the monopole, since it dominates everything and is plotted on its own.
        modes = []
        for key in np.argsort(areas)[::-1]:
            if key == 0:
                continue
            # Invert key = l^2 + l + m to get back (l, m).
            l = int(np.sqrt(key))
            modes.append((l, int(key - l*l - l)))
        return modes[:count]
