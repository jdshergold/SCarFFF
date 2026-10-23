"""Numerical configuration and HDF5 checks, without importing the plot renderer."""

import copy
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from src.jobs.config import ROOT, load_job, read_systems
from src.jobs.runner import calculation_signature, run_job, validate_tddft, validate_form_factors
from src.plotting.config import figures_for_data, figures_for_mode, load_plot_config
from src.plotting.data import CrystalData, MolecularData, group_transitions


def molecular_file(path, mode, grid, amplitudes, energies=None, densities=None):
    """
    Write a numerical fixture in the Julia HDF5 array order.

    # Arguments:
    - path::Path: The fixture file.
    - mode::str: The molecular method.
    - grid::dict: The physical coordinates.
    - amplitudes::list: Amplitudes in (q, theta, phi) or (q_x, q_y, q_z) order.
    - energies::list or None: The transition energies, in eV.
    - densities::list or None: Transition densities in (x, y, z) order, in Angstrom^-3.
    """
    with h5py.File(path, 'w') as results:
        # Tag the file finished, or the reader refuses it.
        results.attrs['mode'] = np.bytes_(mode)
        results.attrs['completed'] = True

        # The coordinates every transition shares.
        coordinates = results.create_group('grid')
        for name, values in grid.items():
            coordinates.create_dataset(name, data=values).attrs['units'] = 'rad' if name in ('theta', 'phi') else 'keV'

        # One group per transition. Transposing puts the axes in the order Julia would have left
        # them, so the reader has to undo the same reversal it does on a real file.
        transitions = results.create_group('transitions')
        for s, amplitude in enumerate(amplitudes, 1):
            group = transitions.create_group(str(s))
            group.create_dataset('Delta_E', data=energies[s-1] if energies else s).attrs['units'] = 'eV'
            group.create_dataset('f_s', data=amplitude.T)
            if densities:
                group.create_dataset('transition_density', data=densities[s-1].T).attrs['units'] = 'angstrom^-3'


class JobConfiguration(unittest.TestCase):
    """Check mode-specific YAMLs and explicit stage controls."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.job = {'mode': 'spherical', 'output': 'results', 'input': {'smiles': 'O'},
                    'nstates': 2, 'stages': {'tddft': 'reuse', 'form_factors': 'skip'}}

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, job=None):
        path = self.directory / 'job.yaml'
        path.write_text(yaml.safe_dump(self.job if job is None else job))
        return load_job(path)

    def test_mode_specific_jobs_and_relative_paths(self):
        # Each mode should pick up only its own settings, and paths should resolve against the YAML
        # rather than wherever the job was launched from. A crystal uses the spherical settings.
        for mode in ('spherical', 'cartesian', 'fourier', 'crystal'):
            job = copy.deepcopy(self.job)
            job['mode'] = mode
            if mode == 'crystal':
                (self.directory / 'sample.cif').touch()
                job['input'] = {'cif': 'sample.cif'}
            result = self.load(job)
            self.assertEqual(result['output'], str(self.directory / 'results'))
            self.assertIn('spherical' if mode == 'crystal' else mode, result)
            self.assertNotIn('cartesian' if mode != 'cartesian' else 'fourier', result)
            if mode == 'crystal':
                self.assertEqual(result['input']['cif'], str(self.directory / 'sample.cif'))

    def test_invalid_jobs(self):
        # An unknown method, no states, a stray field, an unknown stage control, form factors asked
        # for with no TD-DFT, empty settings, a transition beyond nstates, a repeated transition, a
        # polar grid too coarse to average over, and a string where a flag belongs.
        mutations = [dict(mode='fft'), dict(nstates=0), dict(unrecognised=True),
                     dict(stages={'tddft': 'auto', 'form_factors': 'run'}),
                     dict(stages={'tddft': 'skip', 'form_factors': 'run'}),
                     dict(cartesian={}), dict(transitions=[3]), dict(transitions=[1, 1]),
                     dict(spherical={'n_theta': 1}), dict(execution={'gpu': 'false'})]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load({**self.job, **mutation})

    def test_deterministic_batches(self):
        # A CSV is numbered in the order it is written, so a result directory always belongs to the
        # same molecule.
        (self.directory / 'molecules.csv').write_text('smiles\nO\nC\n')
        self.job['input'] = {'csv': 'molecules.csv'}
        systems = read_systems(self.load())
        self.assertEqual([system['number'] for system in systems], [1, 2])
        self.assertEqual([system['input'] for system in systems], ['O', 'C'])

        # A directory of CIFs is sorted instead, since the filesystem gives no order. Either
        # capitalisation of the extension counts, and anything else is left alone.
        for name in ['z.CIF', 'a.cif', 'ignore.txt']:
            (self.directory / name).touch()
        self.job.update(mode='crystal', input={'cif_directory': '.'})
        self.assertEqual([system['label'] for system in read_systems(self.load())], ['a', 'z'])

    def test_reuse_does_not_start_a_calculation(self):
        # Asking to reuse a result that is not there must fail rather than quietly computing it.
        job = self.load()
        with patch('src.jobs.runner.compute_tddft') as tddft, patch('src.jobs.runner.subprocess.run') as worker:
            with self.assertRaisesRegex(ValueError, 'Job incomplete'):
                run_job(job)
            tddft.assert_not_called()
            worker.assert_not_called()

    def test_skip_does_not_start_a_calculation(self):
        # Skipping everything should succeed and still run nothing.
        self.job['stages'] = {'tddft': 'skip', 'form_factors': 'skip'}
        with patch('src.jobs.runner.compute_tddft') as tddft, patch('src.jobs.runner.subprocess.run') as worker:
            run_job(self.load())
            tddft.assert_not_called()
            worker.assert_not_called()

    def test_tddft_counts_precision_and_reuse_signature(self):
        # Two saved states, computed with five, which is the three-state buffer that stops a
        # degenerate pair being split.
        path = self.directory / 'td_dft.h5'
        with h5py.File(path, 'w') as results:
            results.attrs['stage_signature'] = 'valid'
            results.attrs['nstates_td_dft'] = 5
            results['Delta_E_eV'] = np.array([1, 2], dtype=np.float32)
            results['f_osc'] = np.ones(2, dtype=np.float32)
            for s in [1, 2]:
                results[f'X_state_{s}'] = np.ones((1, 1), dtype=np.complex64)
                results[f'Y_state_{s}'] = np.zeros((1, 1), dtype=np.complex64)
        validate_tddft(path, 2, 'valid', 'float32')

        # Asking for a third state, changing the settings, or changing the precision all invalidate
        # it, as does an extra amplitude that no saved state owns.
        for n, signature, precision in [(3, 'valid', 'float32'), (2, 'changed', 'float32'), (2, 'valid', 'float64')]:
            with self.assertRaises(ValueError):
                validate_tddft(path, n, signature, precision)
        with h5py.File(path, 'r+') as results:
            results['X_state_3'] = np.ones((1, 1))
        with self.assertRaises(ValueError):
            validate_tddft(path, 2, 'valid', 'float32')

    def test_stage_signatures_ignore_controls_and_track_physics(self):
        # Switching a stage from reuse to run is a control, not physics, so the signature holds and
        # the old result stays usable. Changing the functional does change the physics.
        job = self.load()
        system = read_systems(job)[0]
        first = calculation_signature(job, system, 'tddft', self.directory)
        job['stages']['tddft'] = 'run'
        self.assertEqual(first, calculation_signature(job, system, 'tddft', self.directory))
        job['tddft']['xc'] = 'pbe'
        self.assertNotEqual(first, calculation_signature(job, system, 'tddft', self.directory))

    def test_plot_config_defaults_and_rejections(self):
        # A figure given on its own should come back with the defaults filled in.
        path = self.directory / 'plots.yaml'
        path.write_text(yaml.safe_dump({'figures': [{'type': 'overview'}]}))
        result = load_plot_config(path)
        self.assertEqual(result['grouping']['energy_tolerance_eV'], .01)
        self.assertEqual(len(result['figures']), 1)
        self.assertEqual(result['figures'][0]['planes'], ['xy'])

        # A shape a figure has no layout for, a third-width frame asked to be rectangular, an
        # impossible (l,m) pair, an unknown figure, and a setting that no longer exists.
        for config in [dict(figures=[{'type': 'real', 'shape': 'rectangular'}]),
                       dict(figures=[{'type': 'angular_mean', 'width': 'third', 'shape': 'rectangular'}]),
                       dict(figures=[{'type': 'structure_mode', 'modes': [[1, 2]]}]),
                       dict(figures=[{'type': 'bands'}]),
                       dict(figures=[{'type': 'overview'}], legacy={'bands': False})]:
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_plot_config(path)

    def test_shared_figure_requests_select_each_calculation_mode(self):
        # One configuration should serve every method: each mode takes the figures it can draw and
        # ignores the rest, so a crystal keeps the structure function plots and drops the molecular
        # slices. f_sq_lm is spherical only, since the other methods never build the coefficients.
        requests = [{'type': kind} for kind in ('overview', 'intensity', 'f_sq_lm',
                                                'transition_density', 'real',
                                                'imag', 'angular_mean', 'angular_std',
                                                'structure_mode', 'structure_modes',
                                                'structure_integral', 'structure_mode_integrals')]
        path = self.directory / 'plots.yaml'
        path.write_text(yaml.safe_dump({'width': 'full', 'shape': 'square', 'figures': requests}))
        figures = load_plot_config(path)['figures']
        self.assertEqual(figures[1]['planes'], ['yz'])
        for mode in ('spherical', 'cartesian', 'fourier', 'crystal'):
            selected = [request['type'] for request in figures_for_mode(figures, mode)]
            if mode == 'crystal':
                expected = ['overview', 'structure_mode', 'structure_modes',
                            'structure_integral', 'structure_mode_integrals']
            else:
                expected = ['overview', 'intensity', 'real', 'imag', 'angular_mean', 'angular_std']
                if mode == 'spherical':
                    expected.insert(2, 'f_sq_lm')
                if mode == 'fourier':
                    expected.insert(2, 'transition_density')
            self.assertEqual(selected, expected)
        self.assertEqual(figures_for_mode([{'type': 'structure_mode'}], 'spherical'), [])
        self.assertEqual(len(figures), len(requests))

    def test_plot_file_types(self):
        # One format or a list of them, kept in the order written.
        path = self.directory / 'plots.yaml'
        for supplied, expected in [('png', ['png']), ('pdf', ['pdf']),
                                   (['png', 'pdf'], ['png', 'pdf']),
                                   (['pdf', 'png'], ['pdf', 'png'])]:
            path.write_text(yaml.safe_dump({'file_type': supplied, 'figures': [{'type': 'overview'}]}))
            self.assertEqual(load_plot_config(path)['file_type'], expected)

        # A format that is not offered, an empty list, a repeat, and the wrong kind of value.
        for supplied in ['both', 'svg', [], ['png', 'png'], ['png', 'svg'], True, {'png': True}]:
            path.write_text(yaml.safe_dump({'file_type': supplied, 'figures': [{'type': 'overview'}]}))
            with self.assertRaises(ValueError):
                load_plot_config(path)

    def test_plot_width_and_shape_combinations(self):
        # One request expands into a figure per width and shape. A third-width frame is always
        # square, so that combination collapses to one. A full page fits eight curves, the
        # narrower ones four, unless a count is given.
        path = self.directory / 'plots.yaml'
        request = {'type': 'f_sq_lm', 'states': [3], 'modes': [[2, 2]],
                   'width': ['full', 'half', 'third'], 'shape': ['square', 'rectangular']}
        path.write_text(yaml.safe_dump({'figures': [request]}))
        figures = load_plot_config(path)['figures']
        self.assertEqual([(figure['width'], figure['shape']) for figure in figures],
                         [('full', 'square'), ('full', 'rectangular'),
                          ('half', 'square'), ('half', 'rectangular'), ('third', 'square')])
        self.assertEqual([figure['count'] for figure in figures], [8, 8, 4, 4, 4])

        # Whatever the layout, every copy keeps the states and harmonics that were asked for.
        for figure in figures:
            self.assertEqual(figure['states'], [3])
            self.assertEqual(figure['modes'], [[2, 2]])
        request.pop('shape')
        request['count'] = 2
        path.write_text(yaml.safe_dump({'figures': [request]}))
        figures = load_plot_config(path)['figures']
        self.assertEqual([figure['shape'] for figure in figures], ['square', 'rectangular', 'square', 'rectangular', 'square'])
        self.assertEqual([figure['count'] for figure in figures], [2, 2, 2, 2, 2])

    def test_plot_lists_keep_fixed_layouts(self):
        # A momentum plane is square whatever shape is asked for, since both axes are a q in keV,
        # and the overview only exists at full width. Asking for everything must not override that.
        path = self.directory / 'plots.yaml'
        for kind in ('intensity', 'real', 'imag', 'overview'):
            request = {'type': kind, 'width': ['full', 'half', 'third'],
                       'shape': ['square', 'rectangular']}
            path.write_text(yaml.safe_dump({'figures': [request]}))
            figures = load_plot_config(path)['figures']
            if kind == 'overview':
                self.assertEqual(len(figures), 1)
                self.assertEqual(figures[0]['width'], 'full')
            else:
                self.assertEqual([figure['width'] for figure in figures], ['full', 'half', 'third'])
                self.assertEqual([figure['shape'] for figure in figures], ['square'] * 3)

    def test_invalid_plot_width_and_shape_lists(self):
        # Nothing to draw at, an unknown size, the same one twice, and the wrong kind of value.
        path = self.directory / 'plots.yaml'
        for field, value in [('width', []), ('width', ['half', 'invalid']),
                             ('width', ['full', 'full']), ('width', True),
                             ('shape', []), ('shape', ['square', 'invalid']),
                             ('shape', ['square', 'square']), ('shape', None)]:
            path.write_text(yaml.safe_dump({'figures': [{'type': 'angular_mean', field: value}]}))
            with self.assertRaises(ValueError):
                load_plot_config(path)

    def test_plot_layout_defaults_and_figure_overrides(self):
        # The widths and shapes at the top are the defaults, and a figure may replace either field
        # on its own. The expected list below is the whole expansion, in configuration order.
        path = self.directory / 'plots.yaml'
        config = {'width': ['full', 'half'], 'shape': ['rectangular'], 'figures': [
            {'type': 'angular_mean'},
            {'type': 'angular_std', 'shape': 'square'},
            {'type': 'f_sq_lm', 'width': 'third'},
            {'type': 'structure_integral', 'width': 'half', 'shape': ['square', 'rectangular']},
            {'type': 'intensity'},
            {'type': 'overview'},
        ]}
        path.write_text(yaml.safe_dump(config))
        figures = load_plot_config(path)['figures']
        actual = [(figure['type'], figure['width'], figure['shape']) for figure in figures]
        self.assertEqual(actual, [
            ('angular_mean', 'full', 'rectangular'), ('angular_mean', 'half', 'rectangular'),
            ('angular_std', 'full', 'square'), ('angular_std', 'half', 'square'),
            ('f_sq_lm', 'third', 'square'),
            ('structure_integral', 'half', 'square'), ('structure_integral', 'half', 'rectangular'),
            ('intensity', 'full', 'square'), ('intensity', 'half', 'square'),
            ('overview', 'full', 'rectangular'),
        ])
        # The overview only exists at full width, so narrowing the default must not produce two of
        # them, nor none.
        config['width'] = 'half'
        path.write_text(yaml.safe_dump(config))
        figures = load_plot_config(path)['figures']
        self.assertEqual(sum(figure['type'] == 'overview' for figure in figures), 1)
        for field, value in [('width', []), ('shape', ['invalid']), ('width', ['full', 'full'])]:
            path.write_text(yaml.safe_dump({**config, field: value}))
            with self.assertRaises(ValueError):
                load_plot_config(path)

    def test_crystal_sum_rule_request(self):
        # The sum rule is a crystal figure, so a molecular run should not pick it up.
        path = self.directory / 'plots.yaml'
        path.write_text(yaml.safe_dump({'width': 'half', 'shape': 'rectangular',
                                       'figures': [{'type': 'structure_sum_rule', 'modes': [[0, 0], [2, 2]]}]}))
        figures = load_plot_config(path)['figures']
        self.assertEqual([figure['type'] for figure in figures_for_mode(figures, 'crystal')], ['structure_sum_rule'])
        self.assertEqual(figures[0]['modes'], [[0, 0], [2, 2]])
        self.assertEqual(figures_for_mode(figures, 'spherical'), [])


class MolecularQuantities(unittest.TestCase):
    """Check grouped intensities and physical sampling, without drawing figures."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'form_factors.h5'

    def tearDown(self):
        self.temporary.cleanup()

    def test_neighbour_chaining_preserves_original_indices(self):
        # States 3, 5 and 8 are each within the tolerance of the one below, so they chain into a
        # single degenerate group even though 3 and 8 are 16 meV apart. 12 is far off on its own.
        self.assertEqual(group_transitions({8: 1.016, 3: 1.0, 5: 1.008, 12: 2.0}, .01), [(3, 5, 8), (12,)])
        self.assertEqual(group_transitions({1: 1., 2: 1.01, 3: 1.02, 4: 1.0300001}, .01), [(1, 2, 3), (4,)])

    def test_grouped_moments_and_unitary_invariance(self):
        # Two amplitudes proportional to q cos(θ) and q sin(θ)cos(ϕ), so the angular mean and
        # spread of their summed intensity are known in closed form.
        q = np.linspace(0, 2, 7)
        theta, phi = np.linspace(0, np.pi, 97), np.linspace(0, 2*np.pi, 97)
        radius, th, ph = np.meshgrid(q, theta, phi, indexing='ij')
        first = radius * np.cos(th)
        second = radius * np.sin(th)*np.cos(ph)
        grid = dict(q=q, theta=theta, phi=phi)
        molecular_file(self.path, 'spherical', grid, [first, second], [1, 1.005])
        data = MolecularData(self.path)

        # The two energies are 5 meV apart, so selecting either one draws both.
        self.assertEqual(data.select_groups([2]), [(1, 2)])
        mean, deviation = data.moments((1, 2))
        np.testing.assert_allclose(mean, 2*q*q/3, atol=.001)
        np.testing.assert_allclose(deviation, q*q*np.sqrt(4/45), atol=.001)

        # Mixing two degenerate states is a change of basis within the group, and the summed
        # intensity cannot tell, so the moments must come back unchanged.
        molecular_file(self.path, 'spherical', grid, [(first+second)/np.sqrt(2), (first-second)/np.sqrt(2)], [1, 1.005])
        rotated = MolecularData(self.path)
        for actual, expected in zip(rotated.moments((1, 2)), (mean, deviation)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

        # The moments are of the intensity, so doubling the amplitude quadruples both of them.
        molecular_file(self.path, 'spherical', grid, [2*first, 2*second], [1, 1.005])
        scaled = MolecularData(self.path)
        for actual, expected in zip(scaled.moments((1, 2)), (4*mean, 4*deviation)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

        # A phase only means anything for one state, so a group cannot be asked for its real part.
        with self.assertRaisesRegex(ValueError, 'individual'):
            data.plane((1, 2), 'yz', 'real')

    def test_cartesian_and_fourier_zero_plane_and_angular_domain(self):
        # An amplitude linear in each axis, so a plane through the origin has a known slope and any
        # mixed-up axis shows up as the wrong gradient. The box is deliberately lopsided.
        grid = dict(q_x=np.array([-1, -.3, .7, 1]), q_y=np.linspace(-2, 2, 5), q_z=np.linspace(-3, 3, 7))
        x, y, z = np.meshgrid(*grid.values(), indexing='ij')
        amplitude = x + 2*y + 3j*z
        for mode in ['cartesian', 'fourier']:
            molecular_file(self.path, mode, grid, [amplitude])
            data = MolecularData(self.path, sampling={'n_q': 7, 'n_theta': 97, 'n_phi': 97})
            horizontal, vertical, real = data.plane((1,), 'yz', 'real')
            yy, zz = np.meshgrid(horizontal, vertical)
            np.testing.assert_allclose(real, 2*yy, atol=1e-12)
            np.testing.assert_allclose(data.plane((1,), 'yz', 'imag')[2], 3*zz, atol=1e-12)

            # Angular averages need a sphere the box encloses, so |q| can only reach the shortest
            # half width, which is 1 along q_x here. The mean of |q_x + 2q_y + 3i q_z|^2 over a
            # shell is (1 + 4 + 9) q^2 / 3.
            self.assertEqual(data.angular_grid[0][-1], 1)
            mean, deviation = data.moments((1,))
            np.testing.assert_allclose(mean, 14*data.angular_grid[0]**2/3, atol=.002)

            # Only the spherical method builds the coefficients, so asking here must fail.
            with self.assertRaisesRegex(ValueError, 'no saved f_sq_lm'):
                data.coefficients((1,))

    def test_transition_density_plane_is_real_space(self):
        # A density linear in each axis, on a real-space box whose three axes have different
        # lengths, so a plane through the origin has a known slope and a swapped axis shows up.
        grid = dict(q_x=np.linspace(-1, 1, 3), q_y=np.linspace(-1, 1, 3), q_z=np.linspace(-1, 1, 3),
                    x=np.array([-2., -.5, .5, 2.]), y=np.linspace(-3, 3, 5), z=np.linspace(-4, 4, 7))
        amplitude = np.zeros((3, 3, 3), dtype=complex)
        x, y, z = np.meshgrid(grid['x'], grid['y'], grid['z'], indexing='ij')
        molecular_file(self.path, 'fourier', grid, [amplitude], densities=[x + 2*y + 3*z])
        data = MolecularData(self.path)

        # The density comes back in (x, y, z) order, undoing the axis reversal HDF5 applies.
        self.assertEqual(data.density(1).shape, (4, 5, 7))
        horizontal, vertical, values = data.density_plane(1, 'yz')
        np.testing.assert_array_equal(horizontal, grid['y'])
        yy, zz = np.meshgrid(horizontal, vertical)
        np.testing.assert_allclose(values, 2*yy + 3*zz, atol=1e-12)
        np.testing.assert_allclose(data.density_plane(1, 'xy')[2],
                                   np.add(*np.meshgrid(grid['x'], 2*grid['y'])), atol=1e-12)

        # Without a saved density there is nothing to draw.
        with h5py.File(self.path, 'r+') as results:
            del results['transitions/1/transition_density']
        with self.assertRaisesRegex(ValueError, 'no saved transition_density'):
            MolecularData(self.path).density_plane(1, 'yz')


class JuliaHDF5(unittest.TestCase):
    """Exercise the real Julia writer and its Python reader together."""

    @classmethod
    def setUpClass(cls):
        # Write the fixtures with the real Julia writer, so what is read back is the layout the
        # calculations actually produce rather than a Python imitation of it.
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name) / 'form_factors.h5'
        subprocess.run(['julia', f'--project={ROOT}', str(ROOT/'test/write_test_results.jl'), str(cls.path)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_writer_axes_precision_units_and_atomic_replacement(self):
        # The fixture entries spell out their indices as digits in the Julia axis order
        # (transition, q, θ, ϕ). The reader hands back (θ, ϕ, q), so entry [1,3,2] of transition 2
        # is θ = 2, ϕ = 4, q = 3, which reads 1324. A permuted axis gives a jumbled number.
        data = MolecularData(self.path)
        self.assertEqual(data.amplitude(2).shape, (4, 5, 3))
        self.assertEqual(data.amplitude(2).dtype, np.complex64)
        self.assertEqual(data.amplitude(2)[1, 3, 2], 1324-1324j)
        self.assertEqual(data.coefficients((2,))[1, 2], 123)

        # Transition 5 carries the larger coefficients, so it ranks first, and the two transitions
        # are far enough apart in energy to stay separate groups. Intensities add.
        self.assertEqual(data.ranked_groups(), [(5,), (2,)])
        self.assertEqual(data.select_groups('dominant', 1), [(5,)])
        np.testing.assert_allclose(data.coefficients((2, 5)), data.coefficients((2,)) + data.coefficients((5,)))

        # The original indices, the grids kept once rather than per transition, the shared Cartesian
        # tensors, and units on the physical quantities but not on the dimensionless ones.
        with h5py.File(self.path) as results:
            self.assertEqual(set(results['transitions']), {'2', '5'})
            self.assertNotIn('grid', results['transitions/2'])
            self.assertIn('cartesian_tensors/V_x', results)
            self.assertEqual(results['grid/q'].attrs['units'], b'keV')
            self.assertEqual(results['transitions/2/Delta_E'].attrs['units'], b'eV')
            self.assertEqual(dict(results['transitions/2/f_s'].attrs), {})
            self.assertEqual(results['transitions/2/Delta_E'].dtype, np.float32)

        # The fixture deliberately failed two writes after this file was finished, and neither may
        # have left a half-written file behind.
        self.assertFalse(Path(str(self.path)+'.tmp').exists())

    def test_cartesian_and_fourier_julia_array_order(self):
        # The same digits on a box grid, where the axes are (transition, q_x, q_y, q_z) and the
        # reader gives them back in that order, so entry [2,1,3] is q_x = 3, q_y = 2, q_z = 4.
        for mode in ('cartesian', 'fourier'):
            path = Path(str(self.path) + '.' + mode)
            data = MolecularData(path)
            self.assertEqual(data.amplitude(2).shape, (3, 4, 5))
            self.assertEqual(data.amplitude(2)[2, 1, 3], 1324-1324j)
            self.assertEqual(data.amplitude(2).dtype, np.complex64)

            # The Fourier method adds the real-space density, so check its axes and units survive
            # too. It is real space, so the grid is in Angstroms rather than keV.
            with h5py.File(path) as results:
                self.assertNotIn('grid', results['transitions/2'])
                if mode == 'fourier':
                    self.assertEqual(results['transitions/2/transition_density'].shape, (5, 4, 3))
                    self.assertEqual(results['transitions/2/transition_density'].attrs['units'], b'angstrom^-3')
                    self.assertEqual(results['transitions/2/transition_density'].dtype, np.float32)
                    self.assertEqual(results['grid/x'].attrs['units'], b'angstrom')

    def test_reuse_rejects_missing_transition(self):
        # The fixture is exactly what this job asks for, so reusing it is allowed.
        job = {'mode': 'spherical', 'transitions': [2, 5], 'nstates': 5,
               'execution': {'precision': 'float32'}, 'spherical': {'save': ['f_s', 'f_sq_lm']}}
        validate_form_factors(job, self.path, 'fixture')

        # Changed settings, or a change of precision, and the saved file no longer belongs.
        with self.assertRaises(ValueError):
            validate_form_factors(job, self.path, 'different')
        job['execution']['precision'] = 'float64'
        with self.assertRaisesRegex(ValueError, 'precision'):
            validate_form_factors(job, self.path, 'fixture')

        # Nor may a transition that was asked for simply be absent.
        job['execution']['precision'] = 'float32'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.h5'
            shutil.copyfile(self.path, path)
            with h5py.File(path, 'r+') as results:
                del results['transitions/5']
            with self.assertRaisesRegex(ValueError, 'transition selection'):
                validate_form_factors(job, path, 'fixture')


class StructureFunctionQuantities(unittest.TestCase):
    """Keep the saved structure-function normalisation and signed energy integrals."""

    def test_integrals_and_dominance(self):
        # A flat monopole of 8 over three 0.5 eV bins integrates to 12. The (1,-1) harmonic changes
        # sign across q and the (1,0) one does not, so ranking by absolute area picks (1,0) and
        # never the monopole, which is plotted on its own.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crystal.h5'
            with h5py.File(path, 'w') as results:
                results['q_grid'] = [0., 1., 2.]
                results['E_grid_eV'] = [5., 5.5, 6.]
                values = np.zeros((9, 3, 3))
                values[0] = 8
                values[1] = [10, 0, -10]
                values[2] = 1
                results['S_lm'] = values
            data = CrystalData(path)
            np.testing.assert_array_equal(data.mode((0, 0)), 8*np.ones((3, 3)))
            np.testing.assert_array_equal(data.integrals[:, 0], [12, 12, 12])

            # The angular average is the monopole over sqrt(4π), from Y_00.
            np.testing.assert_allclose(data.angular_mean, 8/np.sqrt(4*np.pi))
            np.testing.assert_allclose(data.integrated_mean, 12/np.sqrt(4*np.pi))
            self.assertEqual(data.ranked_modes(1), [(1, 0)])
            with self.assertRaises(ValueError):
                data.mode((3, 0))

    def test_state_sum_normalisation_and_incoherent_file(self):
        # Summing the coefficients over states and multiplying by 2π/V_cell is what turns them into
        # the structure function, so that factor has to appear exactly once.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crystal.h5'
            coefficients = np.arange(9*3*2, dtype=np.float32).reshape(9, 3, 2)
            with h5py.File(path, 'w') as results:
                results['q_grid'] = np.array([0, 1, 2], dtype=np.float32)
                results['V_cell_A3'] = 17.
                results['f_sq_lm'] = coefficients
                results['crystal_order'] = np.bytes_('incoherent')
            data = CrystalData(path)
            expected = coefficients.sum(axis=2, dtype=float).T * (2*np.pi/17.)
            np.testing.assert_allclose(data.state_sums, expected)
            self.assertEqual(data.ranked_modes(1, data.state_sums), [(2, 2)])

            # No energy axis was saved, so there is no structure function to read yet.
            with self.assertRaisesRegex(ValueError, 'no structure function'):
                data.mode((0, 0))

            # Spread that same sum over three energy bins in any proportion, and integrating the
            # energy axis has to give it straight back.
            with h5py.File(path, 'r+') as results:
                results['E_grid_eV'] = [4., 4.5, 5.]
                results['S_lm'] = expected.T[:, :, None] * np.array([.2, .3, .5])[None, None, :] / .5
            integrated = CrystalData(path)
            np.testing.assert_allclose(integrated.integrals, integrated.state_sums)

    def test_incoherent_reference_is_read_and_gates_the_sum_rule(self):
        # Four crystal bands built out of two monomer transitions on two molecules, so the mixing is
        # a change of basis and the total intensity has to be the same either way.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crystal.h5'
            states = np.arange(1, 9*3*4 + 1, dtype=np.float32).reshape(9, 3, 4)
            # The same intensity, carried by two monomer transitions rather than four crystal bands.
            reference = states.reshape(9, 3, 2, 2).sum(axis=3)
            summed = states.sum(axis=2, dtype=float).T * (2*np.pi/11.)
            with h5py.File(path, 'w') as results:
                results['q_grid'] = np.array([0, 1, 2], dtype=np.float32)
                results['E_grid_eV'] = [4., 4.5]
                results['V_cell_A3'] = 11.
                results['f_sq_lm'] = states
                results['f_sq_lm_incoherent'] = reference
                results['S_lm'] = summed.T[:, :, None] * np.array([.4, .6])[None, None, :] / .5
                results['crystal_order'] = np.bytes_('coherent')
            data = CrystalData(path)
            # The sum rule the plot draws: the mixed bands and the unmixed molecules must agree.
            np.testing.assert_allclose(data.incoherent_sums, data.state_sums)
            np.testing.assert_allclose(data.integrals, data.incoherent_sums, rtol=1e-6)
            requests = [{'type': 'structure_sum_rule'}, {'type': 'structure_integral'}]
            self.assertEqual(figures_for_data(requests, data.saved), requests)

            # With no reference saved there is nothing to compare against, so the plot is dropped.
            with h5py.File(path, 'r+') as results:
                del results['f_sq_lm_incoherent']
            bare = CrystalData(path)
            self.assertEqual(figures_for_data(requests, bare.saved), requests[1:])
            with self.assertRaisesRegex(ValueError, 'f_sq_lm_incoherent'):
                bare.incoherent_sums

    def test_energy_integral_uses_full_float32_grid(self):
        # 451 bins of 10 meV in float32, where neighbouring energies round to the same value, so the
        # step has to come from the whole axis rather than from one difference. A normalised S_lm
        # must still integrate to one.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crystal.h5'
            step = np.float32(.01)
            energy = np.float32(5.43) + np.arange(451, dtype=np.float32)*step
            with h5py.File(path, 'w') as results:
                results['q_grid'] = [0., 1., 2.]
                results['E_grid_eV'] = energy
                results['S_lm'] = np.ones((1, 3, len(energy)), dtype=np.float32) / (len(energy)*step)
            np.testing.assert_allclose(CrystalData(path).integrals, 1., rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
