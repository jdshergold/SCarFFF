# This module contains DM models to compute scattering rates with VSDM.

module DMModels

using VectorSpaceDarkMatter
using SpecialFunctions: erf

const VSDM = VectorSpaceDarkMatter

const v_0 = 238.0 # Velocity dispersion of the halo in km/s.
const v_Earth = 250.0 # Lab speed relative to the galactic rest frame in km/s.
const v_esc = 544.0 # Galactic escape speed in km/s.
const v_max = v_Earth + v_esc # Largest lab frame speed the halo reaches, in km/s.

function standard_halo()
    """
    Returns the Standard Halo Model velocity distribution, defined by:

        g(v) = (1/N_0) exp(-|v + v_E|^2 / v_0^2) Θ(v_esc - |v + v_E|)

    with N_0 fixed by ∫d³v g(v) = 1. The wind direction -v_E is along +z, so only the m = 0
    modes survive and the distribution is axisymmetric.

    # Arguments:
    - None.

    # Returns:
    - standard_halo::VSDM.f_uSph: The Standard Halo Model velocity distribution.

    """

    dispersion = v_0 * VSDM.km_s
    escape = v_esc * VSDM.km_s
    earth = v_Earth * VSDM.km_s

    # Analytic normalisation factor for the truncated Gaussian.
    ratio = escape / dispersion
    normalisation = π^1.5 * dispersion^3 *
        (erf(ratio) - 2 / sqrt(π) * ratio * exp(-ratio^2))

    function halo(velocity_sph)
        speed, theta = velocity_sph[1], velocity_sph[2]
        # |v + v_E|², the galactic frame speed, with the centre of the Gaussian at +v_E along z.
        galactic_squared = speed^2 - 2 * speed * earth * cos(theta) + earth^2
        galactic_squared > escape^2 && return 0.0
        return exp(-galactic_squared / dispersion^2) / normalisation
    end

    return VSDM.f_uSph(halo; phi_symmetric = true)
end

end
