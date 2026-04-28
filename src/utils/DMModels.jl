# This module contains DM models to compute scattering rates with VSDM.

module DMModels

using VectorSpaceDarkMatter

const VSDM = VectorSpaceDarkMatter

const v_Earth = 250.0 # Earth wind velocity in km/s.
const v_halo = 230.0 # Halo wind velocity in km/s.
const v_max = 960.0 # Escape velocity in km/s.

function spherical_halo()
    """
    Returns the spherical halo velocity model that can be passed to the VSDM rate function, defined by:

        f(v) = (1/π^{3/2} vHalo^3) * exp(-|\vec{v}-\vec{vEarth}|^2 / vHalo^2)

    where we take vEarth along the -z-axis.
    
    # Arguments:
    - None.

    # Returns:
    - spherical_halo::VSDM.GaussianF: The spherical halo velocity model.

    """
    return VSDM.GaussianF(1.0, VSDM.cart_to_sph([0.0,0.0,-v_Earth*VSDM.km_s]), v_halo*VSDM.km_s)
end

end