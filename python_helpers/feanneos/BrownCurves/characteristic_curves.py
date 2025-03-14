import jax.numpy as jnp


def zeno_obj(rhoad, alpha, Tad, dhelmholtz_drho_fun):
    """
    Objetive function to compute Zeno Curve
    Z = P / (rho * T) = 1

    Parameters
    ----------
    rhoad : float
            Reduced density
    alpha : float
            alpha van der Waals parameter
    Tad : float
          Reduced temperature
    dhelmholtz_drho_fun : function
                          Function to compute the derivative of the Helmholtz

    Returns
    -------
    dAres_drho : float
                 Derivative of the Helmholtz energy with respect to the reduced density

    """
    alpha = jnp.atleast_1d(alpha)
    rhoad = jnp.atleast_1d(rhoad)
    Tad = jnp.atleast_1d(Tad)
    out = dhelmholtz_drho_fun(alpha, rhoad, Tad)
    Ares, dAres_drho = out
    return dAres_drho


def boyle_obj(rhoad, alpha, Tad, d2helmholtz_drho2_fun):
    """
    Objetive function to compute Boyle Curve
    dZ/dV = 0

    Parameters
    ----------
    rhoad : float
            Reduced density
    alpha : float
            alpha van der Waals parameter
    Tad : float
          Reduced temperature
    d2helmholtz_drho2_fun : function
                          Function to compute the derivative of the Helmholtz

    Returns
    -------
    dZ_dV : float
            Derivative of the compressibility factor wrt the reduced volume

    """
    alpha = jnp.atleast_1d(alpha)
    rhoad = jnp.atleast_1d(rhoad)
    Tad = jnp.atleast_1d(Tad)
    out = d2helmholtz_drho2_fun(alpha, rhoad, Tad)
    Ares, dAres_drho, d2Ares_drho2 = out
    # dZ_dV = (-rhoad**2/Tad) * (dAres_drho + rhoad*d2Ares_drho2)
    dZ_dV = dAres_drho + rhoad*d2Ares_drho2
    return dZ_dV


def charles_obj(rhoad, alpha, Tad, thermal_expansion_coeff_fun):
    """
    Objetive function to compute Charles Curve
    (dZ/dT)_P = 0

    Parameters
    ----------
    rhoad : float
            Reduced density
    alpha : float
            alpha van der Waals parameter
    Tad : float
          Reduced temperature
    thermal_expansion_coeff_fun : function
                                  Function to compute the thermal expansion coefficient

    Returns
    -------
    of : float
         Tad * alphap - 1.

    """
    alpha = jnp.atleast_1d(alpha)
    rhoad = jnp.atleast_1d(rhoad)
    Tad = jnp.atleast_1d(Tad)
    alphap = thermal_expansion_coeff_fun(alpha, rhoad, Tad)
    of = Tad * alphap - 1.
    return of


def amagat_obj(rhoad, alpha, Tad, d2helmholtz_fun):
    """
    Objetive function to compute Charles Curve
    (dZ/dT)_rho = 0

    Parameters
    ----------
    rhoad : float
            Reduced density 
    alpha : float   
            alpha van der Waals parameter
    Tad : float
          Reduced temperature
    d2helmholtz_fun : function
                      Function to compute the derivative of the Helmholtz

    Returns
    -------
    dZ_dT : float
         Tad * alphap - 1.

    """
    alpha = jnp.atleast_1d(alpha)
    rhoad = jnp.atleast_1d(rhoad)
    Tad = jnp.atleast_1d(Tad)

    out = d2helmholtz_fun(alpha, rhoad, Tad)
    Ares, dAres_drho, dAres_dT, d2Ares_drho2, d2Ares_dT2, d2Ares_drho_dT = out

    # dZ_dT = (rhoad/Tad) * d2Ares_drho_dT
    # dZ_dT -= (rhoad/Tad**2) * dAres_drho
    dZ_dT = Tad*d2Ares_drho_dT - dAres_drho
    return dZ_dT
