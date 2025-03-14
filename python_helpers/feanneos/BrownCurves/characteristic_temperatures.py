import jax.numpy as jnp


#####################
# Boyle Temperature
#####################
def of_boyle_temp(T, alpha, dhelmholtz_drho_fun):
    """
    Objective function for the Boyle temperature.
    B(T) = 0

    Parameters
    ----------
    T : float
        Temperature.
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    dhelmholtz_drho_fun : Callable
        Function that returns the reduced Helmholtz energy and its first derivative.
    
    Returns
    -------
    B_boyle : jnp.ndarray
        Second virial coefficient.
    """

    Tboyle = jnp.atleast_1d(T) 
    out = dhelmholtz_drho_fun(jnp.atleast_1d(alpha), jnp.array([0.]), Tboyle )
    Ares, dAres_drho = out
    B_boyle = dAres_drho/Tboyle
    return B_boyle 


#####################
# Charles Temperature
#####################
def of_charles_temp(T, alpha, d2helmholtz_drho2_dT_fun):
    """
    Objective function for the Charles temperature.
    dB/dT - B/T = 0

    Parameters
    ----------
    T : float
        Temperature.
    alpha : float
        van der Waals alpha parameter for the Mie fluid.
    d2helmholtz_drho2_dT_fun : Callable
        Function that returns the reduced Helmholtz energy and its first and second derivatives.

    Returns
    -------
    of_charles : jnp.ndarray
        Objective function for the Charles temperature (B/T - dB/dT)
    """
    Tcharles = jnp.atleast_1d(T)
    out = d2helmholtz_drho2_dT_fun(jnp.atleast_1d(alpha), jnp.array([0.]), Tcharles)
    Ares, dAres_drho, d2Ares_drho2, d2Ares_drho_dT = out
    B_charles = dAres_drho/Tcharles
    dB_dT_charles = - dAres_drho / Tcharles**2 + d2Ares_drho_dT / Tcharles
    of_charles = dB_dT_charles -  B_charles / Tcharles
    return of_charles

#####################
# Amagat Temperature
#####################
def of_amagat_temp(T, alpha, d2helmholtz_drho2_dT_fun):
    """
    Objective function for the Amagat temperature.
    dB/dT_amagat = 0

    Parameters
    ----------
    T : float
        Temperature.
    alpha : float  
        van der Waals alpha parameter for the Mie fluid.
    d2helmholtz_drho2_dT_fun : Callable
        Function that returns the reduced Helmholtz energy and its first and second derivatives.
    """
    Tamagat = jnp.atleast_1d(T) 
    out = d2helmholtz_drho2_dT_fun(jnp.atleast_1d(alpha), jnp.array([0.]), Tamagat)
    Ares, dAres_drho, d2Ares_drho2, d2Ares_drho_dT = out
    # dB_dT_amagat = - dAres_drho / Tamagat**2 + d2Ares_drho_dT / Tamagat
    dB_dT_amagat = - dAres_drho + d2Ares_drho_dT * Tamagat
    return dB_dT_amagat