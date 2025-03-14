import numpy as np


def Tsle_correlation_khrapak(rho, lambda_r, params=[212., 15., 10., 42.]):

    # Correlation for the SLE boundaries for lambda_r - 6 Mie particles
    # Khrapak, S. A., Chaudhuri, M., & Morfill, G. E. (2011). 
    # Freezing of Lennard-Jones-type fluids. The Journal of Chemical Physics, 134(5), 054120. https://doi.org/10.1063/1.3552948
    p1, p2, p3, p4 = params
    gamma = lambda_r
    L_gamma = p1 + p2 * gamma*np.exp(-p3/gamma)
    T_freeze = 1./L_gamma * (lambda_r/(lambda_r-6.)) * (lambda_r/6.)**(6./(lambda_r-6.)) * (lambda_r*(lambda_r+1.)*rho**(lambda_r/3.) - p4*rho**2)
    return T_freeze
