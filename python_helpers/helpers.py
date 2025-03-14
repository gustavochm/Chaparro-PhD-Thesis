

# Helper function to get the alpha parameter
def helper_get_alpha(lambda_r, lambda_a):
    """
    Helper function to get the alpha parameter

    Parameters
    ----------
    lambda_r : float or array
        lambda_r parameter
    lambda_a : float or array
        lambda_a parameter

    Returns
    -------
    alpha : float or array
        alpha parameter
    """
    c_alpha = (lambda_r / (lambda_r-lambda_a)) * (lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a))
    alpha = c_alpha*(1./(lambda_a-3) - 1./(lambda_r-3))
    return alpha
