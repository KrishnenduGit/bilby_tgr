import numpy as np
import lal

# -------------------------- Useful functions -------------------------- #

def to_SI(expr):

    """
    Convert expressions for absolute deviations to the SI units.

    1. Factor of (c^8/G^4) to convert the prefactor alpha_GB^2/M^4 to SI units.
    2. Factor of (MSUN_SI^-4) to convert M^4 from solar masses to kg
    3. Factor of 10^12 to convert alpha_GB^2 from km^4 to m^4

    Parameters
    ----------
    expr (float): Expression for PPE term in natural units w/ [M] = Solar Masses & [L] = km.

    Returns
    -------
    expr_SI (float): Expression for PPE term in SI units 
        
    """

    expr_SI = (lal.C_SI**8)*(lal.G_SI**-4) *  (lal.MSUN_SI**-4) * (1e12) * expr

    return expr_SI

def calculate_sensitivity_EsGB(chi) -> float:

    """
    Calculate the spin-dependent factor for EsGB corrections

    s_A = 2 * (sqrt(1 - a_A^2) - 1 + a_A^2) / a_A^2 

    where a_A is the dimensionless spin of object A using the LAL notation, denoted chi_A in the literature. 

    Parameters
    ----------
    a: float
        Dimensionless spin parameter (a_A) of the object
    Returns
    -------
    s: float
        Spin-dependent factor for EsGB corrections
    """

    if chi   > 0.0:

        chi2 = chi * chi

        return 2.0 * (np.sqrt(1.0 - chi2) - 1.0 + chi2) / chi2
    
    elif chi < 0.0:

        raise ValueError("Dimensionless spin must be non-negative for black holes.")
    
    else:

        return 1.0 

# -------------------------- EsGB PPE phase coefficients: TGR Parameterization = 2 -------------------------- #

"""
Calculate the EsGB PPE phase corrections up to 2PN using 
coefficients in Erratum Phys. Rev. D 106, 069901 (2022) 

These coefficients are added as absolute deviations to the GR phase.
i.e phi_n = phi_n_GR + dchi_n.

Coefficients are scaled by 3 / (128 * eta) as this prefactor is included
by lal routines at a later stage.

"""

def edgb_dchi_m2(
    mass_1: float,
    mass_2: float,
    s_1: float,
    s_2: float,
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase -1PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2.

    Returns
    -------
    dchi_m2: float
        Deviation coefficient of the waveform phase at -1PN order

    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    c = -(5*np.pi/448)*(((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)/((eta**5)*(M**4))

    beta = c*(alpha_gb**2)/(M**4)

    dchi_m2 = to_SI(beta) / (3/(128*eta))

    return dchi_m2


def edgb_dchi_0(
    mass_1: float,
    mass_2: float,
    s_1: float,
    s_2: float,
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase 0PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2.

    Returns
    -------
    dchi_0: float
        Deviation coefficient of the waveform phase at 0PN order

    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    c = -(5*np.pi/43008)*(659 + 728*eta)*(((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)/((eta**5)*(M**4)) - (5*np.pi/16)*(s_1*s_2/(eta**3))

    beta = c*(alpha_gb**2)/(M**4)

    dchi_0 = to_SI(beta) / (3/(128*eta))

    return dchi_0

def edgb_dchi_1(
    mass_1: float, 
    mass_2: float, 
    s_1: float, 
    s_2: float, 
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase 0.5PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2

    Returns
    -------
    dchi_1: float
        Deviation coefficient of the waveform phase at 0.5PN order

    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    c = (75*(np.pi**2)/448)*(((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)/((eta**5)*(M**4)) 

    beta = c*(alpha_gb**2)/(M**4)

    dchi_1 = to_SI(beta) / (3/(128*eta))

    return dchi_1

def edgb_dchi_2(
    mass_1: float, 
    mass_2: float, 
    s_1: float, 
    s_2: float, 
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase 1PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2.

    Returns
    -------
    dchi_2: float
        Deviation coefficient of the waveform phase at 1PN order

    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    c = -(5*np.pi/48384)*((((mass_1**2)*s_2 + (mass_2**2)*s_1)**2)*(535+924*eta))/((eta**5)*(M**4)) \
        - (25*np.pi/576)*((((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)/((eta**5)*(M**4)))*(12497995/1016064 \
        + 15407*eta/1440 + 165*(eta**2)/16) + (275*np.pi/1152)*(((mass_1**2)*s_2 \
        - (mass_2**2)*s_1)/((eta**5)*(M**5)))*(mass_1 - mass_2)*((mass_1**2)*s_2 + (mass_2**2)*s_1)
        
    beta = c*(alpha_gb**2)/(M**4)

    dchi_2 = to_SI(beta) / (3/(128*eta))

    return dchi_2

def edgb_dchi_3(
    mass_1: float, 
    mass_2: float, 
    s_1: float, 
    s_2: float, 
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase 1.5PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2.

    Returns
    -------
    dchi_3: float
        Deviation coefficient of the waveform phase at 1.5PN order

    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    # f3GB calculated fom 1.5PN expression in https://arxiv.org/abs/2406.03568.
    f3GB = - ((np.pi**2) / (672*(M**4)*(eta**4))) * ((((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)*(8987 + 8148*eta) \
           + 3584*(M**2)*(eta**2)*((mass_1**2)*(s_2**2) + (mass_2**2)*(s_1**2)) + (mass_1**2)*(mass_2**2)*s_1*s_2*(75264 + 7168*eta))

    c = ((np.pi**2)/2)*((((mass_1**2)*s_2 - (mass_2**2)*s_1)**2)/((eta**5)*(M**4))) - (6*(np.pi**2)*s_1*s_2)/(eta**3) \
        - 3*f3GB/(32*eta)
        
    beta = c*(alpha_gb**2)/(M**4)

    dchi_3 = to_SI(beta) / (3/(128*eta))

    return dchi_3

def edgb_dchi_4(
    mass_1: float, 
    mass_2: float, 
    s_1: float, 
    s_2: float, 
    alpha_gb: float
) -> float:

    """
    EsGB PPE phase 2PN coefficient. This is an absolute deviation, not a fractional one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    s_1: float
        Spin-dependent factor for the primary object
    s_2: float
        Spin-dependent factor for the secondary object    
    alpha_gb: float
        EsGB coupling constant in km^2.

    Returns
    -------
    dchi_4: float
        Deviation coefficient of the waveform phase at 2PN order

    Here we ignore a term ~ (15 / * (64*eta)) * f^{GB}_{4} which depends on contributions from the binding energy at 3PN and a 
    as-of-yet-unknown term in the flux the relative 3PN order. This is set to zero.
        
    """

    # Calculate mass parameters
    M = mass_1 + mass_2
    eta = mass_1*mass_2/(M**2)
     
    # f4GB unknown at the time of writing, so set it to zero
    f4GB = 0
    c = (5*np.pi/32514048)*(1/((eta**5)*(M**5)))*(((mass_1**5)*(s_2**2) + (mass_2**5)*(s_1**2))*(-4341025 + 65553264*eta - 684432*(eta**2)) \
        + eta*(M**2)*((mass_1**3)*(s_2**2) + (mass_2**3)*(s_1**2))*(20044511 + 65553264*eta - 684432*(eta**2)) \
        + 54*(eta**2)*(M**5)*s_1*s_2*(12952549 - 19310256*eta - 366128*(eta**2))) - (15*f4GB)/(64*eta)
        
    beta = c*(alpha_gb**2)/(M**4)

    dchi_4 = to_SI(beta) / (3/(128*eta))

    return dchi_4

# -------------------------- EsGB PPE phase coefficients: TGR Parameterization = 0 -------------------------- #

"""
Calculate the EsGB PPE phase corrections up to 1.5PN using 
coefficients in https://arxiv.org/abs/2406.03568.

These coefficients are added as fractional deviations of the non-spinning sector to the GR phase.
i.e phi_n_NS = phi_n_GR_NS(1 + dchi_n).

This is the original TIGER-like implementation.

"""
    
def edgb_dchi_m2_tiger(
    mass_1 : float,
    mass_2 : float, 
    lGB4 : float, 
    s_A : float, 
    s_B : float
) -> float:

    """
    EsGB PPE phase -1PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_m2: float
        Deviation coefficient of the waveform phase at -1PN order

    """

    # Get mass parameters 
    mass_A     = mass_1 
    mass_B     = mass_2 
    mass_A_2   = mass_A * mass_A
    mass_B_2   = mass_B * mass_B
    mass_A_4   = mass_A_2 * mass_A_2
    mass_B_4   = mass_B_2 * mass_B_2

    dchi_m2    = -5.0 * lGB4 * (mass_A_2 * s_A - mass_B_2 * s_B)**2.0 / (168.0 * mass_A_4 * mass_B_4)
    
    return dchi_m2

def edgb_dchi_0_tiger(
    mass_1 : float, 
    mass_2 : float, 
    lGB4 : float, 
    s_A : float, 
    s_B : float
) -> float:
    
    """
    EsGB PPE phase 0PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_0: float
        Deviation coefficient of the waveform phase at 0PN order

    """

    # Get mass parameters 
    mass_A      = mass_1 
    mass_B      = mass_2 
    mass_A_2    = mass_A * mass_A
    mass_B_2    = mass_B * mass_B
    mass_A_4    = mass_A_2 * mass_A_2
    mass_B_4    = mass_B_2 * mass_B_2

    total_mass  = mass_A + mass_B
    eta         = (mass_A * mass_B) / total_mass**2.0 

    s_A_2       = s_A * s_A
    s_B_2       = s_B * s_B

    numerator   = 659.0 * mass_A_4 * s_B_2 + 1370.0 * mass_A_2 * mass_B_2 * s_A * s_B + 659.0 * mass_B_4 * s_A_2 + 728.0 * eta * (mass_A_2 * s_B - mass_B_2 * s_A)**2.0
    denominator = 16128.0 * mass_A_4 * mass_B_4

    dchi_0      = -5.0 * lGB4 * (numerator / denominator)
    
    return dchi_0

def edgb_dchi_1_tiger(
    mass_1 : float,
    mass_2 : float, 
    lGB4 : float,
    s_A : float,
    s_B : float
) -> float:
   
    """
    EsGB PPE phase 0.5PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_1: float
        Deviation coefficient of the waveform phase at 0.5PN order

    """

    # Get mass parameters 
    mass_A      = mass_1 
    mass_B      = mass_2 
    mass_A_2    = mass_A * mass_A
    mass_B_2    = mass_B * mass_B
    mass_A_4    = mass_A_2 * mass_A_2
    mass_B_4    = mass_B_2 * mass_B_2

    numerator   = (mass_A_2 * s_B - mass_B_2 * s_A)**2.0
    denominator = 56.0 * mass_A_4 * mass_B_4

    dchi_1      = 25.0 * np.pi * lGB4 * (numerator / denominator)
    
    return dchi_1

def edgb_dchi_2_tiger(
    mass_1 : float, 
    mass_2 : float, 
    lGB4 : float, 
    s_A : float, 
    s_B : float
) -> float:
    
    """
    EsGB PPE phase 1PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_2: float
        Deviation coefficient of the waveform phase at 1PN order

    """

    # Get mass parameters 
    mass_A      = mass_1 
    mass_B      = mass_2 
    mass_A_2    = mass_A * mass_A
    mass_B_2    = mass_B * mass_B
    mass_A_4    = mass_A_2 * mass_A_2
    mass_B_4    = mass_B_2 * mass_B_2

    s_A2        = s_A * s_A
    s_B2        = s_B * s_B

    total_mass  = mass_A + mass_B
    eta         = (mass_A * mass_B) / total_mass**2.0
    delta       = (mass_A - mass_B) / total_mass

    numerator   = lGB4 * ( 
          5.0 * mass_A_4 * s_B2 * (-13792267.0 + 5588352.0 * delta - 17640.0*eta * (743.0 + 594.0 * eta)) 
        - 5.0 * mass_B_4 * s_A2 * (+13792267.0 + 5588352.0 * delta + 17640.0*eta * (743.0 + 594.0 * eta)) 
        + 2.0 * mass_A_2 * mass_B_2 * s_A * s_B * (56018615.0 + 3528.0*eta * (12239.0 + 14850.0*eta)) 
    )
    denominator = 290304.0 * mass_A_4 * mass_B_4 * (743.0 + 924.0*eta)

    dchi_2      = (numerator / denominator)
    
    return dchi_2

def edgb_dchi_3_tiger(
    mass_1 : float, 
    mass_2 : float, 
    lGB4 : float, 
    s_A : float, 
    s_B : float
) -> float:
    
    """
    EsGB PPE phase 1.5PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_3: float
        Deviation coefficient of the waveform phase at 1.5PN order

    """

    # Get mass parameters 
    mass_A      = mass_1 
    mass_B      = mass_2 
    mass_A_2    = mass_A * mass_A
    mass_B_2    = mass_B * mass_B
    mass_A_4    = mass_A_2 * mass_A_2
    mass_B_4    = mass_B_2 * mass_B_2

    s_A2        = s_A * s_A
    s_B2        = s_B * s_B

    total_mass  = mass_A + mass_B
    eta         = (mass_A * mass_B) / total_mass**2.0
    delta       = (mass_A - mass_B) / total_mass

    numerator   = lGB4 * (
        + (-14363.0 + 1792.0*delta - 4564.0*eta) * mass_A_4 * s_B2 
        + (-3557.0 + 4564.0*eta) * 2.0 * mass_A_2 * mass_B_2 * s_A * s_B 
        - (+14363.0 + 1792.0*delta + 4564.0*eta) * mass_B_2 * s_A2
    )
    denominator = 43008.0 * mass_A_4 * mass_B_4 

    dchi_3      =(numerator / denominator)
    
    return dchi_3

def edgb_dchi_4_tiger(mass_1 : float, mass_2 : float, lGB4 : float, s_A : float, s_B : float) -> float:

    """
    EsGB PPE phase 2PN coefficient in the TIGER framework. This is an fractional deviation, not a absolute one.

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    lGB4: float
        EsGB coupling constant (^4) in solar masses
    s_A: float
        Spin-dependent factor for primary mass
    s_B: float
        Spin-dependent factor for secondary mass
    
    Returns
    -------
    dchi_4: float
        Deviation coefficient of the waveform phase at 2PN order

    Here we ignore a term ~ (15 / * (64*eta)) * f^{GB}_{4} which depends on contributions from the binding energy at 3PN and a 
    as-of-yet-unknown term in the flux the relative 3PN order. This is set to zero.

    """

    # Get mass parameters 
    mass_A      = mass_1 
    mass_B      = mass_2 
    mass_A_2    = mass_A * mass_A
    mass_B_2    = mass_B * mass_B
    mass_A_3    = mass_A_2 * mass_A
    mass_B_3    = mass_B_2 * mass_B
    mass_A_4    = mass_A_2 * mass_A_2
    mass_B_4    = mass_B_2 * mass_B_2
    mass_A_5    = mass_A_4 * mass_A
    mass_B_5    = mass_B_4 * mass_B

    s_A2        = s_A * s_A
    s_B2        = s_B * s_B

    total_mass  = mass_A + mass_B
    eta         = (mass_A * mass_B) / total_mass**2.0
    eta2        = eta * eta

    numerator   = lGB4 * (
        + (-5 (mass_B_5 * s_A2 + mass_A_5 * s_B2) * (4341025 + 1008 * eta * (-65033 + 679 * eta)))
        + (-270.0 * mass_A_2 * mass_B_2 * total_mass * s_A * s_B * (-12952549.0 + 112.0 * eta * (172413.0 + 3269.0 * eta)))
        + (-5 * mass_A * mass_B * (mass_B_3 * s_A2 + mass_A_3 * s_B2) * (-20044511 + 1008 * eta * (-65033 + 679 * eta)))
    )
    denominator = 12192768 * total_mass * mass_A_4 * mass_B_2 * ((15293365/508032) + (27145/504)*eta + (3085/72)*eta2)
    
    dchi_4      = (numerator / denominator)

    return dchi_4
