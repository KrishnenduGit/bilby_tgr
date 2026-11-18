import lalsimulation
from bilby.gw.source import _base_lal_cbc_fd_waveform, _base_waveform_frequency_sequence
from lal import CreateDict
import numpy as np

from .utils import *

# All PN orders at which an EsGB phase correction is available, in ascending order.
EDGB_PN_ORDERS = [-1, 0, 0.5, 1, 1.5, 2]


def _normalize_edgb_pn_order(edgb_pn_order):
    """Normalize the ``edgb_pn_order`` option to a list of PN orders to activate.

    Two input forms are supported:

    1. A list (or tuple) specifies exactly which PN orders to switch on, e.g.
       ``[-1, 0, 2]`` activates only the -1, 0 and 2 PN corrections.
    2. A scalar (int/float) is interpreted as the *maximum* PN order, activating
       every available order up to and including it, e.g. ``1`` activates the
       -1, 0, 0.5 and 1 PN corrections.

    Values are coerced to ``float`` because bilby_pipe passes the contents of
    ``waveform-arguments-dict`` through as strings (e.g. ``[-1]`` arrives as
    ``['-1']`` and ``2`` as ``'2'``). Without this coercion the integer/float
    membership tests below (``if -1 in edgb_pn_order``) would compare against
    strings and silently never match, applying no EsGB correction at all.
    """
    if np.isscalar(edgb_pn_order):
        max_order = float(edgb_pn_order)
        return [order for order in EDGB_PN_ORDERS if order <= max_order]
    return [float(order) for order in edgb_pn_order]


# ----------- Frequency Domain Source Functions ----------- #

def lal_binary_black_hole_esgb(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A binary black hole waveform using lalsimulation with EsGB corrections
    added to the waveform phase.

    Corrections are added as absolute deviations, i.e phi_n = phi_n_GR + dchi_n.

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square Root of Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    # Since we are sampling in alpha_gb_sqrt, we need to square it here
    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Calculate sensitivities
    s_1 = calculate_sensitivity_EsGB(a_1)
    s_2 = calculate_sensitivity_EsGB(a_2)

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0 = 0.0
    dchi_1 = 0.0
    dchi_2 = 0.0
    dchi_3 = 0.0
    dchi_4 = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4(mass_1, mass_2, s_1, s_2, alpha_gb)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use tgr waveform phase parameterisation: Psi = Psi_GR + Psi_NonGR
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 2)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)

def lal_neutron_star_black_hole_esgb(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A neutron star black hole waveform using lalsimulation with EsGB corrections
    added to the waveform phase.

    Corrections are added as absolute deviations, i.e phi_n = phi_n_GR + dchi_n.

    Here we use a BBH baseline waveform, with EsGB corrections as for an NSBH.

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square root of the Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Calculate sensitivities. For a NS, s = 0.
    s_1 = calculate_sensitivity_EsGB(a_1)
    s_2 = 0

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0 = 0.0
    dchi_1 = 0.0
    dchi_2 = 0.0
    dchi_3 = 0.0
    dchi_4 = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4(mass_1, mass_2, s_1, s_2, alpha_gb)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use tgr waveform phase parameterisation: Psi = Psi_GR + Psi_NonGR
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 2)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)


def lal_binary_black_hole_esgb_tiger(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A binary black hole waveform using lalsimulation with an
    EsGB correction in the TIGER framework.

    The corrections here are applied as a shift to the non-spinning PN coefficients. 

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square root of the Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0, edgb_pn_order=2
    )
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict      = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Rescale alpha to lGB  
    lGB4         = alpha_gb * alpha_gb * 16.0 * np.pi
    sA           = calculate_sensitivity_EsGB(a_1)
    sB           = calculate_sensitivity_EsGB(a_2)

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0       = 0.0
    dchi_1       = 0.0
    dchi_2       = 0.0
    dchi_3       = 0.0
    dchi_4       = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4_tiger(mass_1, mass_2, lGB4, sA, sB)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use TIGER parameterisation: psi = (psi_GR_NS + d_psi) + psi_GR_S. This is the default, but just in case...
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 0)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs
        )

def lal_neutron_star_black_hole_esgb_tiger(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A neutron star black hole waveform using lalsimulation with an
    EsGB correction in the TIGER framework.

    The corrections here are applied as a shift to the non-spinning PN coefficients. 

    Here we use a BBH baseline waveform, with EsGB corrections as for an NSBH.

    Parameters
    ----------
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square root of the Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0, edgb_pn_order=2
    )
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict      = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Rescale alpha to lGB
    lGB4         = alpha_gb * alpha_gb * 16.0 * np.pi
    sA           = calculate_sensitivity_EsGB(a_1)
    sB           = 0

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0       = 0.0
    dchi_1       = 0.0
    dchi_2       = 0.0
    dchi_3       = 0.0
    dchi_4       = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4_tiger(mass_1, mass_2, lGB4, sA, sB)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use TIGER parameterisation: psi = (psi_GR_NS + d_psi) + psi_GR_S. This is the default, but just in case...
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 0)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs
        )

# ----------- Frequency Domain Source Functions - Multibanding ----------- #

def lal_binary_black_hole_esgb_frequency_sequence(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A binary black hole waveform using lalsimulation with EsGB corrections
    added to the waveform phase.

    Corrections are added as absolute deviations, i.e phi_n = phi_n_GR + dchi_n.

    This generates a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency samples.
    For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ----------
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square Root of Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Calculate sensitivities
    s_1 = calculate_sensitivity_EsGB(a_1)
    s_2 = calculate_sensitivity_EsGB(a_2)

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0 = 0.0
    dchi_1 = 0.0
    dchi_2 = 0.0
    dchi_3 = 0.0
    dchi_4 = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4(mass_1, mass_2, s_1, s_2, alpha_gb)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use tgr waveform phase parameterisation: Psi = Psi_GR + Psi_NonGR
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 2)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)

def lal_neutron_star_black_hole_esgb_frequency_sequence(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A neutron star black hole waveform using lalsimulation with EsGB corrections
    added to the waveform phase.

    Corrections are added as absolute deviations, i.e phi_n = phi_n_GR + dchi_n.

    Here we use a BBH baseline waveform, with EsGB corrections as for an NSBH.
    
    This generates a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency samples.
    For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ----------
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square Root of Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Required keyword arguments
        - frequencies:
            ndarray of frequencies at which waveforms are evaluated

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Calculate sensitivities. For a NS, s = 0.
    s_1 = calculate_sensitivity_EsGB(a_1)
    s_2 = 0

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_m2 = 0.0
    dchi_0 = 0.0
    dchi_1 = 0.0
    dchi_2 = 0.0
    dchi_3 = 0.0
    dchi_4 = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_m2 = edgb_dchi_m2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3(mass_1, mass_2, s_1, s_2, alpha_gb)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4(mass_1, mass_2, s_1, s_2, alpha_gb)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_m2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use tgr waveform phase parameterisation: Psi = Psi_GR + Psi_NonGR
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 2)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)

def lal_binary_black_hole_esgb_tiger_frequency_sequence(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A binary black hole waveform using lalsimulation with an
    EsGB correction in the TIGER framework.

    This generates a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency samples.
    For example, this is used for MBGravitationalWaveTransient.

    The corrections here are applied as a shift to the non-spinning PN coefficients. 

    Parameters
    ----------
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square root of the Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict      = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Rescale alpha to lGB  
    lGB4         = alpha_gb * alpha_gb * 16.0 * np.pi
    sA           = calculate_sensitivity_EsGB(a_1)
    sB           = calculate_sensitivity_EsGB(a_2)

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_Minus_2 = 0.0
    dchi_0       = 0.0
    dchi_1       = 0.0
    dchi_2       = 0.0
    dchi_3       = 0.0
    dchi_4       = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_Minus_2 = edgb_dchi_m2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4_tiger(mass_1, mass_2, lGB4, sA, sB)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_Minus_2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use TIGER parameterisation: psi = (psi_GR_NS + d_psi) + psi_GR_S. This is the default, but just in case...
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 0)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)

def lal_neutron_star_black_hole_esgb_tiger_frequency_sequence(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase, 
    alpha_gb_sqrt,
    **kwargs
):
    """
    A neutron star black hole waveform using lalsimulation with an
    EsGB correction in the TIGER framework.

    The corrections here are applied as a shift to the non-spinning PN coefficients. 

    Here we use a BBH baseline waveform, with EsGB corrections as for an NSBH.

    This generates a waveform only on specified frequency points. This is useful for
    likelihood requiring waveform values at a subset of all the frequency samples.
    For example, this is used for MBGravitationalWaveTransient.

    Parameters
    ----------
    frequency_array: array_like
        The input is ignored.
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    alpha_gb_sqrt: float
        Square root of the Einstein-scalar-Gauss-Bonnet coupling constant in km
    kwargs: dict
        Optional keyword arguments

    Returns
    -------
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    alpha_gb = alpha_gb_sqrt**2

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXPHM', reference_frequency=50.0,
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    # If no waveform dictionary is passed, create one
    wf_dict      = waveform_kwargs.get("lal_waveform_dictionary", CreateDict())

    # Rescale alpha to lGB  
    lGB4         = alpha_gb * alpha_gb * 16.0 * np.pi
    sA           = calculate_sensitivity_EsGB(a_1)
    sB           = 0

    # Get the PN order of the EdGB corrections
    edgb_pn_order = _normalize_edgb_pn_order(waveform_kwargs.pop("edgb_pn_order", 2))

    # Initialize PN deviation coefficients
    dchi_Minus_2 = 0.0
    dchi_0       = 0.0
    dchi_1       = 0.0
    dchi_2       = 0.0
    dchi_3       = 0.0
    dchi_4       = 0.0
 
    # Compute each coefficient only if its PN order is in the requested list
    if -1 in edgb_pn_order:
        dchi_Minus_2 = edgb_dchi_m2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0 in edgb_pn_order:
        dchi_0       = edgb_dchi_0_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 0.5 in edgb_pn_order:
        dchi_1       = edgb_dchi_1_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1 in edgb_pn_order:
        dchi_2       = edgb_dchi_2_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 1.5 in edgb_pn_order:
        dchi_3       = edgb_dchi_3_tiger(mass_1, mass_2, lGB4, sA, sB)
    if 2 in edgb_pn_order:
        dchi_4       = edgb_dchi_4_tiger(mass_1, mass_2, lGB4, sA, sB)

    # Add corrections to waveform
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChiMinus2(wf_dict, dchi_Minus_2)  
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi0(wf_dict, dchi_0)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi1(wf_dict, dchi_1)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi2(wf_dict, dchi_2)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi3(wf_dict, dchi_3)
    lalsimulation.SimInspiralWaveformParamsInsertNonGRDChi4(wf_dict, dchi_4)

    # Use TIGER parameterisation: psi = (psi_GR_NS + d_psi) + psi_GR_S. This is the default, but just in case...
    lalsimulation.SimInspiralWaveformParamsInsertNonGRParameterization(wf_dict, 0)

    # Update waveform dictionary with TIGER coefficients
    waveform_kwargs["lal_waveform_dictionary"] = wf_dict

    return _base_waveform_frequency_sequence(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)
