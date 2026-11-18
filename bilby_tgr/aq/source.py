import numpy as np
#import lal
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.integrate import quad
from scipy.integrate import odeint
import lal
from bilby.gw.source import _base_lal_cbc_fd_waveform

import bilby
import bilby_tgr
from scipy.stats import norm
from pycbc.types import FrequencySeries
from pycbc.waveform import get_fd_waveform
from pycbc.filter.matchedfilter import match


def z1(spin):
    return (1 + ((1 - spin**2)**(1./3)) * ((1. + spin)**(1/3) + (1. - spin)**(1/3)))

def z2(spin):
    z2 = np.sqrt(3*spin**2 + z1(spin)**2)
    return z2
    
def riscocap(spin):
    return (3. + z2(spin) - 
            (spin/np.abs(spin)) * (np.sqrt((3. - z1(spin)) * (3. + z1(spin) + 
                                                                2. * z2(spin)))))


def Eiscocap(spin):
    return np.sqrt(1. - (2./(3. * riscocap(spin))))

def liscocap(spin):
    lisco = (2/(3 * np.sqrt(3))) * (1. + 2 * np.sqrt(3 * riscocap(spin) - 2.))
    return lisco

def omegacap(spin):
    return 1/(riscocap(spin)**(3./2) + spin)
    
def chifinal(eta, spin, atot):
    k00 = -3.821158961;
    k01 = -1.2019; 
    k02 = -1.20764; 
    k10 = 3.79245; 
    k11 = 1.18385; 
    k12 = 4.90494; 
    chif = atot + eta * (liscocap(spin) - (2 * atot * (Eiscocap(spin) - 1))) + k00 * eta**2. + k01 * eta**2. * spin + k02 * eta**2 * spin**2 + k10 * eta**3 + k11 * eta**3 * spin + k12 * eta**3 * spin**2
    return chif
    
def KerrISCO(mass_1, mass_2, a_1, a_2):
    msun = lal.MTSUN_SI
    m_1 = mass_1 * msun
    m_2 = mass_2 * msun
    eta = (m_1 * m_2)/(m_1 + m_2)**2.
    M = m_1 + m_2 
    z = 0
    k01 = -1.2019; 
    k02 = -1.20764; 
    k10 = 3.79245; 
    k11 = 1.18385; 
    k12 = 4.90494; 
    Zeta = 0.41616; 
    k00 = -3.821158961;
    
    atot = (a_1 + a_2 * (m_2/m_1)**2.)/(1. + m_2/m_1)**2.
    aeff = atot + Zeta * eta * (a_1 + a_2)
    #print("atot",atot)
    #print("aeff",aeff)
    spinFinal = chifinal(eta, aeff, atot)
    OmegaCap = omegacap(spinFinal)
    #print(spinFinal,"spinFinal")
    #print(OmegaCap,"OmegaCap")

    
    Scap = (1./(1. - 2 * eta)) * ((a_1 * m_1**2. + a_2 *  m_2**2.)/(M)**2.)
    #print(Scap)
    SS = (1 + Scap * (-0.00303023 - 2.00661 * eta + 7.70506 * eta**2))/(1 + Scap * (-0.67144 - 1.47569 * eta + 7.30468 *eta**2))
    #print(SS,"SS")
    Erad = M * (0.0559745 * eta + 0.580951 * eta**2 - 0.960673 * eta**3 + 3.35241 * eta**4) * SS
    #print(Erad,"Erad")
    Mfin = M * (1. - (Erad/M))
    #print(Mfin,"Mfin")
    freq = (1./(1. + z)) * (OmegaCap/(np.pi * Mfin))
    # returning the orbital frequency, use it as it is, no multiplication factor
    return freq

#GW frequency 

def v(freqs, mass_1, mass_2):
    return (np.pi * (mass_1 + mass_2)* lal.MTSUN_SI * freqs)**(1./3.)


def mass_ratio(mass_1, mass_2):
    mr = mass_1/mass_2 
    return mr

def sym_mass_ratio(mass_1, mass_2):
    eta = (mass_1*mass_2)/(mass_1+mass_2)**2.
    return eta

# functions go into the H(f) defenition

def kappa(mass, spin):
    return np.sqrt(1 - spin**2)/(2*mass*lal.MTSUN_SI*(1 + np.sqrt(1 - spin**2)))

def omegaH(mass, spin):
    return (np.abs(spin))/(2*mass*lal.MTSUN_SI*(1+np.sqrt(1- spin**2)))

def gamma(mass, spin):
    return (1./(2. * np.pi)) * 1.005*(1/(mass*lal.MTSUN_SI))*(np.exp(-6.42 + 1.8*spin**2 + 1.9*spin**(12) - 0.1*spin**(14)))

def nMaxValueBH1(mass_1, mass_2, a_1, a_2, alpha):
    omega_0=2*omegaH(mass_1, a_1)
    nMax = math.floor((8*np.pi/(kappa(mass_1, a_1)*alpha))*((2*np.pi*KerrISCO(mass_1, mass_2, a_1, a_2))-omega_0))
    if nMax >= 0:
        return nMax
    else:
        return -1

def nMaxValueBH2(mass_1, mass_2, a_1,  a_2, alpha):
    omega_0=2*omegaH(mass_2, a_2)
    nMax = math.floor((8*np.pi/(kappa(mass_2, a_2)*alpha))*((2*np.pi*KerrISCO(mass_1, mass_2, a_1, a_2))-omega_0))
    if nMax >= 0:
        return nMax
    else:
        return -1

def omega_n(n, mass, spin, alpha):
    omg = n * ((kappa(mass, spin)*alpha)/(8*np.pi)) + 2*omegaH(mass, spin)
    return omg

def H1funTrialVectv4(freqs, mass_1, mass_2, a_1, a_2, alpha):
    nmax = nMaxValueBH1(mass_1, mass_2, a_1, a_2, alpha)
    fn_1 = omega_n(np.arange(0, nmax+2), mass_1, a_1, alpha) / (2 * np.pi)
    gamma_val = gamma(mass_1, a_1)
    
    mask = np.logical_and((freqs[:, np.newaxis] - fn_1) > -gamma_val / 2, (freqs[:, np.newaxis] - fn_1) < gamma_val / 2)
    phase_corr = np.where(mask, 0.5 + 0.5 * np.cos(2 * np.pi * (freqs[:, np.newaxis] - fn_1) / gamma_val), 0)
    out = np.where(np.sum(phase_corr, axis=1) <= 1, np.sum(phase_corr, axis=1), 1)
    
    out[nmax < 0] = 0  # Set output to zero where nmax <= 0
    
    return out
    #return np.round(out,1) 

def H2funTrialVectv4(freqs, mass_1, mass_2, a_1, a_2, alpha):
    nmax = nMaxValueBH2(mass_1, mass_2, a_1, a_2, alpha)
    fn_1 = omega_n(np.arange(0, nmax+2), mass_2, a_2, alpha) / (2 * np.pi)
    gamma_val = gamma(mass_2, a_2)
    
    mask = np.logical_and((freqs[:, np.newaxis] - fn_1) > -gamma_val / 2, (freqs[:, np.newaxis] - fn_1) < gamma_val / 2)
    phase_corr = np.where(mask, 0.5 + 0.5 * np.cos(2 * np.pi * (freqs[:, np.newaxis] - fn_1) / gamma_val), 0)
    out = np.where(np.sum(phase_corr, axis=1) <= 1, np.sum(phase_corr, axis=1), 1)
    
    out[nmax < 0] = 0  # Set output to zero where nmax <= 0
    return out
    #return np.round(out,1) 


def H1fun(freqs, mass_1, mass_2, a_1, a_2, alpha):
    nmax = nMaxValueBH1(mass_1, mass_2, a_1, a_2, alpha)
    fn_1 = omega_n(np.arange(0, nmax+2), mass_1, a_1, alpha) / (2 * np.pi)
    gamma_val = gamma(mass_1, a_1)
    freqs = np.atleast_1d(freqs)
    fn_1 = np.atleast_1d(fn_1)
    mask = np.logical_and((freqs[:, np.newaxis] - fn_1) > -gamma_val / 2, 
                          (freqs[:, np.newaxis] - fn_1) < gamma_val / 2)
    phase_corr = np.where(mask, 
                          np.exp(-4 * np.pi**2 * (freqs[:, np.newaxis] - fn_1)**2 / (2*(gamma_val/2)**2)), 
                          np.exp(-4 * np.pi**2 * (freqs[:, np.newaxis] - fn_1)**2 / (2*(gamma_val/2)**2)))
    out = np.where(np.sum(phase_corr, axis=1) <= 1, 
                   np.sum(phase_corr, axis=1), 
                   1)
    if nmax < 0:
        out = np.zeros_like(freqs)
    return out

def H2fun(freqs, mass_1, mass_2, a_1, a_2, alpha):
    nmax = nMaxValueBH2(mass_1, mass_2, a_1, a_2, alpha)
    fn_1 = omega_n(np.arange(0, nmax+2), mass_2, a_2, alpha) / (2 * np.pi)
    gamma_val = gamma(mass_2, a_2)
    freqs = np.atleast_1d(freqs)
    fn_1 = np.atleast_1d(fn_1)
    mask = np.logical_and((freqs[:, np.newaxis] - fn_1) > -gamma_val / 2, 
                          (freqs[:, np.newaxis] - fn_1) < gamma_val / 2)
    phase_corr = np.where(mask, 
                          np.exp(-4 * np.pi**2 * (freqs[:, np.newaxis] - fn_1)**2 / (2*(gamma_val/2)**2)), 
                          np.exp(-4 * np.pi**2 * (freqs[:, np.newaxis] - fn_1)**2 / (2*(gamma_val/2)**2)))
    out = np.where(np.sum(phase_corr, axis=1) <= 1, 
                   np.sum(phase_corr, axis=1), 
                   1)
    if nmax < 0:
        out = np.zeros_like(freqs)
    return out



#Eqn 13(a) Sayak et al.

def A2p5PNPrimaryBH(mass_1, mass_2, a_1):
    return (mass_1/(mass_1 + mass_2))**3 *a_1* (3*a_1**2 + 1)

def A2p5PNSecondayBH(mass_1, mass_2, a_2):
    return (mass_2/(mass_1 + mass_2))**3 *a_2* (3*a_2**2 + 1)

#Eqn 13(b) Sayak et al.

#Eqn 13(b) Sayak et al.
def A4PNPrimaryBH(mass_1, mass_2, a_1):
    term = (4. * np.pi * A2p5PNPrimaryBH(mass_1, mass_2, a_1)) + (mass_1/(mass_1 + mass_2))**4 * (3. * a_1**2 + 1) * (np.sqrt(1. - a_1**2) + 1.)
    return term

def A4PNSecondayBH(mass_1, mass_2, a_2):
    term = (4. * np.pi * A2p5PNSecondayBH(mass_1, mass_2, a_2)) + (mass_2/(mass_1 + mass_2))**4 * (3. * a_2**2 + 1) * (np.sqrt(1. - a_2**2) + 1.)
    return term


def psiSO(mass_1, mass_2, a_1, a_2):
    mr = mass_ratio(mass_1, mass_2) # Samanvaya 2202.08661 for psiSO
    psiSO = ((73. / (3. * (1. + mr)**2.)) * (mr**2. * a_1 + a_2)) + ((((15. * mr) / (1. + mr)**2.)) * (a_1 + a_2))
    return  psiSO


# For calculating QBH phase', Primary: (p1 = 1, p2 = 0), Seconadry: (p1 = 0, p2 = 1)

def psi5CBH(mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psi5CBH_1 + p2 * psi5CBH_2
    return (A2p5PNPrimaryBH(mass_1, mass_2, a_1) * p1) + (A2p5PNSecondayBH(mass_1, mass_2, a_2)  * p2)

def psi8CBH(mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psi8CBH_1 + p2 * psi8CBH_2
    return (A4PNPrimaryBH(mass_1, mass_2, a_1) * p1) + (A4PNSecondayBH(mass_1, mass_2, a_2) * p2)

def psi2p5PNCBH(freqs, mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psi2p5PNCBH_1 + p2 * psi2p5PNCBH_2
    return (-10./9.) * (psi5CBH(mass_1, mass_2, a_1, a_2, p1, p2) * (3. * np.log(v(freqs, mass_1, mass_2)) + 1))

def psi3p5PNCBH(mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psi3p5PNCBH_1 + p2 * psi3p5PNCBH_2
    eta = sym_mass_ratio(mass_1, mass_2)
    return (-5./168.) * (psi5CBH(mass_1, mass_2, a_1, a_2, p1, p2) * (952. * eta + 995.))

def psi4PNCBH(freqs, mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psi4PNCBH_1 + p2 * psi4PNCBH_2
    return -(5./9.) * (3. * np.log(v(freqs, mass_1, mass_2)) - 1) * ((psi5CBH(mass_1, mass_2, a_1, a_2, p1, p2) * psiSO(mass_1, mass_2, a_1, a_2))- (4. * psi8CBH( mass_1, mass_2, a_1, a_2, p1, p2)))

def delta_phase_CBH(freqs, mass_1, mass_2, a_1, a_2, p1, p2):
    # p1 * psiCBH_1 + p2 * psiCBH_2 (total: p1 = p2 = 1)

    psiCBH2p5PN = psi2p5PNCBH(freqs, mass_1, mass_2, a_1, a_2, p1, p2)
    psiCBH3p5PN = psi3p5PNCBH(mass_1, mass_2, a_1, a_2, p1, p2)
    psiCBH4PN = psi4PNCBH(freqs, mass_1, mass_2, a_1, a_2, p1, p2)
    vel = v(freqs, mass_1, mass_2)
    eta = sym_mass_ratio(mass_1, mass_2)
    phaseCBH = 3./(128.* eta * vel**5.) * (psiCBH2p5PN * vel**5. + psiCBH3p5PN * vel**7. + psiCBH4PN * vel**8.)
    return phaseCBH


def first_derivativeCBH(freq, mass_1, mass_2, a_1, a_2, p1, p2): # p1 * psiCBH_1' + p2 * psiCBH_2'
    h = 1e-3  # Small step size for differentiation
    fstderiv = (delta_phase_CBH(freq + h, mass_1, mass_2, a_1, a_2, p1, p2) - delta_phase_CBH(freq, mass_1, mass_2, a_1, a_2, p1, p2)) / (h)
    return fstderiv

def diff_delta_phase_QBH(freq, mass_1, mass_2, a_1, a_2, alpha): # psiQBH' = H1 * psiCBH_1' + H2 * psiCBH_2'
    diff_psiCBH1 = first_derivativeCBH(freq, mass_1, mass_2, a_1, a_2, 1, 0)
    diff_psiCBH2 = first_derivativeCBH(freq, mass_1, mass_2, a_1, a_2, 0, 1)
    diff_phase = (H1fun(freq, mass_1, mass_2, a_1, a_2, alpha)*diff_psiCBH1) + (H2fun(freq, mass_1, mass_2, a_1, a_2, alpha)*diff_psiCBH2)
    return diff_phase


def diff2_delta_phase_QBH(freq, mass_1, mass_2, a_1, a_2, alpha): # (psiQBH')'
    h = 1e-3  # Small step size for differentiation
    secderiv = (diff_delta_phase_QBH(freq + h, mass_1, mass_2, a_1, a_2, alpha) - diff_delta_phase_QBH(freq, mass_1, mass_2, a_1, a_2, alpha)) / (h)
    return secderiv

def second_derivativeCBH(freq, mass_1, mass_2, a_1, a_2): # psiCBH"
    h = 1e-3  # Small step size for differentiation
    deriv = (delta_phase_CBH(freq + h, mass_1, mass_2, a_1, a_2, 1, 1) - 2 * delta_phase_CBH(freq, mass_1, mass_2, a_1, a_2, 1, 1) + delta_phase_CBH(freq - h, mass_1, mass_2, a_1, a_2, 1, 1)) / (h**2)
    return deriv






def testint(freqs, mass_1, mass_2, a_1, a_2, alpha): # integrating psiQBH' using the explicit Euler method
    z = np.zeros(len(freqs))
    psi20 = delta_phase_CBH(20, mass_1, mass_2, a_1, a_2, 1, 1) # Intial condition: psiQBH[f=20] = 0
    z[0]=psi20
    for n in range(0,len(freqs)-1):
        z[n+1] = z[n] + diff_delta_phase_QBH(freqs[n], mass_1, mass_2, a_1, a_2, alpha)*(freqs[n+1] - freqs[n])  
    return z


def lal_CBHModel(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    **kwargs,
):
    waveform_kwargs = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=20.0,
        minimum_frequency=20.0,
        maximum_frequency=1024.0,
        catch_waveform_errors=True,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)
    waveform_to_return = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        theta_jn=theta_jn,
        phase=phase,
        a_1=a_1,
        a_2=a_2,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        phi_jl=phi_jl,
        **waveform_kwargs,
    )
    phase_correction = delta_phase_CBH(frequency_array, mass_1, mass_2, a_1, a_2, 1, 1)

    for key in ["plus", "cross"]:
        waveform_to_return[key] = (
            np.exp(-1j * phase_correction) * waveform_to_return[key]
        )
    return waveform_to_return

def lal_QBHModel(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    qbh_alpha,
    **kwargs,
):
    waveform_kwargs = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=20.0,
        minimum_frequency=20.0,
        maximum_frequency=1024.0,
        catch_waveform_errors=True,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)
    waveform_to_return = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        theta_jn=theta_jn,
        phase=phase,
        a_1=a_1,
        a_2=a_2,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        phi_jl=phi_jl,
        **waveform_kwargs,
    )
    phase_correction = testint(frequency_array, mass_1, mass_2, a_1, a_2, qbh_alpha)
    #phase_correction = diff2_delta_phase_QBH(frequency_array, mass_1, mass_2, a_1, a_2, qbh_alpha)

    for key in ["plus", "cross"]:
        waveform_to_return[key] = (
            np.exp(-1j * phase_correction) * waveform_to_return[key]
        )
    return waveform_to_return





