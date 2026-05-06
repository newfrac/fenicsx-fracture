import numpy as np
from math import*
import matplotlib.pyplot as plt
import sys
sys.path.append("./pycodes_coupledcriterion")
from plot_results import plot_cc_europe, plot_cc_france

def coupled_criterion(S,Ginc,L,Sc,Gc):
    
    P_ginc = np.sqrt(np.divide(Gc,Ginc))
    P_strs = np.divide(Sc,S)
    
    Pmin_index_ginc = np.argwhere( np.isclose( P_ginc,np.amin(P_ginc) ) )
    Pmin_index_strs = np.argwhere( np.isclose( P_strs,np.amin(P_strs) ) )
    
    if Pmin_index_ginc >= Pmin_index_strs:
        Pmin = P_ginc[Pmin_index_ginc]
    else:
        Pmin = P_strs[Pmin_index_strs]
    
    plot_cc_europe(L, P_ginc/Pmin[0], P_strs/Pmin[0])
    plot_cc_france(L, Ginc/Gc*Pmin[0]**2, S/Sc*Pmin[0])
    
    return Pmin
