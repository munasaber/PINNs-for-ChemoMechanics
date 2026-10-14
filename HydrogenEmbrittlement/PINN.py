import torch
import torch.nn
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.elastic.tensor import ElasticTensor
import math
import numpy as np

"""
This package calculates loss functions for mass transport in extreme environements for a metal rod using a PINN
"""

def radial_gradient(f, r)
    return np.gradient(f, r, edge_order=2)

def radial_laplacian(f, r):
    df_dr=radial_gradient(f, r)
    return np.gradient(df_dr, r, edge_order=2)

class LossFunction(nn.Module):
    def __init__(self):
        #Constants as used from Jarvis-DFT (stress tensor constants) for pure Zr as an initial approximation
        data=get_jid_data(jid="JVASP-10143", dataset="dft_3d"
        hcp_zr_stress_tensors=data.get("elastic_tensor")
        tensor_array=ElasticTensor(hcp_zr_stress)
        self.C11=tensor_array.et_tensor[0][0]
        self.C12=tensor_array.et_tensor[0][1]
        self.C13=tensor_array.et_tensor[0][2]
        self.C33=tensor_array.et_tensor[2][2]
        self.C44=tensor_array.et_tensor[3][3]
        #thermal conductivity
        self.k=22.6 #W/(m.K)
        #Area (where r=1m) 
        self.A=math.pi
    def mechanical_loss:                     
        #Cauchy-Navier equation
        rho*d^2(u)/dt^2=(lambda+mu)radial_gradient(radial_gradient(u, r), r)+mu*radial_laplacian(u)+force
    def thermal_loss:
        #Fourier's law (-kAdT/dr)
        f_thermal=-k*A*dT/dr
    def transport_loss:
        #Flux equation
