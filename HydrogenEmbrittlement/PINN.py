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

def q_flux(r, theta, t, k, q):
    #Flux for a 2D radial system
    return -k*(d^2T/dr^2+(1/r)*(dT/dr)+(1/r^2)*(d^2T/dtheta^2)-q

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
        self.T=1000 #(in Kelvin) the temperature at the boundary (at r=1m)
        self.R=8.3145 #J/(K*mol)
    def mechanical_loss(r, t):                     
        #Cauchy-Navier equation
        rho*d^2(u)/dt^2=(lambda+mu)radial_gradient(radial_gradient(u, r), r)+mu*radial_laplacian(u)+force
    def thermal_loss(r, theta, t):
        #Fourier's law (-kAdT/dr)
        f_thermal=q_flux(r, theta, t, self.k, self.q)+rho*C_p*dT/dt
    def transport_loss(r, t):
        #Flux equation
        J=-D*radial_gradient(C_L, r)+(D*C_L*V_H)/(R*T)*radial_gradient(sigma_h)-(D*C_L*Q)/(R*T^2)*radial_gradient(T)
        f_transport=d(C_L)/dt+d(C_T)/dt+radial_gradient(J, r)
    def total_loss():
        #f_total=f_mechanical+f_thermal+f_transport
        f_total=self.f_mechanical()+self.f_thermal()+self.f_transport()
