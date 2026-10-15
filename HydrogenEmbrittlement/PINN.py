import torch
import torch.nn
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.elastic.tensor import ElasticTensor
import math
import numpy as np
import jarvis.db.figshare import data
import pandas as pd

"""
This package calculates loss functions for mass transport in extreme environements for a metal rod using a PINN
"""

def radial_gradient(func, r)
    """
    Calculates the radial gradient for a function

    Args:
    ----
        func: function with which the gradient is calculated
        r: radial component

    Returns:
    -------
        The gradient of the function
    """
    return np.gradient(func, r, edge_order=2)

def radial_laplacian(func, r):
    """
    Calculates the radial laplacian for a function

    Args:
    ----
        func: function with which the laplacian is calculated
        r: radial component

    Returns:
    -------
        The laplacian of the function
    """
    df_dr=radial_gradient(func, r)
    return np.gradient(df_dr, r, edge_order=2)

def heat_conduction(r, theta, t, k):
    """
    Calculates the heat conduction through radial and angular coordinates

    Args:
    ----
        r: radial coordinate (m)
        theta: angular coordinate (
        t: time (s)
        k: thermal conductivity W/(m*K)

    Returns:
    -------

    """
    #Heat conduction for a 2D radial system
    d2T_dr2=second_order_derivatives(model, r, T)
    dT_dr=first_order_derivatives(model, r, T)
    d2T_dtheta2=second_order_derivatives(model, theta, T)
    return -k*(d2T_dr2+(1/r)*(dT_dr)+(1/r**2)*(d2T_dtheta2)

def first_order_derivatives(model, x, y):
    #Use neural net to find the first order derivatives 
    dy_dx=torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return dy_dx

def second_order_derivatives(model, x, y):
    #Use neural net to find the second order derivatives
    dy_dx=first_order_derivatives(model, x, y)
    d2y_dx2=torch.autograd.grad(outputs=dy_dx, inputs=x.ones_like(y), create_graph=True)[0]
    return d2y_dx2

def get_density(chemical_species):
    #Determine the density of chemical species using jarvis-DFT
    dft_data=data('dft_3d')
    df=pd.DataFrame(dft_data)
    return df[df['formula']==chemical_species].sort_values('formation_energy_peratom').iloc[0]['density']

class LossFunction(nn.Module):
    def __init__(self):
        #Constants as used from Jarvis-DFT (stress tensor constants) for pure Zr (HCP) as an initial approximation
        data=get_jid_data(jid="JVASP-10143", dataset="dft_3d"
        hcp_zr_stress_tensors=data.get("elastic_tensor")
        tensor_array=ElasticTensor(hcp_zr_stress)
        # Values for stiffness tensor elements
        self.C11=tensor_array.et_tensor[0][0]
        self.C12=tensor_array.et_tensor[0][1]
        self.C13=tensor_array.et_tensor[0][2]
        self.C33=tensor_array.et_tensor[2][2]
        self.C44=tensor_array.et_tensor[3][3]
        
        #thermal conductivity
        self.k=22.6 #W/(m.K)
        
        #Area (where r=0.01m) 
        self.A=math.pi*(0.01**2)
        self.T=1000+273.15 #(in Kelvin) the temperature at the boundary (at r=0.01m)
        self.R=8.3145 #J/(K*mol)
        self.rho=get_density('Zr')
    def mechanical_loss(r, t):                     
        #Cauchy-Navier equation for solid mechanics propagation
        d2u_dt2=second_order_derivatives(self.model, t, u)   
        rho*d2u_dt2=(lambda+mu)radial_gradient(radial_gradient(u, r), r)+mu*radial_laplacian(u)+force
    def thermal_loss(r, theta, t):
        #Fourier's law for a cylindrical vessel
        dT_dt=first_order_derivatives(self.model, t, T)
        f_thermal=heat_conduction(r, theta, t, self.k) -self.q_dot+rho*C_p*dT_dt
    def transport_loss(r, t):
        #Flux equation
        d_C_L_dt=first_order_derivatives(model, t, C_L)
        d_C_T_dt=first_order_derivatives(model, t, C_T)
        J=-D*radial_gradient(C_L, r)+(D*C_L*V_H)/(R*T)*radial_gradient(sigma_h)-(D*C_L*Q)/(R*T**2)*radial_gradient(T)
        f_transport=d_C_L_dt+d_C_T_dt+radial_gradient(J, r)
    def total_loss():
        #f_total=f_mechanical+f_thermal+f_transport
        f_total=self.mechanical_loss(r, t)+self.thermal_loss(r, theta, t)+self.transport_loss(r, t)
