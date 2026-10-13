import torch
import torch.nn
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.elastic.tensor import ElasticTensor

"""
This package calculates loss functions for mass transport in extreme environements for a metal rod using a PINN
"""


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
    def mechanical_loss:                     
        #Cauchy-Navier equation
    def thermal_loss:
        #Fourier's law (-kAdT/dr)
    def transport_loss:
        #Flux equation
