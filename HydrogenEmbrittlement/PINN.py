import torch
import torch.nn
from jarvis.db.figshare import get_jid_data
from jarvis.analysis.elastic.tensor import ElasticTensor

#
class LossFunction(nn.Module):
    #Constants as used from Jarvis-DFT (stress tensor constants) for pure Zr as an initial approximation
    data=get_jid_data(jid="JVASP-10143", dataset="dft_3d"
    hcp_zr_stress_tensors=data.get("elastic_tensor")
    tensor_array=ElasticTensor(hcp_zr_stress)
    C11=tensor_array.et_tensor[0][0]
    C12=tensor_array.et_tensor[0][1]
    C13=tensor_array.et_tensor[0][2]
    C33=tensor_array.et_tensor[2][2]
    C44=tensor_array.et_tensor[3][3]
    
    
