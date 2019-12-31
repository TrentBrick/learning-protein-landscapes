
import torch
import numpy as np
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.device):
            return str(obj)
        return json.JSONEncoder.default(self, obj)