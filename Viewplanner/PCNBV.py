from Viewplanner.viewPlanner import NBVPlanner
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from stuff.pcnbv.models.pc_nbv import AutoEncoder

class PCNBV(NBVPlanner):
    def __init__(self, robot_sensor, partial_model, weights, viewspace_path):
        super().__init__(robot_sensor, partial_model)
        self.robot_sensor = robot_sensor
        self.partial_model = partial_model
        self.weights= weights
        self.device = torch.cuda.current_device()
        self.__initNetwork()
        self.viewspace= np.loadtxt(viewspace_path)
        self.viewstate =  np.zeros(self.viewspace.shape[0], dtype=np.int32) 
        self.scan_pc = np.zeros((1, 3))

    def __initNetwork(self):
        self.network = AutoEncoder()
        self.network.load_state_dict(torch.load(self.weights))
        self.network.to(self.device)
        self.network.eval()

    def __resample_pcd(self, pcd, n):
        """Drop or duplicate points so that pcd has exactly n points"""
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
        return pcd[idx[:n]]


    def savePartialModel(self, file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)
    
    def updateWithScan(self,**kwargs):
        pc_file = kwargs["pointcloud"]
        self.partial_model.updateWithScan(pc_file)


    def PlanNBV(self):
        super().PlanNBV()
        partial = np.asarray(self.partial_model.getPartialModel().points)
        self.scan_pc = np.append(self.scan_pc, partial, axis=0)
        partial = self.__resample_pcd(self.scan_pc, 1024)
        with torch.no_grad():
            partial_tensor = torch.tensor(partial[np.newaxis, ...].astype(np.float32)).permute(0, 2, 1).to(self.device)
            view_state_tensor = torch.tensor(self.viewstate.astype(np.float32))[np.newaxis, ...].to(self.device)
            _, eval_value = self.network(partial_tensor, view_state_tensor)
            eval_value = eval_value[0].cpu().detach().numpy()      
        view = np.argmax(eval_value, axis = 0)
        nbv = self.viewspace[view]
        self.viewstate[view] = 1
        # due to memory consumption
        del partial
        del partial_tensor
        del view_state_tensor
        del eval_value
        del view
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return nbv
    