from Viewplanner.viewPlanner import NBVPlanner
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import stuff.nbvnet.nbvnet as nbvnet

class NBVNET(NBVPlanner):
    def __init__(self, robot_sensor, partial_model, weights_file):
        super().__init__(robot_sensor, partial_model)
        self.robot_sensor = robot_sensor
        self.partial_model = partial_model
        self.device = torch.cuda.current_device()
        self.weigths = weights_file
        self.nbv_positions = np.genfromtxt('stuff/nbvnet/points_in_sphere.txt')
        self.__netPreparation()

    def __netPreparation(self, dropout=0.2):
        #load NBV-Net
        self.net = nbvnet.NBV_Net(dropout)
        self.net.to(self.device)
        self.net.eval()
        #load weights
        self.net.load_state_dict(torch.load(self.weigths, map_location='cpu'), strict=False)
        

    def __netInference(self,grids):
        #grids = torch.from_numpy(np.array([[grids]]))
        grids = grids.type(torch.FloatTensor)
                
        # wrap them in a torch Variable
        grids = Variable(grids)    
        grids = grids.to(self.device)

        # forward pass to get net output
        output = self.net.forward(grids)

        #print("processing time:", end-start)

        grids = grids.cpu()
        output = output.cpu()
        output = output.detach().numpy()
        output = output.squeeze()

        #print(output)

        class_nbv = np.where(output == np.amax(output))
        return class_nbv

    def __getPositions(self, nbv_class, positions):
        return np.array(positions[nbv_class])

    def __getPose(self, nbv_class,scale=2.5):
        bar_nbv = self.__getPositions(nbv_class, self.nbv_positions) 
        nbv_scale = scale
        bar_nbv  = nbv_scale * bar_nbv
        bar_nbv = bar_nbv.squeeze()
        return bar_nbv


    def savePartialModel(self, file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)
    
    def updateWithScan(self,**kwargs):
        pc_file = kwargs["pointcloud"]
        origin = kwargs["origin"]
        self.partial_model.updateWithScan(pc_file, origin)


    def PlanNBV(self):
        super().PlanNBV()
        grid = np.reshape(self.partial_model.getOccupancyProbs(), (1,1,32,32,32))  
        torch_grid = torch.from_numpy(grid)
        #IA-NBV
        inferen = self.__netInference(torch_grid) 
        nbv = self.__getPose(inferen)
        #nbv = output.numpy().reshape(3,).astype("double") 
        return nbv

