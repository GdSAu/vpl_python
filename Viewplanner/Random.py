from Viewplanner.viewPlanner import NBVPlanner
import numpy as np


class RandomPlanner(NBVPlanner):
    def __init__(self, robot_sensor, partial_model, minrange=0):
        super().__init__(robot_sensor, partial_model)
        self.poses_list = np.loadtxt("/mnt/6C24E28478939C77/Saulo/vpl_python/stuff/pcnbv/viewspace_shapenet_33_normal.txt") #np.load("stuff/poses.npy")
        self.poses_len = len(self.poses_list)
        self.random = np.random.default_rng()
        self.minrange = minrange
        self.maxrange = self.poses_len - 1

    def __getRandomView(self):
        randomnumber = self.random.integers(self.minrange,self.maxrange) # random uniform discrete distribution
        return randomnumber

    
    def savePartialModel(self,file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)
    
    
    def updateWithScan(self,**kwargs):
        pc_file = kwargs["pointcloud"]
        origin = kwargs["origin"]
        self.partial_model.updateWithScan(pc_file, origin)

    
    def PlanNBV(self):
        super().PlanNBV()
        random = self.__getRandomView()
        nbv = self.poses_list[random]
        return nbv