from Viewplanner.viewPlanner import NBVPlanner
import numpy as np
import torch


class RandomContinuousPlanner(NBVPlanner):
    def __init__(self, robot_sensor, partial_model):
        super().__init__(robot_sensor, partial_model)
        self.device = torch.cuda.current_device()

    def __getsample_spherical_noise(self, n=1, radius=0.4, dtype=torch.float32):
        """
        Genera n puntos uniformemente distribuidos sobre la superficie de una esfera.

        Parámetros:
        - n: número de puntos a generar
        - radius: radio de la esfera
        - device: dispositivo ('cpu' o 'cuda')
        - dtype: tipo de dato (por ejemplo, torch.float32)

        Retorna:
        - Tensor de tamaño (n, 3) con coordenadas [x, y, z]
        """
        # Ángulo azimutal (0, 2π)
        theta = torch.rand(n, device=self.device, dtype=dtype) * 2 * torch.pi
        # Ángulo polar (0, π), usando muestreo uniforme en la esfera
        u = torch.rand(n, device=self.device, dtype=dtype)
        phi = torch.acos(u)

        # Conversión a coordenadas cartesianas
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)

        return torch.stack((x, y, z), dim=0).cpu()  # Tensor de forma (n, 3)

    
    def savePartialModel(self,file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)
    
    
    def updateWithScan(self,**kwargs):
        pc_file = kwargs["pointcloud"]
        origin = kwargs["origin"]
        self.partial_model.updateWithScan(pc_file, origin)

    
    def PlanNBV(self):
        super().PlanNBV()
        nbv = self.__getsample_spherical_noise()
        nbv = nbv.numpy().reshape(3,).astype("double") 
        return nbv