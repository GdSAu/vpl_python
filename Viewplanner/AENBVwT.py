from Viewplanner.viewPlanner import NBVPlanner
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Codificador(nn.Module):
    def __init__(self, input_channels, latent_space):
        super(Codificador, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, 5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1)
        self.mpool = nn.MaxPool3d(2, 2, return_indices=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6912, 128)
        self.linear2 = nn.Linear(128, latent_space)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x, index = self.mpool(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x, index

class Decodificador(nn.Module):
    def __init__(self, latent_space, output_channels=1):
        super(Decodificador, self).__init__()
        self.linear3 = nn.Linear(latent_space, 128)
        self.linear4 = nn.Linear(128, 6912)
        self.uflatten = nn.Unflatten(1, (32, 6, 6, 6))
        self.umpool = nn.MaxUnpool3d(2, stride=2)
        self.tconv1 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=1)
        self.tconv2 = nn.ConvTranspose3d(32, output_channels, kernel_size=5, stride=2, padding=0)
    
    def forward(self, x, index):
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.uflatten(x)
        x = self.umpool(x, index)
        x = F.leaky_relu(self.tconv1(x))
        x = F.leaky_relu(self.tconv2(x))
        return x

class VoxNetAutoencoder(nn.Module):
    def __init__(self, latent_space, input_channels=1, output_channels=1):
        super(VoxNetAutoencoder, self).__init__()
        self.latent_space = latent_space
        self.codificador = Codificador(input_channels, self.latent_space)
        self.decodificador = Decodificador(self.latent_space, output_channels)
    
    def forward(self, x):
        codificacion, indices = self.codificador(x)
        reconstruccion = self.decodificador(codificacion, indices)
        return reconstruccion


class MLP(nn.Module):
        def __init__(self,input = 16, output = 3):
            super(MLP, self).__init__()
            self.input = input
            self.output = output
            self.mlp = nn.Sequential(nn.Linear(self.input,2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048,512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512,512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, self.output),
                                    nn.Tanh())
        
        def forward(self,x):
            
            x = self.mlp(x)
            return x

class AE():
    def __init__(self, weights):
        self.weightspath = weights
        pass
     
    def __get_latent_space(self,grids, device):
        model1= VoxNetAutoencoder(latent_space = 16).cuda()
        model1.load_state_dict(torch.load(self.weightspath))
        
        #grids = grids.type(torch.FloatTensor)
            # wrap them in a torch Variable
        grids = Variable(grids)    
        grids = grids.to(device)
            # forward pass to get net coder output
        output = model1.codificador(grids)
        grids = grids.cpu()
        output = output[0].cpu()
        del model1
        del grids
        return output

    
    def net_position_nbv(self, model, grid, device):
    
        with torch.no_grad():
            grid = grid.type(torch.FloatTensor)# no pude transformar en get_l
            l_s = self.__get_latent_space(grid, device)
            l_s = Variable(l_s)  
            l_s = l_s.to(device)
            # forward pass to get net output
            output = model(l_s)
            output = output.cpu()
        return output
    



class AENBVwT(NBVPlanner):
    def __init__(self, robot_sensor, partial_model, mlp_weights, ae_weights):
        super().__init__(robot_sensor, partial_model)
        self.robot_sensor = robot_sensor
        self.partial_model = partial_model
        self.mlp_weights = mlp_weights
        self.ae_weights = ae_weights
        self.mlp = self.__loadMLP()
        self.device = torch.cuda.current_device()
        self.ae = AE(self.ae_weights)
        self.lbv= torch.zeros(1, 3)


    def __loadMLP(self):
        model = MLP().cuda()
        model.load_state_dict(torch.load(self.mlp_weights))
        return model
    
    def __sample_spherical_noise(self, device, n=1, radius=0.4, dtype=torch.float32):
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
        theta = torch.rand(n, device=device, dtype=dtype) * 2 * torch.pi
        # Ángulo polar (0, π), usando muestreo uniforme en la esfera
        u = torch.rand(n, device=device, dtype=dtype)
        cos_phi_min = torch.cos(torch.tensor(torch.pi / 2, dtype=dtype, device=device))  # = 0
        cos_phi_max = torch.cos(torch.tensor(0.0, dtype=dtype, device=device))           # = 1
        cos_phi = torch.lerp(cos_phi_min, cos_phi_max, u)  # lineal entre cos(π/2) y cos(0)
        phi = torch.acos(cos_phi)

        # Conversión a coordenadas cartesianas
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)

        return torch.stack((x, y, z), dim=1)  # Tensor de forma (n, 3)

    def savePartialModel(self, file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)
    
    def updateWithScan(self,**kwargs):
        pc_file = kwargs["pointcloud"]
        origin = kwargs["origin"]
        self.partial_model.updateWithScan(pc_file, origin)


    def PlanNBV(self):
        super().PlanNBV()
        grid = np.reshape(self.partial_model.getOccupancyProbs(), (1,1,31,31,31))  
        torch_grid = torch.from_numpy(grid)
        #IA-NBV
        output= self.ae.net_position_nbv(model= self.mlp, grid=torch_grid, device=self.device)
        """
        if torch.equal(output, self.lbv) == True:
            noise = self.__sample_spherical_noise(self.device)
            print("Added noise:", noise)
            nbv = output + noise.cpu()
            nbv = nbv.numpy().reshape(3,).astype("double") 
        else:
            nbv = output
            nbv = nbv.numpy().reshape(3,).astype("double") 
        self.lbv = output
        """
        noise = self.__sample_spherical_noise(self.device)
        nbv = output + noise.cpu()
        nbv = nbv.numpy().reshape(3,).astype("double") 
        return nbv
    