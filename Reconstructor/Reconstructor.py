import os
import octomap
import numpy as np
import pandas as pd

from utils.utils_o3d import Get_PointcloudGT, Get_voxpoints
from Simulator.sceneCreator import SceneLoader
from Robotsensor.sensor import Sensor
from Partialmodel.modelupdate import updateModel
from utils.reconstructorParams import Params

class Reconstructor:

    def __init__(self,file_name):
        self.file_name = file_name
        self.metricas = {"ID": [],"id_objeto": [], "iteracion_objeto":[],"pose_inicial":[], "nube_puntos":[], "rejilla":[], "nbv":[], "id_anterior":[], "id_siguiente":[], "chamfer":[], "ganancia_cobertura":[], "cobertura":[]}
        self.__loadParams()
    
    def __readParams(self):
        self.params = Params(self.file_name)

    def __loadParams(self):
        self.__readParams()
        self.carpeta_metodo = self.params.getCarpetaMetodo()
        self.carpeta_iter = self.params.getCarpetaIter()
        self.direccion = self.params.getDireccionCarpeta()
        self.objeto = self.params.getObjetoCarpeta()
        self.direccion = self.direccion + self.objeto + "/"
        self.listado_objetos = os.listdir(self.direccion)
        #self.weights_path = self.params.getPesosCarpeta()
        self.csv_name = self.params.getCSVName()
        self.umbral = self.params.getUmbralVariable()
        self.max_views = self.params.getMaximumVariable()
        self.img_H, self.img_W, self.up, self.fov = self.params.getCameraParams()
        self.voxel_resolution = self.params.getVoxelVariable()
        self.dim_arreglo = 32768
    
    def __createFolder(self, dir_carpeta):
        try:
            if os.path.lexists(dir_carpeta+"Point_cloud/"+self.carpeta_metodo) == False:
                os.mkdir(dir_carpeta +"Point_cloud/" + self.carpeta_metodo)
                os.mkdir(dir_carpeta +"Octree/"+ self.carpeta_metodo)
                os.mkdir(dir_carpeta + "RGB/" + self.carpeta_metodo)
                os.mkdir(dir_carpeta + "Depth/" + self.carpeta_metodo)
            if os.path.lexists(dir_carpeta+"Point_cloud/"+ self.carpeta_metodo) == True:
                os.mkdir(dir_carpeta +"Point_cloud/"+ self.carpeta_iter)
                os.mkdir(dir_carpeta +"Octree/"+ self.carpeta_iter)
                os.mkdir(dir_carpeta + "RGB/" + self.carpeta_iter)
                os.mkdir(dir_carpeta + "Depth/"+ self.carpeta_iter)
        except:
            print("La carpeta de {} ya existe, no se sobreescribe".format(self.carpeta_iter))

    def __initSensor(self,dir_carpeta,render,scene):
        miSensor = Sensor(self.fov,self.up,dir_carpeta,self.carpeta_iter,self.img_W,self.img_H,render=render,scene=scene)
        return miSensor
    

    def runExperiment(self):
        I = 0
        for l in range (0, len(self.listado_objetos)):
            #carpeta = input("A que carpeta quieres acceder?: ") #object folder
            dir_carpeta = self.direccion + self.listado_objetos[l] + "/"
            
            #Creamos la carpeta contenedora del experimento
            self.__createFolder(dir_carpeta)
            
            # TODO: esto va en el viewplanner
            #Cargamos los modelos de predicción de posición
            model= MLP().cuda() 
            ## Modificar direccion de pesos
            model.load_state_dict(torch.load(self.weights_path))
            device = torch.cuda.current_device()

            #Inicializamos el octomap
            octree = octomap.OcTree(self.voxel_resolution) # inicializamos el octree

            #Cargamos malla
            miEscena = SceneLoader(dir_carpeta,self.floor)
            render, scene = miEscena.get_scenes(self.img_H,self.img_W)
            
            #Obtenemos pointcloud GT
            Get_PointcloudGT(dir_carpeta, miEscena.mesh, self.carpeta_iter)

            #Camera vectors setup
            cent = miEscena.mesh.get_center()
            
            poses = np.load("poses.npy")
            eye_init = poses[116]
            eye = eye_init
            
            puntos = Get_voxpoints()
            
            #Set sensor
            sensor = self.__initSensor(dir_carpeta,render,scene)
            #Set partialmodel
            partialmodel = updateModel(self.carpeta_iter)
            
            #print("Inicia el proceso de reconstrucción ...")
            #while condicion == False:
            for i in range(0,self.max_views):    
                # RGBD and pointcloud extraction
                sensor.Get_Pointcloud(cent, eye, i)
                sensor.Get_RGBD(cent, eye, i)
                #UpdateModels
                partialmodel.updatePointCloud(dir_carpeta,i)
                partialmodel.Get_octree(octree, dir_carpeta, i, eye)

                #TODO: Add viewplanner

                #Occupancy grid
                occupancy_probs =  partialmodel.getOccupancyProbs(puntos, self.dim_arreglo)
                ## Aqui evaluamos si esta completo el modelo en este punto
                CD = chamfer_distance(dir_carpeta, self.carpeta_iter)
                condicion, coverage_gain = Get_cloud_distance(dir_carpeta, i, self.carpeta_iter)
                cov = getCobertura(dir_carpeta, self.carpeta_iter, i, umbral=self.umbral)
                #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
                #if condicion == True:
                #    GuardarDS(self.metricas,I, i, self.listado_objetos[l], eye_init, eye, octree, occupancy_probs, dir_carpeta, self.carpeta_iter, CD, coverage_gain, cov)
                #    break
                ## De no estarlo, se consulta a la NN el NBV 
                #else:
                grid = np.reshape(occupancy_probs, (1,1,31,31,31))  
                torch_grid = torch.from_numpy(grid)
                #IA-NBV
                output = net_position_nbv(model, torch_grid, device) 
                eye = output.numpy().reshape(3,).astype("double")
                GuardarDS(self.metricas,I, i, self.listado_objetos[l], eye_init, eye, octree, occupancy_probs, dir_carpeta, self.carpeta_iter, CD, coverage_gain,cov) 
                #print("nbv:", eye)
                I += 1
            del octree
            del scene
            del render
            del miEscena

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metricas, index=None)
        dataframe.to_csv(self.csv_name ,index=False)