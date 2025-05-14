import os
from Partialmodel.partialModel import PMOctomapPy, PMPointCloudPy
import octomap
import numpy as np
import pandas as pd
import time

from utils.utils_o3d import Get_PointcloudGT, Get_voxpoints
from Simulator.sceneLoader import SceneLoader
from Robotsensor.sensor import Sensor
from utils.reconstructorParams import Params
from utils.utils_metrics import chamfer_distance, Get_cloud_distance, getCobertura
from utils.utils_save import GuardarDS

class Reconstructor:

    def __init__(self,file_name):
        self.file_name = file_name
        self.metrics = {"ID": [],"id_objeto": [], "iteracion_objeto":[],"pose_inicial":[], "nube_puntos":[], "rejilla":[], "nbv":[], "id_anterior":[], "id_siguiente":[], "chamfer":[], "ganancia_cobertura":[], "cobertura":[]}
        self.__loadParams()
    
    def __readParams(self):
        self.params = Params(self.file_name)

    def __loadParams(self):
        self.__readParams()
        self.carpeta_metodo = self.params.getParameter("carpetas.carpeta_metodo")
        self.carpeta_iter = self.params.getParameter("carpetas.iter")
        self.direccion = self.params.getParameter("carpetas.direccion")
        self.objeto = self.params.getParameter("carpetas.objeto")
        #self.direccion = self.direccion + self.objeto + "/"
        self.csv_name = self.params.getParameter("carpetas.csv_name")
        self.umbral = self.params.getParameter("variables.umbral")
        self.max_views = self.params.getParameter("variables.maximum_views")
        self.img_H = self.params.getParameter("camera.img_H")
        self.img_W = self.params.getParameter("camera.img_W")
        self.up = self.params.getParameter("camera.up")
        self.fov = self.params.getParameter("camera.fov")
        self.requieredPM = self.params.getParameter("data_structure")
        self.planner = self.params.getParameter("planner")
        self.floor = self.params.getParameter("simulation.requieres_floor")
        self.dirtoPC = self.direccion + "Point_cloud/" + self.carpeta_metodo + self.carpeta_iter
    
    def __createFolder(self):
        try:
            if os.path.lexists(self.direccion+"Point_cloud/") == False:
                os.mkdir(self.direccion +"Point_cloud/" )
                os.mkdir(self.direccion +"Octree/")
                os.mkdir(self.direccion + "RGB/" )
                os.mkdir(self.direccion + "Depth/")
            if os.path.lexists(self.direccion+"Point_cloud/"+ self.carpeta_metodo) == False:
                os.mkdir(self.direccion +"Point_cloud/" + self.carpeta_metodo)
                os.mkdir(self.direccion +"Octree/"+ self.carpeta_metodo)
                os.mkdir(self.direccion + "RGB/" + self.carpeta_metodo)
                os.mkdir(self.direccion + "Depth/" + self.carpeta_metodo)
            if os.path.lexists(self.direccion+"Point_cloud/"+ self.carpeta_metodo) == True:
                os.mkdir(self.direccion +"Point_cloud/" + self.carpeta_metodo + "/"+ self.carpeta_iter)
                os.mkdir(self.direccion +"Octree/"+ self.carpeta_metodo+ "/"+ self.carpeta_iter)
                os.mkdir(self.direccion + "RGB/" + self.carpeta_metodo+ "/"+ self.carpeta_iter)
                os.mkdir(self.direccion + "Depth/" + self.carpeta_metodo+ "/"+ self.carpeta_iter)

        except:
            print("La carpeta de {} ya existe, no se sobreescribe".format(self.carpeta_iter))

    def __createFolderMulti(self):
        try:
            if os.path.lexists(self.direccionobj+"Point_cloud/") == False:
                os.mkdir(self.direccionobj +"Point_cloud/" )
                os.mkdir(self.direccionobj +"Octree/")
                os.mkdir(self.direccionobj + "RGB/" )
                os.mkdir(self.direccionobj + "Depth/")
            if os.path.lexists(self.direccionobj+"Point_cloud/"+ self.carpeta_metodo) == False:
                os.mkdir(self.direccionobj +"Point_cloud/" + self.carpeta_metodo)
                os.mkdir(self.direccionobj +"Octree/"+ self.carpeta_metodo)
                os.mkdir(self.direccionobj + "RGB/" + self.carpeta_metodo)
                os.mkdir(self.direccionobj + "Depth/" + self.carpeta_metodo)
            if os.path.lexists(self.direccionobj+"Point_cloud/"+ self.carpeta_metodo) == True:
                os.mkdir(self.direccionobj +"Point_cloud/" + self.carpeta_metodo + "/"+ self.carpeta_iter)
                os.mkdir(self.direccionobj +"Octree/"+ self.carpeta_metodo+ "/"+ self.carpeta_iter)
                os.mkdir(self.direccionobj + "RGB/" + self.carpeta_metodo+ "/"+ self.carpeta_iter)
                os.mkdir(self.direccionobj + "Depth/" + self.carpeta_metodo+ "/"+ self.carpeta_iter)
        except:
            print("La carpeta de {} ya existe, no se sobreescribe".format(self.carpeta_iter))

    def __initSensor(self,render,scene):
        self.sensor = Sensor(self.fov,self.up,self.img_W,self.img_H,render=render,scene=scene)
        
    
    def __initViewPlanner(self):
        if self.planner == "AutoEncoder":
            try:
                from Viewplanner.AENBV import AENBV
                self.viewPlanner = AENBV("None", self.PM, self.params.getParameter("carpetas.mlp_weights"), self.params.getParameter("carpetas.ae_weights"))
            except Exception as e:
                print("Error while loading AutoEncoder : {}".format(e))
        
        if self.planner == "NBVNet":
            try:
                from Viewplanner.NBVNET import NBVNET
                self.viewPlanner = NBVNET("None", self.PM, self.params.getParameter("carpetas.weights"))
            except Exception as e:
                print("Error while loading NBVNet : {}".format(e))
        
        if self.planner == "PCNBV":
            try:
                from Viewplanner.PCNBV import PCNBV
                self.viewPlanner = PCNBV("None", self.PM, self.params.getParameter("carpetas.weights"), self.params.getParameter("carpetas.viewspace"))
            except Exception as e:
                print("Error while loading PCNBV : {}".format(e))

    def __initPartialModel(self):
        if self.requieredPM == "octree":
            self.PM = PMOctomapPy(self.params.getParameter("variables.voxel_resolution"),self.params.getParameter("variables.voxel_dim"))
        if self.requieredPM == "pointcloud":
            self.PM = PMPointCloudPy(self.params.getParameter("variables.umbral"))

    def runReconstuctor(self):
        self.__initPartialModel()
        self.__initViewPlanner()
        self.direccion = self.direccion + self.objeto + "/"
        self.__createFolder()
        #Cargamos malla
        miEscena = SceneLoader(self.direccion,self.floor)
        render, scene = miEscena.get_scenes(self.img_H,self.img_W)
        
        #Obtenemos pointcloud GT
        Get_PointcloudGT(self.direccion, miEscena.mesh, self.carpeta_metodo + self.carpeta_iter)

        #Camera vectors setup
        cent = miEscena.mesh.get_center()
        
        poses = np.load("stuff/poses.npy")
        eye_init = poses[116]
        eye = eye_init
        
        #Set sensor
        self.__initSensor(render,scene)

        I = 0
        print("Initializing reconstruction process ...")
        #while condicion == False:
        for i in range(0,self.max_views):    
            # RGBD and pointcloud extraction
            self.sensor.savePointCloud(cent, eye, file_name= self.dirtoPC  + self.params.getParameter("filenames.Pointcloud").format(i) )
            self.sensor.saveAccPointCloud(i,direction = self.dirtoPC, file_name= self.dirtoPC + self.params.getParameter("filenames.Pointcloud"))
            self.sensor.saveRGBD(cent, eye, 
                                 rgb_file_name= self.direccion + "RGB/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.RGB").format(i), 
                                 depth_file_name =self.direccion + "Depth/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Depth").format(i))
            #UpdateModels
            if self.requieredPM == "octree":
                self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i)  , origin = eye)
                self.viewPlanner.savePartialModel(file_name= self.direccion + "Octree/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Octree").format(i) )
            
            if self.requieredPM == "pointcloud":
                self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i))

            condicion = False
            ## Aqui evaluamos si esta completo el modelo en este punto
            CD = chamfer_distance(self.direccion + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter)
            condicion, coverage_gain = Get_cloud_distance(self.direccion + "Point_cloud/" + self.carpeta_metodo , i, self.carpeta_iter)
            cov = getCobertura(self.direccion + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter, i, umbral=self.umbral)
            #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
            if condicion == True:
                GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain, cov)
                break
            ## De no estarlo, se consulta a la NN el NBV 
            else:
                eye = self.viewPlanner.PlanNBV()
                GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain,cov) 
            #print("nbv:", eye)
            I += 1
        del scene
        del render

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccion + self.csv_name ,index=False)

    def runReconstuctorWoC(self):
        self.__initPartialModel()
        self.__initViewPlanner()
        self.direccion = self.direccion + self.objeto + "/"
        self.__createFolder()
        self.dirtoPC = self.direccion + "Point_cloud/" + self.carpeta_metodo + self.carpeta_iter
        #Cargamos malla
        miEscena = SceneLoader(self.direccion,self.floor)
        render, scene = miEscena.get_scenes(self.img_H,self.img_W)
        
        #Obtenemos pointcloud GT
        Get_PointcloudGT(self.direccion, miEscena.mesh, self.carpeta_metodo + self.carpeta_iter)

        #Camera vectors setup
        cent = miEscena.mesh.get_center()
        
        poses = np.load("stuff/poses.npy")
        eye_init = poses[116]
        eye = eye_init
        
        #Set sensor
        self.__initSensor(render,scene)

        I = 0
        print("Initializing reconstruction process ...")
        #while condicion == False:
        for i in range(0,self.max_views):    
            # RGBD and pointcloud extraction
            self.sensor.savePointCloud(cent, eye, file_name= self.dirtoPC  + self.params.getParameter("filenames.Pointcloud").format(i) )
            self.sensor.saveAccPointCloud(i,direction = self.dirtoPC, file_name= self.dirtoPC + self.params.getParameter("filenames.Pointcloud"))
            self.sensor.saveRGBD(cent, eye, 
                                 rgb_file_name= self.direccion + "RGB/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.RGB").format(i), 
                                 depth_file_name =self.direccion + "Depth/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Depth").format(i))
            #UpdateModels
            if self.requieredPM == "octree":
                self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i)  , origin = eye)
                self.viewPlanner.savePartialModel(file_name= self.direccion + "Octree/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Octree").format(i) )
            
            if self.requieredPM == "pointcloud":
                self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i))

            condicion = False
            ## Aqui evaluamos si esta completo el modelo en este punto
            CD = chamfer_distance(self.direccion + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter)
            condicion, coverage_gain = Get_cloud_distance(self.direccion + "Point_cloud/" + self.carpeta_metodo , i, self.carpeta_iter)
            cov = getCobertura(self.direccion + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter, i, umbral=self.umbral)
            #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
            start_time = time.time()    
            eye = self.viewPlanner.PlanNBV()
            print("--- %s seconds ---" % (time.time() - start_time))
            GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain,cov)
            #print("nbv:", eye)
            I += 1
        del scene
        del render

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccion + self.csv_name ,index=False)

    def runReconstuctorMultiple(self):
        self.objectFolder = self.params.getParameter("carpetas.objectFolder")
        self.direccionf = self.direccion + self.objectFolder+ "/"
        self.listado_objetos = os.listdir(self.direccionf)
        for l in range (0, len(self.listado_objetos)):
            self.__initPartialModel()
            self.__initViewPlanner()
            #Cargamos malla
            self.direccionobj = self.direccionf + self.listado_objetos[l] + "/"
            self.__createFolderMulti()
            self.objeto = self.listado_objetos[l]
            miEscena = SceneLoader(self.direccionobj,self.floor)
            render, scene = miEscena.get_scenes(self.img_H,self.img_W)
            self.dirtoPC = self.direccionobj + "Point_cloud/" + self.carpeta_metodo + self.carpeta_iter
            #Obtenemos pointcloud GT
            Get_PointcloudGT(self.direccionobj, miEscena.mesh, self.carpeta_metodo + self.carpeta_iter)

            #Camera vectors setup
            cent = miEscena.mesh.get_center()
            
            poses = np.load("stuff/poses.npy")
            eye_init = poses[116]
            eye = eye_init
            
            #Set sensor
            self.__initSensor(render,scene)

            I = 0
            print("Initializing reconstruction process ...")
            #while condicion == False:
            for i in range(0,self.max_views):    
                # RGBD and pointcloud extraction
                self.sensor.savePointCloud(cent, eye, file_name= self.dirtoPC  + self.params.getParameter("filenames.Pointcloud").format(i) )
                self.sensor.saveAccPointCloud(i,direction = self.dirtoPC, file_name= self.dirtoPC + self.params.getParameter("filenames.Pointcloud"))
                self.sensor.saveRGBD(cent, eye, 
                                    rgb_file_name= self.direccionobj + "RGB/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.RGB").format(i), 
                                    depth_file_name =self.direccionobj + "Depth/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Depth").format(i))
                #UpdateModels
                if self.requieredPM == "octree":
                    self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i)  , origin = eye)
                    self.viewPlanner.savePartialModel(file_name= self.direccionobj + "Octree/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Octree").format(i) )
                
                if self.requieredPM == "pointcloud":
                    self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i))

                ## Aqui evaluamos si esta completo el modelo en este punto
                CD = chamfer_distance(self.direccionobj + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter)
                condicion, coverage_gain = Get_cloud_distance(self.direccionobj + "Point_cloud/" + self.carpeta_metodo , i, self.carpeta_iter)
                cov = getCobertura(self.direccionobj + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter, i, umbral=self.umbral)
                #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
                if condicion == True:
                    GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain, cov)
                    break
                ## De no estarlo, se consulta a la NN el NBV 
                else:
                    eye = self.viewPlanner.PlanNBV()
                    GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain,cov) 
                #print("nbv:", eye)
                I += 1
            del scene
            del render
            del miEscena
            del self.sensor
            del self.PM
            del self.viewPlanner

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccionf + self.csv_name ,index=False)

    def runReconstuctorMultipleWoC(self):

        self.objectFolder = self.params.getParameter("carpetas.objectFolder")
        self.direccionf = self.direccion + self.objectFolder+ "/"
        self.listado_objetos = os.listdir(self.direccionf)
        for l in range (0, len(self.listado_objetos)):
            self.__initPartialModel()
            self.__initViewPlanner()
            #Cargamos malla
            self.direccionobj = self.direccionf + self.listado_objetos[l] + "/"
            self.__createFolderMulti()
            self.objeto = self.listado_objetos[l]
            miEscena = SceneLoader(self.direccionobj,self.floor)
            render, scene = miEscena.get_scenes(self.img_H,self.img_W)
            self.dirtoPC = self.direccionobj + "Point_cloud/" + self.carpeta_metodo + self.carpeta_iter
            #Obtenemos pointcloud GT
            Get_PointcloudGT(self.direccionobj, miEscena.mesh, self.carpeta_metodo + self.carpeta_iter)

            #Camera vectors setup
            cent = miEscena.mesh.get_center()
            
            poses = np.load("stuff/poses.npy")
            eye_init = poses[116]
            eye = eye_init
            
            #Set sensor
            self.__initSensor(render,scene)

            I = 0
            print("Initializing reconstruction process ...")
            #while condicion == False:
            for i in range(0,self.max_views):    
                # RGBD and pointcloud extraction
                self.sensor.savePointCloud(cent, eye, file_name= self.dirtoPC  + self.params.getParameter("filenames.Pointcloud").format(i) )
                self.sensor.saveAccPointCloud(i,direction = self.dirtoPC, file_name= self.dirtoPC + self.params.getParameter("filenames.Pointcloud"))
                self.sensor.saveRGBD(cent, eye, 
                                    rgb_file_name= self.direccionobj + "RGB/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.RGB").format(i), 
                                    depth_file_name =self.direccionobj + "Depth/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Depth").format(i))
                #UpdateModels
                if self.requieredPM == "octree":
                    self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i)  , origin = eye)
                    self.viewPlanner.savePartialModel(file_name= self.direccionobj + "Octree/" + self.carpeta_metodo + self.carpeta_iter + self.params.getParameter("filenames.Octree").format(i) )
                
                if self.requieredPM == "pointcloud":
                    self.viewPlanner.updateWithScan(pointcloud = self.dirtoPC + self.params.getParameter("filenames.Pointcloud").format(i))

                ## Aqui evaluamos si esta completo el modelo en este punto
                CD = chamfer_distance(self.direccionobj + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter)
                condicion, coverage_gain = Get_cloud_distance(self.direccionobj + "Point_cloud/" + self.carpeta_metodo , i, self.carpeta_iter)
                cov = getCobertura(self.direccionobj + "Point_cloud/" + self.carpeta_metodo, self.carpeta_iter, i, umbral=self.umbral)
                #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
                eye = self.viewPlanner.PlanNBV()
                GuardarDS(self.metrics,I, i, self.objeto, eye_init, eye, self.carpeta_iter, CD, coverage_gain,cov) 
                #print("nbv:", eye)
                I += 1
            del scene
            del render
            del miEscena
            del self.sensor
            del self.PM
            del self.viewPlanner

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccion + self.csv_name ,index=False)