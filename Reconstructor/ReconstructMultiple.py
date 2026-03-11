import os
import numpy as np
import time
import pandas as pd
from .Reconstructor import Reconstructor
from utils.utils_save import GuardarDS, save_camera_trayectory


class ReconstructorMultiple(Reconstructor):
    def __init__(self, file_name):
        super().__init__(file_name)

    def _listObjects(self):
        self.objectFolder = self.params.getParameter("carpetas.objectFolder")
        self.direccionf = self.direccion + self.objectFolder+ "/"
        self.listado_objetos = os.listdir(self.direccionf)

    def _cleanVariables(self):
        del self.scene
        del self.render
        del self.miEscena
        del self.sensor
        del self.PM
        del self.viewPlanner

    def runReconstructionWithCondition(self, initialpose: int = 116):
            self._listObjects()

            for l in range (0, len(self.listado_objetos)):
                self._initProcess(initialpose, type_process="multiple", l=l)

                I = 0
                print("Initializing reconstruction process ...")
                #while condicion == False:
                for i in range(0,self.max_views):    
                    # RGBD and pointcloud extraction
                    self._saveData(i)
                    #UpdateModels
                    self._updateModels(i)    
                    ## Aqui evaluamos si esta completo el modelo en este punto
                    CD, condicion, coverage_gain, cov = self._evaluateModel(i)
                    if condicion == True:
                        GuardarDS(self.metrics,I, i, self.objeto, self.eye_init, self.eye, self.carpeta_iter, CD, coverage_gain, cov)
                        break
                    ## De no estarlo, se consulta a la NN el NBV 
                    else:
                        self.eye = self.viewPlanner.PlanNBV()
                        self.eyes.append(self.eye)
                        GuardarDS(self.metrics,I, i, self.objeto, self.eye_init, self.eye, self.carpeta_iter, CD, coverage_gain,cov) 
                    #print("nbv:", eye)
                    I += 1
                self._cleanVariables()
                save_camera_trayectory(
                    direccion=self.direccionobj,
                    eyes=self.eyes,
                    cent=self.cent,
                    up=self.up,
                    fov=self.fov,
                    width=self.img_W,
                    height=self.img_H,
                    method_name=str(self.carpeta_metodo),
                    iteration_name=str(self.carpeta_iter)
                )
            dataframe = pd.DataFrame(self.metrics, index=None)
            dataframe.to_csv(self.direccionf + self.csv_name ,index=False)
        
    def runReconstructionWithoutCondition(self, initialpose: int = 116):
        self._listObjects()

        for l in range (0, len(self.listado_objetos)):
            self._initProcess(initialpose, type_process="multiple", l=l)

            I = 0
            print("Initializing reconstruction process ...")
            #while condicion == False:
            for i in range(0,self.max_views):    
                # RGBD and pointcloud extraction
                self._saveData(i)
                #UpdateModels
                self._updateModels(i)    
                ## Aqui evaluamos si esta completo el modelo en este punto
                CD, _, coverage_gain, cov = self._evaluateModel(i)
                self.eye = self.viewPlanner.PlanNBV()
                self.eyes.append(self.eye)
                GuardarDS(self.metrics,I, i, self.objeto, self.eye_init, self.eye, self.carpeta_iter, CD, coverage_gain,cov) 
                #print("nbv:", eye)
                I += 1
            
            self._cleanVariables()
            save_camera_trayectory(
                direccion=self.direccionobj,
                eyes=self.eyes,
                cent=self.cent,
                up=self.up,
                fov=self.fov,
                width=self.img_W,
                height=self.img_H,
                method_name=str(self.carpeta_metodo),
                iteration_name=str(self.carpeta_iter)
            )
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccionf + self.csv_name ,index=False)
