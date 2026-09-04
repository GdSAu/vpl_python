import numpy as np
import time
import pandas as pd
from .Reconstructor import Reconstructor
from utils.utils_save import GuardarDS, save_camera_trayectory

class ReconstructorPointCloud(Reconstructor):
    def __init__(self, file_name):
        super().__init__(file_name)

    def runReconstructionMultipleWithoutCondition(self,initialpose: int = 116):
        self._initProcess(type_process="multiple",initialpose=initialpose)
        I = 0
        print("Initializing reconstruction MaxViews process ...")
        #while condicion == False:
        for i in range(0,self.max_views):    
            # RGBD and pointcloud extraction
            self._saveData(i)
            #UpdateModels
            self._updateModels(i)            
            ## Aqui evaluamos si esta completo el modelo en este punto
            CD, _, coverage_gain, cov = self._evaluateModel(i)
            start_time = time.time()    
            self.eye = self.viewPlanner.PlanNBV()
            self.eyes.append(self.eye)
            print("--- %s seconds ---" % (time.time() - start_time))
            GuardarDS(self.metrics,I, i, self.objeto, self.eye_init, self.eye, self.carpeta_iter, CD, coverage_gain,cov)
            I += 1
        del self.scene
        del self.render

        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metrics, index=None)
        dataframe.to_csv(self.direccion + self.csv_name ,index=False)
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
