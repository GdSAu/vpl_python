from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import octomap

class PartialModelBase:
    def __init__(self):
        self.object_points_filename = ""
        self.objectPointCloud = None
        self.objectSphereCenter = None
        self.objectSphereRadius = 0.0
        self.objRadius2 = 0.0
        self.directorRay = None
        self.configFolder = ""
        self.dataFolder = ""
        self.evaluationsFile = ""
        self.candidateViews = []
        self.rays = []
        self.minDOV = 0.0
        self.maxDOV = 0.0
        self.scanOrigin = None
        self.scanCloudOrigins = None
        self.scanCloud = None
        # Object Capsule variables
        self.x_cap_1, self.y_cap_1, self.z_cap_1 = 0.0, 0.0, 0.0
        self.x_cap_2, self.y_cap_2, self.z_cap_2 = 0.0, 0.0, 0.0
        # Scene capsule
        self.x_sce_1, self.y_sce_1, self.z_sce_1 = 0.0, 0.0, 0.0
        self.x_sce_2, self.y_sce_2, self.z_sce_2 = 0.0, 0.0, 0.0
        self.objectResolution = 0.0
    
    
    @abstractmethod
    def updateWithScan(self, file_name_scan: str, file_name_origin: str) -> float:
        pass
    
    def evaluateCandidateViews(self):
        pass
    
    def evaluateCandidateViews(self, views: 'ViewList'):
        pass
    
    @abstractmethod
    def evaluateView(self, v: 'ViewStructure') -> int:
        pass
    
    @abstractmethod
    def stopCriteriaReached(self) -> bool:
        pass
    
    @abstractmethod
    def savePartialModel(self, file_name: str) -> bool:
        pass
    
    @abstractmethod
    def loadPartialModel(self, file_name: str) -> bool:
        pass
    
    @abstractmethod
    def getUnknownVolume(self) -> float:
        pass
    
    @abstractmethod
    def readRays(self, file_address: str) -> int:
        pass
    
    @abstractmethod
    def insertUnknownSurface(self, pc: 'Pointcloud') -> bool:
        pass
    
    def init(self) -> bool:
        pass
    
    def saveEvaluatedViews(self, file_name: str) -> bool:
        pass
    
    def readCandidateViews(self, file_name: str) -> bool:
        pass
    
    def saveOnlyNViews(self, n: int, file_name: str) -> bool:
        pass
    
    def sortCandidateViews(self):
        pass
    
    def setObjectCapsule(self, x1: float, y1: float, z1: float, 
                         x2: float, y2: float, z2: float):
        self.x_cap_1, self.y_cap_1, self.z_cap_1 = x1, y1, z1
        self.x_cap_2, self.y_cap_2, self.z_cap_2 = x2, y2, z2
    
    def setScene(self, x1: float, y1: float, z1: float, 
                x2: float, y2: float, z2: float):
        self.x_sce_1, self.y_sce_1, self.z_sce_1 = x1, y1, z1
        self.x_sce_2, self.y_sce_2, self.z_sce_2 = x2, y2, z2
    
    def setConfigFolder(self, folder: str):
        self.configFolder = folder
    
    def setDataFolder(self, folder: str):
        self.dataFolder = folder
    

class PMOctomapPy(PartialModelBase):
    def __init__(self):
        self.octree = None
        self.resolution = 0.0
        self.unknownVolume = 0.0
        self.unknownVolumeThreshold = 0.0
        self.unknownVolumeThresholdReached = False

    def init(self):
        super().init()
        # El Partial model va a leer su configuración y va a generar el octree

        #Inicializamos el octomap
        octree = octomap.OcTree(self.voxel_resolution) # inicializamos el octree
        return True
        
    def updateWithScan(self, file_name_scan: str, file_name_origin: str):
        pass

