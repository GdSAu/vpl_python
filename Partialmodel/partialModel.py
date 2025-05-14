from abc import ABC, abstractmethod
import numpy as np # type: ignore
import octomap # type: ignore
import open3d as o3d # type: ignore
import trimesh # type: ignore

class PartialModelBase(ABC):
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

    def readPointCloudfromPCD(self, file_name):
        try:
            cloud = o3d.io.read_point_cloud(file_name, remove_nan_points=True, remove_infinite_points = True)
        except Exception as e:
            print("Error while reading pointcloud : {}".format(e))
            return None
        return cloud    
    
    @abstractmethod
    def getPartialModel(self):
        pass

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
    
    def getUnknownVolume(self) -> float:
        pass
    
    def readRays(self, file_address: str) -> int:
        pass
    
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
    def __init__(self,voxel_resolution, voxel_voxdim=31):
        self.octree = None
        self.resolution = voxel_resolution
        self.unknownVolume = 0.0
        self.unknownVolumeThreshold = 0.0
        self.unknownVolumeThresholdReached = False
        self.voxel_voxdim = voxel_voxdim
        self.dim_arreglo =  self.voxel_voxdim *  self.voxel_voxdim *  self.voxel_voxdim
        self.voxel_points = self.__Get_voxpoints()
        self.init()

    def init(self):
        super().init()
        # El Partial model va a leer su configuración y va a generar el octree

        #Inicializamos el octomap
        self.octree = octomap.OcTree(self.resolution) # inicializamos el octree
        return True
        
    
    def __Get_voxpoints(self, max_dim=.4):
        #BBOX min & max
        resolution = max_dim/self.voxel_voxdim
        aabb_min = np.asarray([-(max_dim/2),-(max_dim/2),0.0])
        aabb_max = np.asarray([(max_dim/2),(max_dim/2),max_dim])
        center = (aabb_min + aabb_max) / 2
        dimension = np.array([self.voxel_voxdim, self.voxel_voxdim, self.voxel_voxdim]) # Voxelization dimensions
        origin = center - dimension/2  * resolution
        #New BBox given the new resolution
        grid = np.full(dimension, -1, np.int32)
        transform = trimesh.transformations.scale_and_translate(
            scale=resolution, translate=origin
        )
        # Voxelgrid encoding (create grid)
        points = trimesh.voxel.VoxelGrid(encoding=grid, transform=transform).points # Voxel grid con los puntos de la nube
        puntos = np.asarray(points)
        return puntos
    
    def getPartialModel(self):
        super().getPartialModel()
        return self.octree

    def updateWithScan(self, file_name_scan: str, origin):#
        try:
            
            scan_cloud = self.readPointCloudfromPCD(file_name_scan)
    
            self.octree.insertPointCloud(
                pointcloud= np.asarray(scan_cloud.points), 
                origin= np.asarray(origin), #Measurement origin
                maxrange=-1, # maximum range for how long individual beams are inserted
                )
            self.octree.updateInnerOccupancy()
            return True
        except Exception as e:
            print("Error while updating octree : {}".format(e))
            return False



    def evaluateView(self, v: 'ViewStructure') -> int:
        print("NO")
    
 
    def stopCriteriaReached(self) -> bool:
        return False

    def savePartialModel(self, file_name: str) -> bool:
        try:
            direccion_octree= bytes(file_name, encoding='utf8')
            self.octree.writeBinary(direccion_octree) 
            return True
        except Exception as e:
            print("Error while saving octree : {}".format(e))
            return False
        

    def loadPartialModel(self, file_name: str) -> bool:
        try:
            direccion_octree= bytes(file_name, encoding='utf8')
            self.octree.readBinary(direccion_octree)
            return True
        except Exception as e:
            print("Error while loading octree : {}".format(e))
            return False

    def getOccupancyProbs(self):
        arreglo = np.full((self.dim_arreglo), 0.5)
        j = 0 
        for i in self.voxel_points:
            #Get occupancy probability given a position (x,y,z)
            node = self.octree.search(i, 0)
            # If the returned value is different than -1, indicates that a node is in that position
            if node is not None:
                try:
                    probability = node.getOccupancy() #Can use getValues(), but extracts the log-odds
                    arreglo[j] = probability 
                except:
                    pass
            j += 1 
        return arreglo  

    def getUnknownVolume(self) -> float:
        occupation = self.getOccupancyProbs()         
        return np.sum(occupation == 0.5)
    

    def readRays(self, file_address: str) -> int:
        print("NO")
    

    def insertUnknownSurface(self, pc: 'Pointcloud') -> bool:
       print("NO")

class PMPointCloudPy(PartialModelBase):
    def __init__(self,voxel_resolution):
        self.pointcloud = None
        self.resolution = voxel_resolution
        self.init()

    def init(self):
        super().init()
        self.pointcloud = o3d.geometry.PointCloud()
        return True

    def getPartialModel(self):
        super().getPartialModel()
        return self.pointcloud
    
    def updateWithScan(self, file_name_scan: str):
        try:
            scan = self.readPointCloudfromPCD(file_name_scan)
            Nube_acc = o3d.geometry.PointCloud()
            Nube_acc.points = o3d.utility.Vector3dVector(np.vstack((self.pointcloud.points, scan.points)))
            self.pointcloud = Nube_acc.voxel_down_sample(voxel_size= self.resolution)
            return True
        except Exception as e:
            print("Error while updating pointcloud : {}".format(e))
            return False

    def evaluateCandidateViews(self):
        pass
    
    def evaluateCandidateViews(self, views: 'ViewList'):
        pass
    
    
    def evaluateView(self, v: 'ViewStructure') -> int:
        pass
    
   
    def stopCriteriaReached(self) -> bool:
        pass
    
    def savePartialModel(self, file_name: str) -> bool:
        try:
            o3d.io.write_point_cloud(file_name, self.pointcloud, write_ascii=True)
            return True
        except Exception as e:
            print("Error while saving pointcloud : {}".format(e))
            return False
    
    def loadPartialModel(self, file_name: str) -> bool:
        try:
            self.pointcloud = self.readPointCloudfromPCD(file_name)
            return True
        except Exception as e:
            print("Error while loading pointcloud : {}".format(e))
            return False

    