import open3d as o3d
import numpy as np
from Robotsensor.utils_o3d_v2 import Nube_acumulada_filtrada

class updateModel:

    def __init__(self,itera):
        self.itera = itera

    def updatePointCloud(self, direccion, i, voxel_size=0.0005):
        # if the poincloud is new it begins the cloud, else the new perception is added to the accumulated cloud
        if i == 0:
            Nube_acc = o3d.io.read_point_cloud(direccion + "/Point_cloud/" + self.itera + "cloud_{}.pcd".format(i))
        else:
            p_acc = o3d.io.read_point_cloud(direccion + "/Point_cloud/" + self.itera + "cloud_acc.pcd")
            pcd = o3d.io.read_point_cloud(direccion + "/Point_cloud/" + self.itera + "cloud_{}.pcd".format(i))
            Nube_acc = o3d.geometry.PointCloud()
            Nube_acc.points = o3d.utility.Vector3dVector(Nube_acumulada_filtrada(p_acc,pcd, voxel_size=voxel_size).points)
        o3d.io.write_point_cloud(direccion + "/Point_cloud/"+ self.itera + "cloud_acc.pcd", Nube_acc, write_ascii=True)# accumulated cloud 

    def updateOctree(self, direccion, octree, i, origin):
        '''octree -> occupation prob [n,]
            octree: octree object
            direccion: root folder direction
            i = index in loop
            origin = camera eye (position)
            '''
        p_c = o3d.io.read_point_cloud(direccion + "/Point_cloud/"+ self.itera +"cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
        octree.insertPointCloud(
            pointcloud= np.asarray(p_c.points), 
            origin= np.asarray(origin), #Measurement origin
            maxrange=-1, # maximum range for how long individual beams are inserted
            )  
        direccion_octree= bytes(direccion + "/Octree/"+ self.itera +"octree_{}.ot".format(i), encoding='utf8')
        octree.writeBinary(direccion_octree) 
    
    def getOccupancyProbs(octree, puntos, dim_arreglo):

        arreglo = np.full((dim_arreglo), 0.5)
        j = 0 
        for i in puntos:
            #Get occupancy probability given a position (x,y,z)
            node = octree.search(i, 0)
            # If the returned value is different than -1, indicates that a node is in that position
            if node is not None:
                try:
                    probability = node.getOccupancy() #Can use getValues(), but extracts the log-odds
                    arreglo[j] = probability 
                except:
                    pass
            j += 1 
        return arreglo  
