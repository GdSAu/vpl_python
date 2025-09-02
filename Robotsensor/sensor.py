import open3d as o3d
import numpy as np
import cv2
from utils.utils_o3d import Nube_acumulada_filtrada


class Sensor:
    '''
    This class extract depth, RGB images, and pointclouds 
    '''

    def __init__(self, fov, up,  width, height, **kwargs):
        self.fov = fov
        self.up = up
        self.width = width
        self.height = height
        self.only_raycasti = kwargs["raycast"]
        if self.only_raycasti == True:
            self.scene = kwargs["scene"]
        else:
            self.render = kwargs["render"]
            self.scene = kwargs["scene"]


    def __extract_RGBD(self, center, eye):
        """(render,fov,center,eye,up) -> RGB-D"""
        self.render.setup_camera(self.fov, center, eye, self.up)
        img = self.render.render_to_image()
        depth = self.render.render_to_depth_image()
        return img, depth
    
    def saveRGBD(self, center, eye, rgb_file_name, depth_file_name):
        """Save RGBD images"""
        img, depth= self.__extract_RGBD(center,eye)
        o3d.io.write_image(rgb_file_name, img, 9)
        cv2.imwrite(depth_file_name, np.asarray(depth))


    def getRGBD(self, center, eye):
        """return RGBD images"""
        img, depth = self.__extract_RGBD(center,eye)
        return img, depth

    def __extract_PointCloud(self, center, eye):
        """(scene, fov, center, eye, up, width, height) -> point cloud"""
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg = self.fov,
            center = center,
            eye = eye,
            up = self.up,
            width_px = self.width,
            height_px = self.height,
        )
        # We can directly pass the rays tensor to the cast_rays function.
        ans = self.scene.cast_rays(rays)
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        return points
    
    def savePointCloud(self, center, eye, file_name):
        """Save point cloud"""
        if self.only_raycasti == True:
            points = self.scene.create_visible_pointcloud(center, eye, step_factor=2)
        else:
            points = self.__extract_PointCloud(center, eye)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        o3d.io.write_point_cloud(file_name, pcd, write_ascii=True)

    def saveAccPointCloud(self, i, direction, file_name):
        if i == 0:
            Nube_acc = p_c = o3d.io.read_point_cloud(file_name.format(i))
        else:
            p_c = o3d.io.read_point_cloud(direction + "cloud_acc.pcd")
            pcd = o3d.io.read_point_cloud(file_name.format(i))
            Nube_acc = o3d.geometry.PointCloud()
            Nube_acc.points = o3d.utility.Vector3dVector(Nube_acumulada_filtrada(p_c,pcd, voxel_size=0.0035).points)
            #print(Nube_acc)
        
        o3d.io.write_point_cloud(direction + "cloud_acc.pcd", Nube_acc, write_ascii=True)# accumulated cloud


    def getPointcloud(self,center, eye):
        """return a point cloud object"""
        if self.only_raycasti == True:
            points = self.scene.create_visible_pointcloud(center, eye, step_factor=2)
        else:
            points = self.__extract_PointCloud(center, eye)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.numpy())
        return pc


        
        