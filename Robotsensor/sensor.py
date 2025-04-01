import open3d as o3d
import numpy as np
import cv2


class Sensor:
    '''
    This class extract depth, RGB images, and pointclouds 
    '''

    def __init__(self, fov, up, itera,  width, height, *kwargs):
        self.fov = fov
        self.up = up
        self.itera = itera
        self.width = width
        self.height = height
        self.render = kwargs["render"]
        self.scene = kwargs["scene"]


    def __extract_RGBD(self, center, eye):
        """(render,fov,center,eye,up) -> RGB-D"""
        self.render.setup_camera(self.fov, center, eye, self.up)
        img = self.render.render_to_image()
        depth = self.render.render_to_depth_image()
        return img, depth
    
    def saveRGBD(self, rgb_file_name, depth_file_name, center, eye):
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
    
    def savePointCloud(self, file_name, center, eye):
        """Save point cloud"""
        points = self.__extract_PointCloud(center, eye)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        o3d.io.write_point_cloud(file_name, pcd, write_ascii=True)


    def getPointcloud(self,center, eye):
        """return a point cloud object"""
        points = self.__extract_PointCloud(center, eye)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.numpy())
        return pc


        
        