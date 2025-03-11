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

    def Get_RGBD(self, direccion, center, eye,i):
        ''' (render,fov,center,eye,up) -> RGB-D
            render : es el objeto de escena que contiene el objeto
            fov : vertical_field of view
            center: camera center (orientation, where the camera see)
            eye: camera eye (position)
            up: camera up vector ()
            direccion: root folder direction
            i = index in loop
            '''
        self.render.setup_camera(self.fov, center, eye, self.up)
        img = self.render.render_to_image()
        depth = self.render.render_to_depth_image()
        o3d.io.write_image(direccion +"RGB/"+ self.itera +"RGB_{}.png".format(i), img, 9)
        cv2.imwrite(direccion + "Depth/"+ self.itera +"D_{}.tiff".format(i), np.asarray(depth))


    def Get_Pointcloud(self, direccion, center, eye,i):
        '''scene: scene object which contains the mesh
            fov: vertical field of view
            center: camera center (orientation, where the camera see)
            eye: camera eye (position)
            up: camera up vector ()
            width, height : width and height of the image 
            direccion: root folder direction
            i = index in loop
        '''

        # (scene, fov, center, eye, up, width, height) -> point cloud
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        o3d.io.write_point_cloud(direccion + "/Point_cloud/"+ self.itera +"cloud_{}.pcd".format(i), pcd, write_ascii=True)# cloud in t-time