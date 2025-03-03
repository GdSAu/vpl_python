# Repositorio para View planning 
En este repo buscaremos implementar VPL, con python, open3d y varias implementaciones de planificadores :D

This implementation contains objects from the [**Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items**](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/), this dataset is composed of 1030 objects, and provides the following data:
- Object:
    + model.sdf
    + model.config
    + metadata.pbtxt
    + meshes
        - model.mtl
        - model.obj
    + materials
        - textures/textures.png
    + thumbnails
        - 0 ... 4 .jpg

We put this info into a folder named **Model**. 
Our idea is to use these objects for 3D reconstructions via view planning methods. So to achive that, a couple of folders are added which are the ones that will contains the information requiered for the view planning processes.
So the dataset is ordered as follows:
- Object
    + Model
    + Depth
    + Octree
    + Point_cloud
        - Maximization/It#
        - Object_Point_Cloud
    + RGB



## Environment setup


    ```
	conda env create -f vpl_env.yml
	conda activate vpl
	```