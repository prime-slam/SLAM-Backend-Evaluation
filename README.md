# SLAMBackendAlgos

A Project for benchmarking existing SLAM algorithms on planes dataset

# Implemented classes
### Abstract class PcdBuilder:
loads point cloud from main data file and Annotator object

### Subclasses:

PcdBuilderPointCloud  - builds pcd from .pcd and .npy files

PcdBuilderOffice - builds pcd from .depth file, uses pinhole camera parameters

PcdBuilderLiving - builds pcd from depth image, uses pinhole camera parameters

### Abstract class Annotator: 
extracts planes from annotation

### Subclasses:

AnnotatorImage - extracts planes from annotation in rgb format

AnnotatorPointCloud - extracts planes from annotation in .npy format

### Abstract class Associator: 
gets correct indices for planes of each image

### Subclasses:


AssociatorAnnot - associates planes with annotation 

AssociatorFront - associates planes with associate_front function with frontend data  

### class PostProcessing:
chooses planes with maximum points

### class SLAMGraph:
builds and estimtates the SLAM graph

### class MeasureError:
evaluates ate and rpe errors with TUM scripts for evaluating

### class Visualisation:
visualises work of the algorithm as a PointCloud object of open3d library



