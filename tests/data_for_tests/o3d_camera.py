import open3d as o3d
from open3d.cpu import pybind

O3D_CAMERA = o3d.cpu.pybind.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    cx=319.50,
    cy=239.50,
    fx=481.20,
    fy=-480.00,
)
