from annotators.Annotator import *
from Camera import Camera
from dto.Pcd import Pcd


class PcdBuilder(ABC):
    def __init__(self, camera: Camera, annot):
        self.cam = camera
        self._annot = annot

    @abstractmethod
    def _get_points(self, path_depth_image, pcd_num) -> Pcd:
        pass

    def build_pcd(self, path_depth_image, pcd_num):
        pcd = self._get_points(path_depth_image, pcd_num)
        pcd = self._annot.annotate(pcd, pcd_num)

        print(path_depth_image)

        return pcd
