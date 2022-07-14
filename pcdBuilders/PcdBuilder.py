from annotators.Annotator import *
from Camera import Camera
from Pcd import Pcd


class PcdBuilder(ABC):
    def __init__(self, camera: Camera, annot=None):
        self.cam = camera
        self.annot = annot

    @abstractmethod
    def _get_points(self, image_number: int, array_file_names: List[str]) -> Pcd:
        pass

    def build_pcd(self, image_number: int, array_file_names_depth: List[str]):
        pcd = self._get_points(image_number, array_file_names_depth)
        pcd = self.annot.annotate(pcd, image_number)

        print(array_file_names_depth[image_number])

        return pcd
