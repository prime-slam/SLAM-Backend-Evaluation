from annotators.Annotator import *
from Camera import Camera
from Pcd import Pcd


class PcdBuilder(ABC):
    def __init__(self, camera: Camera, annot=None):
        self.cam = camera
        self.annot = annot

    # @abstractmethod
    # def __get_points(self, image_number: int, array_file_names: list[str]) -> Pcd:
    #     pass

    def build_pcd(self, image_number: int, array_file_names_depth: list[str]):
        pcd = self.__get_points(image_number, array_file_names_depth)
        pcd = self.annot.annotate(pcd, image_number)

        return pcd
