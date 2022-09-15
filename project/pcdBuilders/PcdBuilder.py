from project.annotators.Annotator import *
from project.Camera import Camera
from project.dto.Pcd import Pcd


class PcdBuilder(ABC):
    """
    Loads point cloud from main data file and Annotator object
    :attribute cam: pin hole camera parameters
    :attribute _annot: annotator object to extract planes
    """

    def __init__(self, camera: Camera, annot):
        self.cam = camera
        self._annot = annot

    @abstractmethod
    def _get_points(self, path_depth_image) -> Pcd:
        """
        :param path_depth_image: path to a main data file
        :return: pcd object
        """
        pass

    def build_pcd(self, path_depth_image, pcd_num, verbose):
        """
        :param path_depth_image: path to a main data file
        :param pcd_num: number of the file
        :return: object pcd
        """
        pcd = self._get_points(path_depth_image)
        pcd = self._annot.annotate(pcd, pcd_num)

        if verbose:
            print(path_depth_image)

        return pcd
