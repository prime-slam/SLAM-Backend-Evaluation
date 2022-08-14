from project import read_office
from project.Camera import Camera
from project.dto.Pcd import Pcd
from project.annotators.AnnotatorImage import AnnotatorImage
from project.pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderOffice(PcdBuilder):
    """
    Builds pcd from .depth file, uses pinhole camera parameters
    """

    def __init__(
        self,
        camera: Camera,
        annot: AnnotatorImage,
    ):
        super().__init__(camera, annot)

    def _get_points(self, depth_image_path):
        points_of_image = read_office.get_points(depth_image_path, self.cam)
        return Pcd(points_of_image)
