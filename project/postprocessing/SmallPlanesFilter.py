from project.dto.Pcd import Pcd


class SmallPlanesFilter:
    @staticmethod
    def filter(pcd: Pcd) -> Pcd:
        result = Pcd(pcd.points)
        for plane in pcd.planes:
            if len(plane.plane_indices) < 1000:
                continue
            result.planes.append(plane)
        return result
