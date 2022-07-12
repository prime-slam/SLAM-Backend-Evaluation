class Camera(object):
    def __init__(self, width, height, cx, cy, focal_x, focal_y, scale):
        self.width = width
        self.height = height
        self.cx = cx
        self.cy = cy
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.scale = scale
