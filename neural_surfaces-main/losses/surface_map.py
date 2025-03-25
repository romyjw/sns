
from .differential import SurfaceMapLoss
from .mixin import Loss

class IsometricSurfaceMapLoss(Loss, SurfaceMapLoss):

    def __init__(self, **kwargs):
        Loss.__init__(self, **kwargs)
        SurfaceMapLoss.__init__(self)

    def map_distortion(self, FFF):
        return self.symmetric_dirichlet(FFF)


class ARAPSurfaceMapLoss(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.arap(FFF)


class ConformalSurfaceMapLoss1(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.conformal1(FFF)


class EquiarealSurfaceMapLoss(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.equiareal(FFF)
        
class ConformalSurfaceMapLoss2(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.conformal_AMIPS_2D(FFF)
