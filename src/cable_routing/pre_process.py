from OCC.Core.gp import gp_Pnt  
from OCC.Core.Bnd import Bnd_Box

from typing import List, Optional
from abc import ABC, abstractmethod 

from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode


class PreProcessor:
    def __init__(self) -> None:
        pass
    
    
    def process(self, 
                available_area: Bnd_Box, 
                terminal_blocks: List[Bnd_Box], 
                hot_zones: Optional[List[Bnd_Box]] = None,  
                resolution: int = 10) -> VoxelGrids3D:
        grids: VoxelGrids3D =  self._create_voxel_grids(available_area, resolution)    
        
        for bnd_box in terminal_blocks:
            maxx, maxy, maxz = bnd_box.CornerMax().Coord()
            minx, miny, minz = bnd_box.CornerMin().Coord()
            for node in grids:
                if (maxx > node.center_pnt.X() > minx) and \
                    (maxy > node.center_pnt.Y() > miny):
                    node.is_obstacle = True
        if hot_zones is not None:   
            for bnd_box in hot_zones:
                maxx, maxy, maxz = bnd_box.CornerMax().Coord()
                minx, miny, minz = bnd_box.CornerMin().Coord()
                for node in grids:
                    if (maxx > node.center_pnt.X() > minx) and \
                        (maxy > node.center_pnt.Y() > miny):
                        node.is_hot_zone = True
        return grids    
    
    def _create_voxel_grids(self, available_area: Bnd_Box, resolution: int) -> VoxelGrids3D:
        corner_max = available_area.CornerMax()   
        corner_min = available_area.CornerMin()  
        
        x_gap = corner_max.X() - corner_min.X() 
        y_gap = corner_max.Y() - corner_min.Y() 
        z_gap = corner_max.Z() - corner_min.Z() 
        
        gap = max(x_gap, y_gap, z_gap)
        voxel_corner_max = gp_Pnt(corner_max.X(), corner_max.Y(), corner_max.Z())
        voxel_corner_min = gp_Pnt(corner_max.X() - gap, 
                                corner_max.Y() - gap, 
                                corner_max.Z() - gap)
        voxel_grids: VoxelGrids3D = VoxelGrids3D(
            corner_max=voxel_corner_max, 
            corner_min=voxel_corner_min, 
            map_size=resolution) 
        for node in voxel_grids:
            if available_area.IsOut(node.position):
                node.is_obstacle = True
        return voxel_grids