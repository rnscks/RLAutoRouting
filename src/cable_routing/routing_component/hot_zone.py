from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt

import pandas as pd
from typing import List 
from src.display.entity import Entity


class HotZone(Entity):
    def __init__(self,
                bnd: Bnd_Box,
                max_dist: float = 0.0,  
                index: int = 1) -> None:
        super().__init__()  
        self.bnd: Bnd_Box = bnd
        self.max_dist: float = max_dist 
        self.index: int = index
    
    
    def init_brep_solid(self):
        brep_solid: TopoDS_Shape = BRepPrimAPI_MakeBox(self.bnd.CornerMin(), self.bnd.CornerMax()).Shape()
        self.set_brep_solid(brep_solid)
        self.set_visual_properties(color="red", transparency=0.5)   
        return
    
    def get_center(self) -> gp_Pnt:
        center_x = (self.bnd.CornerMin().X() + self.bnd.CornerMax().X()) / 2
        center_y = (self.bnd.CornerMin().Y() + self.bnd.CornerMax().Y()) / 2
        center_z = (self.bnd.CornerMin().Z() + self.bnd.CornerMax().Z()) / 2
        return gp_Pnt(center_x, center_y, center_z) 
    
    @classmethod
    def read_hot_zones(cls, file_name: str) -> List['HotZone']:
        try:
            xl = pd.ExcelFile(file_name)
            if "HotZone" not in xl.sheet_names:
                return []
        except:
            return []
        hot_zone_df = pd.read_excel(file_name, sheet_name="HotZone")
        corner_mins = [gp_Pnt(x, y, z) for x, y, z in hot_zone_df[['X1', 'Y1', 'Z1']].values] 
        corner_maxs = [gp_Pnt(x, y, z) for x, y, z in hot_zone_df[['X2', 'Y2', 'Z2']].values] 
        max_dists = hot_zone_df['MaxDist'].values   
        hot_zones: List[HotZone] = []
        for i in range(len(corner_mins)):
            bnd: Bnd_Box = Bnd_Box(corner_mins[i], corner_maxs[i])
            hot_zone = cls(bnd, max_dist=max_dists[i], index=i+1)
            hot_zone.init_brep_solid()
            hot_zones.append(hot_zone)
        return hot_zones
    
    
if __name__ == "__main__":
    from src.display.scene import Scene
    
    
    scene = Scene()
    hot_zones = HotZone.read_hot_zones("data/라우팅_데이터/1번_패널/1번_패널_라우팅_정보.xlsx")
    for hot_zone in hot_zones:
        hot_zone.init_brep_solid()  
        scene.add_entity(hot_zone)
    
    hot_zone.display()