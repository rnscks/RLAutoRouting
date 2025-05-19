from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.STEPControl import STEPControl_Reader 
from OCC.Core.IFSelect import IFSelect_RetDone  

from typing import Dict
import json
import pandas as pd 


def read_stp_file(file_path: str) -> TopoDS_Shape:
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        shape = step_reader.Shape()
        
        return shape
    else:
        raise ValueError(f"Error reading STEP file: {file_path}")
    
def read_panel_bnd(file_name: str = "data/routing_info/1번_패널/1번 패널_라우팅_정보.xlsx") -> Bnd_Box:
    df = pd.read_excel(file_name, sheet_name='Panel', dtype=float) 
    
    min_x, min_y, min_z = df[["MINX", "MINY", "MINZ"]].values[0]
    max_x, max_y, max_z = df[["MAXX", "MAXY", "MAXZ"]].values[0]
    corner_min = gp_Pnt(min_x, min_y, min_z)
    corner_max = gp_Pnt(max_x, max_y, max_z)
    
    available_area = Bnd_Box()
    available_area.SetGap(0.0)  
    available_area.Update(*corner_min.Coord(), *corner_max.Coord())   
    return available_area

def read_routing_json(file_name: str = "data/routing_info/1번_패널/1번 패널_라우팅_정보.json") -> Dict:
    with open(file_name, mode="r", encoding='utf-8') as f:
        routing_info = json.load(f)
    return routing_info 
