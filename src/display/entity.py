from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt  
from OCC.Display.SimpleGui import init_display    
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_Color

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional, Union
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop



class Entity(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.brep_solid: Optional[TopoDS_Shape] = None   
        self.transparency: float = 0.0
        self.msg: str = ""  
        self.color: str = "black"
        self.center_pnt: gp_Pnt = gp_Pnt()  
    
    def set_brep_solid(self, brep_solid: TopoDS_Shape) -> None:
        if isinstance(brep_solid, type(None)):
            raise ValueError("brep_solid is None")
        
        self.brep_solid = brep_solid
        return
    
    def display(self) -> None:
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        display.View.SetBgGradientColors(
            Quantity_Color(Quantity_NOC_WHITE),
            Quantity_Color(Quantity_NOC_WHITE),
            2,
            True)
        if self.brep_solid == None: 
            return
        display.DisplayShape(self.brep_solid, update=True, color=self.color, transparency=self.transparency)  
        if self.msg != "":
            display.DisplayMessage(self.center_pnt, self.msg)
        start_display() 
        return  

    def set_visual_properties(self, color: str = "black", transparency: float = 0.0, msg: str = '') -> None:
        self.color = color
        self.transparency = transparency
        self.msg = msg
        return  

    @abstractmethod
    def init_brep_solid(self, *args, **kwargs) -> None:
        pass

class BRepEntity(Entity):
    def __init__(self, brep_solid: Optional[TopoDS_Shape] = None) -> None:
        super().__init__()
        self.brep_solid = brep_solid


    def init_brep_solid(self, brep_solid: TopoDS_Shape) -> None:    
        self.set_brep_solid(brep_solid)
        self._init_center_pnt()
        return

    def _init_center_pnt(self) -> None:
        
        props = GProp_GProps()
        brepgprop.VolumeProperties(self.brep_solid, props)
        center = props.CentreOfMass()
        self.center_pnt = center
        return