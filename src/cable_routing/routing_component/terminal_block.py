from typing import List, Optional, Set

from OCC.Core.gp import gp_Pnt  
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox    
from OCC.Core.TopoDS import TopoDS_Shape

from src.cable_routing.routing_component.terminal import Terminal
from src.display.entity import Entity
from read_file import read_stp_file

class TerminalBlock(Entity):
    def __init__(self,
                brep_solid: TopoDS_Shape,
                terminals: List[Terminal] = [],
                index: int = 1) -> None:
        super().__init__()
        self.bnd = Bnd_Box()
        brepbndlib.Add(brep_solid, self.bnd)
        self.bnd.Enlarge(10.0)  
        self.set_brep_solid(brep_solid)
        self.index: int = index
        self.terminals: List[Terminal] = []
        
        self.color = "black"
        self.center_pnt = gp_Pnt()  
        self.center_pnt.SetX(
            (self.bnd.CornerMin().X() + self.bnd.CornerMax().X()) / 2.0)
        self.center_pnt.SetY(   
            (self.bnd.CornerMin().Y() + self.bnd.CornerMax().Y()) / 2.0)
        self.center_pnt.SetZ(
            (self.bnd.CornerMin().Z() + self.bnd.CornerMax().Z()) / 2.0)
    
        for terminal in terminals:
            if terminal.index == index and terminal not in self.terminals:
                self.terminals.append(terminal) 

    
    def init_brep_solid(self, brep_solid: TopoDS_Shape) -> None:
        self.bnd = Bnd_Box()
        brepbndlib.Add(brep_solid, self.bnd)
        self.bnd.Enlarge(10.0)
        self.set_brep_solid(brep_solid)
        return
    
    def set_terminals(self, terminals: List[Terminal]) -> None:
        for terminal in terminals:
            if terminal.index == self.index and terminal not in self.terminals:
                self.terminals.append(terminal) 
        return  
    
    @classmethod
    def read_terminal_blocks(cls, stp_file_name: str, index: int = 1) -> 'TerminalBlock':
        terminal_block = cls(brep_solid=read_stp_file(stp_file_name), index=index)
        return terminal_block
    
    def __iter__(self):
        for terminal in self.terminals:
            yield terminal  


if __name__ == "__main__":
    from src.display.entity import BRepEntity
    from src.display.scene import Scene
    
    terminal_block = TerminalBlock.read_terminal_blocks(
        stp_file_name='data/라우팅_데이터/1번_패널/1번_레일(1번_패널).stp',
        index=1)
    terminal_block.set_visual_properties(color="black", transparency=0.2, msg="Terminal Block")
    
    terminals = Terminal.read_terminals(
        file_name='data/라우팅_데이터/1번_패널/1번_패널_라우팅_정보.xlsx',
        front_gap=10.0,
        radius=4.0)
    terminal_block.set_terminals(terminals)
    
    bnd_box = BRepPrimAPI_MakeBox(terminal_block.bnd.CornerMin(), terminal_block.bnd.CornerMax()).Shape()   
    bnd_entity = BRepEntity(bnd_box)
    bnd_entity.set_visual_properties(color="black", transparency=0.5)
    
    scene = Scene() 
    scene.add_entities([terminal_block, bnd_entity])
    scene.add_entities(terminal_block.terminals)
    scene.display()