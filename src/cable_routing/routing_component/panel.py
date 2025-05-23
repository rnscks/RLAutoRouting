from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse   
from typing import List, Tuple, Set, Optional

from src.cable_routing.pre_process import PreProcessor
from src.cable_routing.routing_component.terminal import Terminal
from src.cable_routing.routing_component.terminal_block import TerminalBlock
from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode
from src.display.entity import Entity   
from src.cable_routing.routing_component.hot_zone import HotZone
from read_file import read_stp_file, read_routing_json, read_panel_bnd


class Panel(Entity):
    def __init__(self,
                 bnd: Bnd_Box,
                 terminal_blocks: List[TerminalBlock],
                 hot_zones: List[HotZone]) -> None:        
        super().__init__()
        self.bnd: Bnd_Box = bnd
        self.grids: Optional[VoxelGrids3D] = None
        self.terminal_blocks: List[TerminalBlock] = terminal_blocks 
        self.hot_zones: List[HotZone] = hot_zones   
    
    
    def init_voxel_grids(self, 
                        resolution: int = 10) -> None:
        terminal_block_bnds: List[Bnd_Box] = [block.bnd for block in self.terminal_blocks]  
        hot_zone_bnds: List[Bnd_Box] = [zone.bnd for zone in self.hot_zones]
        self.grids = PreProcessor().process(
            available_area=self.bnd, 
            terminal_blocks=terminal_block_bnds, 
            hot_zones=hot_zone_bnds,    
            resolution=resolution)
        return

    def init_brep_solid(self) -> None:              
        if len(self.terminal_blocks) == 1:
            self.brep_solid = self.terminal_blocks[0].brep_solid
            return
            
        try:
            fused_shape = BRepAlgoAPI_Fuse(self.terminal_blocks[0].brep_solid, self.terminal_blocks[1].brep_solid).Shape()
            
            for block in self.terminal_blocks[2:]:
                fused_shape = BRepAlgoAPI_Fuse(fused_shape, block.brep_solid).Shape()
            self.set_brep_solid(fused_shape)
            
        except RuntimeError as e:
            print(f"Error during shape fusion: {e}")
            return None
        
    def set_terminals(self, terminals: List[Terminal]) -> None:
        for block in self.terminal_blocks:
            block.set_terminals(terminals)            
    
    @classmethod
    def read_panel(cls, file_name: str) -> 'Panel':  
        routing_json = read_routing_json(file_name)
        
        available_area: Bnd_Box = read_panel_bnd(routing_json['excel_file'])          
        terminal_blocks: List[TerminalBlock] = [TerminalBlock.read_terminal_blocks(file, index=i+1) for i, file in enumerate(routing_json['stp_files'])]
        hot_zones: List[HotZone] = HotZone.read_hot_zones(routing_json['excel_file'])
        terminals: List[Terminal] = Terminal.read_terminals(routing_json['excel_file'])
        for block in terminal_blocks:   
            block.set_terminals(terminals)
        
        panel: Panel = cls(available_area, terminal_blocks, hot_zones)
        return panel   

    def __iter__(self):
        for block in self.terminal_blocks:
            yield block


if __name__ == "__main__":
    from src.cable_routing.pre_process import PreProcessor
    from src.display.scene import Scene

    
    panel: Panel = Panel.read_panel("data/패널_1번_관련_파일_정보.json")      
    panel.init_voxel_grids(resolution=10)
    panel.init_brep_solid()
    scene = Scene() 
    
    for block in panel.terminal_blocks:
        for terminal in block.terminals:
            src_terminal = terminal
            dst_terminal = terminal.dst_terminal
            if dst_terminal is None:
                continue
            dst_terminal.set_visual_properties(color='red', transparency=0.5)
            src_terminal.set_visual_properties(color='blue', transparency=0.5)  
            scene.add_entity(src_terminal)  
            scene.add_entity(dst_terminal)
            
    for node in panel.grids:
        if node.is_obstacle == False:
            scene.add_entity(node)  
            node.set_visual_properties(color='green', transparency=0.99)
    
    scene.add_entity(panel)
    for hot_zone in panel.hot_zones:    
        scene.add_entity(hot_zone)  
        hot_zone.set_visual_properties(color="red", transparency=0.5)
    scene.display()