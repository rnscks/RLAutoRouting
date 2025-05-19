from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

from typing import List, Tuple, Set, Optional
import pandas as pd

from src.datastruct.voxel_grids import VoxelNode, VoxelGrids3D
from src.display.entity import Entity

class Terminal(Entity):
    def __init__(self,
                terminal_pnt: gp_Pnt,
                front_pnt: gp_Pnt,
                terminal_dir: Tuple[int, int, int],
                index: int = 0) -> None:
        super().__init__()
        self.terminal_pnt: gp_Pnt = terminal_pnt
        self.front_pnt: gp_Pnt = front_pnt
        self.terminal_dir: Tuple[int, int, int] = terminal_dir
        self.index: int = index
        self.dst_terminal: Optional['Terminal'] = None
        self.is_routed: bool = False
    
    
    def init_brep_solid(self, radius: float = 5.0) -> None:
        self.set_brep_solid(BRepPrimAPI_MakeSphere(self.terminal_pnt, radius).Shape())  
        return 
    
    @classmethod
    def search_terminal_in_grid(cls,
                            grids: VoxelGrids3D,
                            terminal: 'Terminal') -> VoxelNode:
        terminal_pnt = terminal.terminal_pnt    
        terminal_dir = terminal.terminal_dir
        node: Optional[VoxelNode] = grids[0, 0, 0]
        
        for other in grids:
            if other.center_pnt.Distance(terminal_pnt) < node.center_pnt.Distance(terminal_pnt):
                node = other
        if node.is_obstacle == False:
            return node
        i,j,k = node.i, node.j, node.k
        dirx, diry, dirz = terminal_dir
        map_size = grids.map_size   
        
        while True:
            i, j, k = i + dirx, j + diry, k + dirz
            if i < 0 or i >= map_size or j < 0 or j >= map_size or k < 0 or k >= map_size:  
                raise ValueError("Terminal is out of range")    
            node = grids[i, j, k]
            if not node.is_obstacle:
                return node
    
    @classmethod    
    def read_terminals(self,
                file_name: str,
                front_gap: float = 10.0,
                radius: float = 4.0) -> List['Terminal']:
        
        terminal_df = pd.read_excel(file_name, sheet_name="Terminal")
        io_df = pd.read_excel(file_name, sheet_name="IO")

        terminals: List[Terminal] = []
        io_table: List[Tuple[int, int]] = [(int(input), int(output)) for input, output in io_df[['IN', 'OUT']].values]

        socket_pnts: List[gp_Pnt] = [gp_Pnt(x, y, z) for x, y, z in terminal_df[['X', 'Y', 'Z']].values]   
        socket_dirs: List[Tuple[int, int, int]] = [(dirx, diry, dirz) for dirx, diry, dirz in terminal_df[['DIRX', 'DIRY', 'DIRZ']].values]
        socket_index: List[int] = [int(index) for index in terminal_df['INDEX'].values] 
        
        front_pnts: List[gp_Pnt] = []
        for pnt, dir in zip(socket_pnts, socket_dirs):  
            front_pnts.append(gp_Pnt(pnt.X() + dir[0] * front_gap, pnt.Y() + dir[1] * front_gap, pnt.Z() + dir[2] * front_gap))    
        
        for i in range(len(socket_pnts)): 
            terminal = Terminal(
                terminal_pnt=socket_pnts[i],
                terminal_dir=socket_dirs[i],
                front_pnt=front_pnts[i],  
                index=socket_index[i])
            terminal.init_brep_solid(radius=radius)
            terminal.color = 'red'
            terminals.append(terminal) 
        
        for input, output in io_table: 
            in_terminal = terminals[input]
            out_terminal = terminals[output]
            
            in_terminal.dst_terminal = out_terminal
            out_terminal.dst_terminal = in_terminal 
        return terminals
    
    def __hash__(self):
        return hash(self.terminal_pnt)  
    
    def __eq__(self, other: 'Terminal') -> bool:
        if self.terminal_pnt.Distance(other.terminal_pnt) < 0.01:
            return True 
        return False    



if __name__ == "__main__":  
    from src.display.scene import Scene
    
    terminals: List[Terminal] =Terminal.read_terminals(
        'data/라우팅_데이터/1번_패널/1번_패널_라우팅_정보.xlsx')
    
    scene = Scene() 
    scene.add_entities(terminals)
    scene.display()