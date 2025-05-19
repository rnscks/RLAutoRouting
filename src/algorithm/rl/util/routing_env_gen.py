from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape

from typing import List, Tuple, Union, Dict, Set
import random
from src.cable_routing.routing_component.panel import Panel
from src.cable_routing.routing_component.terminal import Terminal
from src.cable_routing.routing_component.terminal_block import TerminalBlock
from src.cable_routing.routing_component.hot_zone import HotZone
from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode
from src.algorithm.pathfinding.pathfinding import AStar


class TerminalBlockGen:
    # x축 터미널 최대 길이는 360mm, 최소 길이는 180mm   
    def __init__(self,
                available_area: Bnd_Box,
                width_bnd: Tuple[float, float] = (170.0, 180.0), 
                height_bnd: Tuple[float, float] = (46.0, 85.0),
                depth_bnd: Tuple[float, float] = (41.0, 107.0),
                n_sections: int = 2) -> None:
        super().__init__()
        self.available_area: Bnd_Box = available_area
        self.sections: List[Bnd_Box] = []
        self.xlen_bnd: Tuple[float, float] = width_bnd   
        self.ylen_bnd: Tuple[float, float] = height_bnd   
        self.zlen_bnd: Tuple[float, float] = depth_bnd  
        
        x_min, y_min, z_min, x_max, y_max, z_max = self.available_area.Get()    
        y_interval: float = (y_max - y_min) / n_sections
        for i in range(n_sections):
            sperated_bnd_box = Bnd_Box()
            new_y_min = y_min + y_interval * i
            new_y_max = y_min + y_interval * (i + 1)
            sperated_bnd_box.SetGap(0.0)    
            sperated_bnd_box.Update(x_min, new_y_min, z_min, x_max, new_y_max, z_max)   
            self.sections.append(sperated_bnd_box)
    
    
    def _sample_1d_point(self, max_value: float, min_value: float, dist_bnd: Tuple[float, float]) -> Tuple[float, float]:
        while True:
            max_point = random.uniform(min_value, max_value)  
            min_point = random.uniform(min_value, max_value)  
            min_dist, max_dist = dist_bnd
            if abs(max_point - min_point) > min_dist and abs(max_point - min_point) < max_dist:
                return min(max_point, min_point), max(max_point, min_point) 
            
    def _is_out_of_section(self, section: Bnd_Box, terminal_block: Bnd_Box) -> bool:  
        z: float = (section.CornerMax().Z() + section.CornerMin().Z()) / 2
        maxx, maxy, maxz = terminal_block.CornerMax().Coord()
        minx, miny, minz = terminal_block.CornerMin().Coord()   
        
        corners: List[gp_Pnt] = [
            gp_Pnt(maxx, maxy, z), 
            gp_Pnt(minx, miny, z),
            gp_Pnt(minx, maxy, z),
            gp_Pnt(maxx, miny, z)]
        
        for corner in corners:  
            if section.IsOut(corner):
                return True 
        return False    

    def _sample_random_terminal_block(self, 
                                      section: Bnd_Box, 
                                      xlen_bnd: Tuple[float, float], 
                                      ylen_bnd: Tuple[float, float],
                                      zlen_bnd: Tuple[float, float]) -> Bnd_Box:
        # 최대 및 최소 x, y를 결정
        maxx, maxy, maxz = section.CornerMax().Coord()  
        minx, miny, minz = section.CornerMin().Coord()  
        
        while True: 
            corner_minx, corner_maxx  = self._sample_1d_point(maxx, minx, xlen_bnd)
            corner_miny, corner_maxy = self._sample_1d_point(maxy, miny, ylen_bnd)    
            corner_minz, corner_maxz = self._sample_1d_point(maxz, minz, zlen_bnd)
            corner_maxz = min(corner_maxz, self.available_area.CornerMax().Z()) 
            corner_minz = self.available_area.CornerMin().Z()
            
            corner_max = gp_Pnt(corner_maxx, corner_maxy, corner_maxz)
            corner_min = gp_Pnt(corner_minx, corner_miny, corner_minz)
            
            bnd_box: Bnd_Box = Bnd_Box()    
            bnd_box.SetGap(0.0) 
            bnd_box.Update(*corner_min.Coord(), *corner_max.Coord())    
            if not self._is_out_of_section(section, bnd_box):
                return bnd_box
    
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
        
        voxel_grids = VoxelGrids3D(corner_max=voxel_corner_max, corner_min=voxel_corner_min, map_size=resolution)
        for node in voxel_grids:
            if available_area.IsOut(node.position):
                node.is_obstacle = True
        return voxel_grids

    def generate(self) -> List[TerminalBlock]:
        i_section: int = 0  
        terminal_blocks: List[TerminalBlock] = []  
        self.inside_sections: List[Bnd_Box] = []    
        
        while len(terminal_blocks) < len(self.sections):
            corner_min = self.sections[i_section].CornerMin()   
            corner_max = self.sections[i_section].CornerMax()   
            inside_section: Bnd_Box = Bnd_Box()    
            inside_section.SetGap(0.0)
            corner_min.SetY(corner_min.Y() + 50.0)
            corner_max.SetY(corner_max.Y() - 50.0)
            corner_max.SetX(corner_max.X() - 40.0)
            corner_min.SetX(corner_min.X() + 40.0)
            
            inside_section.Update(*corner_min.Coord(), *corner_max.Coord())
            self.inside_sections.append(inside_section)
            terminal_bnd_box = self._sample_random_terminal_block(inside_section, xlen_bnd=self.xlen_bnd, ylen_bnd=self.ylen_bnd, zlen_bnd=self.zlen_bnd)   
            box_shape = BRepPrimAPI_MakeBox(terminal_bnd_box.CornerMin(), terminal_bnd_box.CornerMax()).Shape() 
            terminal_blocks.append(TerminalBlock(box_shape, index=len(terminal_blocks)+1))
            i_section += 1
        
        return terminal_blocks

class HotZoneGen:
    # x축 터미널 최대 길이는 360mm, 최소 길이는 180mm   
    def __init__(self,
                available_area: Bnd_Box,
                terminal_blocks: List[TerminalBlock] = [], 
                width_bnd1d: Tuple[float, float] = (170.0, 180.0), 
                height_bnd1d: Tuple[float, float] = (46.0, 85.0),
                depth_bnd1d: Tuple[float, float] = (41.0, 107.0)) -> None:
        super().__init__()
        self.available_area: Bnd_Box = available_area
        self.sections: List[Bnd_Box] = []
        self.xlen_bnd: Tuple[float, float] = width_bnd1d   
        self.ylen_bnd: Tuple[float, float] = height_bnd1d   
        self.zlen_bnd: Tuple[float, float] = depth_bnd1d  
        self.terminal_blocks: List[TerminalBlock] = terminal_blocks 
    
    
    def _sample_1d_point(self, max_value: float, min_value: float, dist_bnd: Tuple[float, float]) -> Tuple[float, float]:
        while True:
            max_point = random.uniform(min_value, max_value)  
            min_point = random.uniform(min_value, max_value)  
            min_dist, max_dist = dist_bnd
            if abs(max_point - min_point) > min_dist and abs(max_point - min_point) < max_dist:
                return min(max_point, min_point), max(max_point, min_point) 
    
    def _is_out_of_available_area(self, hot_zone: Bnd_Box) -> bool:  
        z: float = (self.available_area.CornerMax().Z() + self.available_area.CornerMin().Z()) / 2
        maxx, maxy, maxz = hot_zone.CornerMax().Coord()
        minx, miny, minz = hot_zone.CornerMin().Coord()   
        
        corners: List[gp_Pnt] = [
            gp_Pnt(maxx, maxy, z), 
            gp_Pnt(minx, miny, z),
            gp_Pnt(minx, maxy, z),
            gp_Pnt(maxx, miny, z)]
        
        for corner in corners:  
            if self.available_area.IsOut(corner):
                return True 
        return False    

    def _sample_random_hot_zone(self, 
                                xlen_bnd: Tuple[float, float], 
                                ylen_bnd: Tuple[float, float],
                                zlen_bnd: Tuple[float, float]) -> Bnd_Box:
        # 최대 및 최소 x, y를 결정
        maxx, maxy, maxz = self.available_area.CornerMax().Coord()  
        minx, miny, minz = self.available_area.CornerMin().Coord()  
        
        while True: 
            corner_minx, corner_maxx  = self._sample_1d_point(maxx, minx, xlen_bnd)
            corner_miny, corner_maxy = self._sample_1d_point(maxy, miny, ylen_bnd)    
            corner_minz, corner_maxz = self._sample_1d_point(maxz, minz, zlen_bnd)
            corner_maxz = min(corner_maxz, self.available_area.CornerMax().Z()) 
            corner_minz = self.available_area.CornerMin().Z()
            
            corner_max = gp_Pnt(corner_maxx, corner_maxy, corner_maxz)
            corner_min = gp_Pnt(corner_minx, corner_miny, corner_minz)
            
            bnd_box: Bnd_Box = Bnd_Box()    
            bnd_box.SetGap(0.0) 
            bnd_box.Update(*corner_min.Coord(), *corner_max.Coord())    
            if not self._is_out_of_available_area(bnd_box):
                return bnd_box
    
    def generate(self, max_dist: float = 50.1, n_hot_zones: int = 1) -> List[TerminalBlock]:
        hot_zones: List[HotZone] = []  
        
        while len(hot_zones) < n_hot_zones:            
            hot_zone_bnd = self._sample_random_hot_zone(xlen_bnd=self.xlen_bnd, ylen_bnd=self.ylen_bnd, zlen_bnd=self.zlen_bnd)   
            hot_zone = HotZone(hot_zone_bnd, max_dist, index=len(hot_zones)+1)
            hot_zone.init_brep_solid()
            hot_zones.append(hot_zone)
        return hot_zones

class TerminalGen:   
    def __init__(self):
        pass
        
    def _sample_terminal_on_bnd_box(self, bnd_box: Bnd_Box, front_gap: float = 10.0) -> Terminal:
        maxx, maxy, maxz = bnd_box.CornerMax().Coord()
        minx, miny, minz = bnd_box.CornerMin().Coord()
        if random.random() > 0.5:
            y: float = maxy
            dir: Tuple[int, int, int] = (0, 1, 0)
        else:   
            y: float = miny 
            dir: Tuple[int, int, int] = (0, -1, 0)  
        
        minx += 15.0
        maxx -= 15.0
        minz += 15.0    
        maxz -= 15.0    
        x: float = random.uniform(minx, maxx)
        z: float = random.uniform(minz, maxz)
        terminal_pnt = gp_Pnt(x, y, z)  
        front_pnt = gp_Pnt(x, y + dir[1]*front_gap, z)
        return Terminal(
            terminal_pnt=terminal_pnt, 
            terminal_dir=dir,
            front_pnt=front_pnt)      
    
    def _create_bnd_box(self, terminal_shape: TopoDS_Shape) -> Bnd_Box:   
        bnd_box = Bnd_Box()
        brepbndlib.Add(terminal_shape, bnd_box)
        return bnd_box  

    def generate(self, 
                 panel: Panel, 
                 n_terminals = 1,
                 len_boundary: Tuple[int, int] = (5, 100),
                 turning_boundary: Tuple[int, int] = (0, 100),
                 front_gap: float = 10.0) -> List[Tuple[Terminal, Terminal]]:
        terminals: List[Terminal] = []
        terminal_blocks: List[TerminalBlock] = panel.terminal_blocks
        grids: VoxelGrids3D = panel.grids
        
        while n_terminals > len(terminals):
            src_terminal_block = random.choice(terminal_blocks)
            dst_terminal_block = random.choice(terminal_blocks)
            
            if src_terminal_block == dst_terminal_block:
                continue
            try:
                src_terminal: Terminal = self._sample_terminal_on_bnd_box(src_terminal_block.bnd, front_gap)   
                src_terminal.index = terminal_blocks.index(src_terminal_block) + 1
                dst_terminal: Terminal = self._sample_terminal_on_bnd_box(dst_terminal_block.bnd, front_gap)
                dst_terminal.index = terminal_blocks.index(dst_terminal_block) + 1
                src_node: VoxelNode = Terminal.search_terminal_in_grid(grids, src_terminal)    
                dst_node: VoxelNode = Terminal.search_terminal_in_grid(grids, dst_terminal)    
            except ValueError: 
                continue
            
            if src_node == dst_node:
                continue
            grids.set_start_node(src_node)
            grids.set_goal_node(dst_node)
            
            if not self._is_valid_path(grids, len_boundary, turning_boundary):
                continue
            
            src_terminal.dst_terminal = dst_terminal
            dst_terminal.dst_terminal = src_terminal
            src_terminal.init_brep_solid(radius=4.0)
            src_terminal.set_visual_properties(color="red", transparency=0.5)
            dst_terminal.init_brep_solid(radius=4.0)
            dst_terminal.set_visual_properties(color="red", transparency=0.5)
            terminals.extend([src_terminal, dst_terminal])
        return terminals
    
    def _count_turning(self, path_nodes: List[VoxelNode]) -> int:
        n_turning = 0
        for idx, node in enumerate(path_nodes[1:-1]):
            i, j, k = node.i, node.j, node.k
            prev_i, prev_j, prev_k = path_nodes[idx - 1].i, path_nodes[idx - 1].j, path_nodes[idx - 1].k
            next_i, next_j, next_k = path_nodes[idx + 1].i, path_nodes[idx + 1].j, path_nodes[idx + 1].k  
            nxt_dir_i, nxt_dir_j, nxt_dir_k = next_i - i, next_j - j, next_k - k    
            prv_dir_i, prv_dir_j, prv_dir_k = i - prev_i, j - prev_j, k - prev_k    
            if nxt_dir_i != prv_dir_i or nxt_dir_j != prv_dir_j or nxt_dir_k != prv_dir_k:
                n_turning += 1
        return n_turning
    
    def _is_valid_path(self, 
                       grids: VoxelGrids3D, 
                       len_boundary: Tuple[int, int], 
                       turning_boundary: Tuple[int, int]) -> bool:
        min_len, max_len = len_boundary
        min_turning, max_turning = turning_boundary 
        
        ast = AStar()
        grids.reset()
        if ast.search(grids):
            path_nodes: List[VoxelNode] = ast.get_path_nodes(grids)   
            n_nodes = len(path_nodes)
            if n_nodes > max_len or n_nodes < min_len:
                return False
            if self._count_turning(path_nodes) > max_turning or self._count_turning(path_nodes) < min_turning:     
                return False
            return True
        return False

class RandomPanelGen:
    def __init__(self) -> None:
        pass
    
    
    def generate(self,
                 available_area: Bnd_Box,
                 n_sections: int,
                 n_terminals: int, 
                 len_boundary: Tuple[int, int] = (5, 100), 
                 turning_boundary: Tuple[int, int] = (0, 100),
                 resolution: int = 10,
                 n_hot_zones: int = 1) -> Panel:
        terminal_blocks: List[TerminalBlock] = TerminalBlockGen(
            available_area=available_area,
            width_bnd=(170.0, 360.0),
            height_bnd=(46.0, 85.0),
            n_sections=n_sections).generate()
        hot_zones: List[HotZone] = HotZoneGen(
            available_area=available_area,
            width_bnd1d=(20.0, 200.0),
            height_bnd1d=(20.0, 200.0),
            terminal_blocks=terminal_blocks).generate(n_hot_zones=n_hot_zones)
        
        panel: Panel = Panel(
            bnd=available_area,
            terminal_blocks=terminal_blocks,
            hot_zones=hot_zones)
        panel.init_voxel_grids(
            resolution=resolution)
        panel.init_brep_solid()
        
        terminals: List[Terminal] = TerminalGen().generate(
            panel, 
            n_terminals,
            len_boundary=len_boundary,
            turning_boundary=turning_boundary)  

        panel.set_terminals(terminals)
        return panel
    
if __name__ == "__main__":
    from src.display.scene import Scene
    from src.display.engine import SimpleDisplayEngine
    
    engine = SimpleDisplayEngine()
    panel: Panel = RandomPanelGen().generate(
        available_area=Bnd_Box(gp_Pnt(-320, -400, 5), gp_Pnt(270, 310, 130)),
        n_sections=3,
        n_terminals=30,
        len_boundary=(5, 100),
        turning_boundary=(0, 100),
        resolution=20)
    
    scene1 = Scene('s1')
    scene2 = Scene('s2')
    scene3 = Scene('s3')
    for block in panel:
        scene1.add_entity(block)
        scene2.add_entity(block)
        scene3.add_entity(block)
        for terminal in block.terminals:
            scene1.add_entity(terminal)   
            scene2.add_entity(terminal)
    
    for hot_zone in panel.hot_zones:    
        scene1.add_entity(hot_zone)
        scene2.add_entity(hot_zone)
        scene3.add_entity(hot_zone)
        hot_zone.set_visual_properties(color="red", transparency=0.5)  
    
    for node in panel.grids:
        if node.is_obstacle == False:
            scene1.add_entity(node)
            node.set_visual_properties(color="green", transparency=0.99)
    engine.add_scene(scene1)
    engine.add_scene(scene2)
    engine.add_scene(scene3)
    
    engine.display()