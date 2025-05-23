from OCC.Core.gp import gp_Vec  
from OCC.Core.Bnd import Bnd_Box    
import torch

from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import numpy as np

from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode
from src.cable_routing.routing_component.panel import Panel


class Observation(ABC):
    def __init__(self) -> None:
        self.obs_dims: int = 0  
        self.obs_name: str = "Observation"  
        pass
    
    @abstractmethod
    def get_observation(self, cur_node: VoxelNode) -> np.ndarray:   
        pass    
    
    @abstractmethod
    def set_panel(self, panel: Panel) -> None:
        pass

class SensorObservation(Observation):
    def __init__(self, panel: Optional[Panel] = None) -> None:
        super().__init__()
        self.obs_dims = 56
        self.obs_name: str = "SensorObservation"
        self.panel = panel
        self.dir_table = {
            (1, 0, 0),
            (-1, 0, 0), 
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (-1, 0, 1),
            (-1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
            (0, -1, 1),
            (0, -1, -1),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1)
        }
        self.max_sensing_range: int = 10

    def set_panel(self, panel: Panel) -> None:   
        self.panel = panel
        return

    def is_valid_node(self, 
                      src_node: VoxelNode, 
                      dir_i: int, dir_j: int, dir_k: int, 
                      grids: VoxelGrids3D) -> bool:     
        map_size: int = grids.map_size
        nxt_i, nxt_j, nxt_k = src_node.i + dir_i, src_node.j + dir_j, src_node.k + dir_k
        
        if  0 <= nxt_i < map_size and \
            0 <= nxt_j < map_size and \
            0 <= nxt_k < map_size:
            if grids[nxt_i, nxt_j, nxt_k].is_obstacle:
                return False
            else:
                return True
        return False
    
    def sensing_forward(self, 
                        src_node: VoxelNode, 
                        dir_i: int, dir_j: int, dir_k: int,
                        grids: VoxelGrids3D) -> Tuple[float, float]: 
        node_state: float = 1.0
        cur_node: VoxelNode = src_node
        n_steps = 0
        
        while self.is_valid_node(cur_node, dir_i, dir_j, dir_k, grids) and n_steps < self.max_sensing_range:   
            nxt_i, nxt_j, nxt_k = cur_node.i + dir_i, cur_node.j + dir_j, cur_node.k + dir_k    
            cur_node = grids.nodes_map[nxt_i][nxt_j][nxt_k]    
            n_steps += 1
            if cur_node.is_obstacle:
                node_state = 0.0
                break
            if cur_node.is_goal_node:   
                node_state = 0.5
                break
            if cur_node.is_hot_zone:
                node_state = 0.2
                break   
        
        relative_distance: float = n_steps / self.max_sensing_range
        return relative_distance, node_state

    def get_observation(self, cur_node: VoxelNode) -> np.ndarray:
        grids: VoxelGrids3D = self.panel.grids
        observation: np.ndarray = np.zeros(self.obs_dims)
        goal_node = grids.goal_node
        for idx, dir in enumerate(self.dir_table):
            observation[idx], observation[idx+len(self.dir_table)] = self.sensing_forward(cur_node, *dir, grids)
        start_to_dst  = grids.start_node.center_pnt.Distance(grids.goal_node.center_pnt)  
        cur_to_dst = grids.start_node.center_pnt.Distance(cur_node.center_pnt)
        to_dst_feature = cur_to_dst / (start_to_dst + 1e6)
        
        observation[52] = np.clip(to_dst_feature, 0, 1)
        magnitude = goal_node.center_pnt.Distance(cur_node.center_pnt)
        if magnitude == 0:
            return observation
        dx = (goal_node.center_pnt.X() - cur_node.center_pnt.X())/magnitude
        dy = (goal_node.center_pnt.Y() - cur_node.center_pnt.Y())/magnitude
        dz = (goal_node.center_pnt.Z() - cur_node.center_pnt.Z())/magnitude 

        observation[53], observation[54], observation[55] = dx, dy, dz
        return observation

class ActionObservation(Observation):
    def __init__(self) -> None:
        super().__init__()
        self.obs_dims = 6
        self.obs_name: str = "ActionObservation"    
        pass
    
    def set_panel(self, panel: Panel):
        return
        
    def get_observation(self, cur_node: VoxelNode) -> np.ndarray:
        observation = np.zeros(self.obs_dims)
        parnet_node: VoxelNode = cur_node.parent
        
        if parnet_node != None: 
            gparnet_node: VoxelNode = parnet_node.parent    
        else:
            gparnet_node = None
        
        nodes: List[VoxelNode] = [gparnet_node, parnet_node, cur_node]  
        for idx in range(len(nodes) - 1):
            if nodes[idx] == None:
                observation[idx*3] = 0
                observation[idx*3+1] = 0    
                observation[idx*3+2] = 0    
                continue
            
            vec = gp_Vec(nodes[idx].center_pnt, nodes[idx+1].center_pnt)    
            vec.Normalize() 
            observation[idx*3] = vec.X()    
            observation[idx*3+1] = vec.Y()  
            observation[idx*3+2] = vec.Z()  
        return observation

class PanelConfigObservation(Observation):
    def __init__(self, panel: Optional[Panel] = None) -> None:
        super().__init__()
        self.obs_dims = 45
        self.obs_name: str = "PanelConfigObservation"
        self.panel: Optional[Panel] = panel
        
    def set_panel(self, panel: Panel) -> None:
        self.panel =  panel
        return
    
    def get_observation(self, cur_node: VoxelNode) -> np.ndarray:    
        corner_max = self.panel.bnd.CornerMax()
        corner_min = self.panel.bnd.CornerMin()
        bnd_boxes: List[Bnd_Box] = []
        hot_zone_bnd_boxes: List[Bnd_Box] = []  
        
        for block in self.panel:
            bnd_boxes.append(block.bnd)

        for hot_zone in self.panel.hot_zones:
            hot_zone_bnd_boxes.append(hot_zone.bnd) 
        
        grids = self.panel.grids
        
        # terminal block observation(3 * 6)
        terminal_observation = np.zeros(3 * 6)
        for idx, bnd in enumerate(bnd_boxes):
            maxx, maxy, maxz = bnd.CornerMax().Coord()
            minx, miny, minz = bnd.CornerMin().Coord()
            maxz = min(maxz, corner_max.Z())   
            minz = max(minz, corner_min.Z())   
            
            terminal_observation[idx] = (maxx - corner_min.X()) / (corner_max.X() - corner_min.X()) 
            terminal_observation[idx + 1] = (maxy - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
            terminal_observation[idx + 2] = (maxz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
            
            terminal_observation[idx * 3] = (minx - corner_min.X()) / (corner_max.X() - corner_min.X())
            terminal_observation[idx * 3 + 1] = (miny - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
            terminal_observation[idx * 3 + 2] = (minz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
            
        
        # hot zone observation(3 * 6)
        hot_zone_observation = np.zeros(3 * 6)
        for idx, bnd in enumerate(hot_zone_bnd_boxes):
            maxx, maxy, maxz = bnd.CornerMax().Coord()
            minx, miny, minz = bnd.CornerMin().Coord()  
            
            maxx = min(maxx, corner_max.X())
            minx = max(minx, corner_min.X())
            
            maxy = min(maxy, corner_max.Y())
            miny = max(miny, corner_min.Y())
            
            maxz = min(maxz, corner_max.Z())    
            minz = max(minz, corner_min.Z())    
            
            hot_zone_observation[idx] = (maxx - corner_min.X()) / (corner_max.X() - corner_min.X())
            hot_zone_observation[idx + 1] = (maxy - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
            hot_zone_observation[idx + 2] = (maxz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
            hot_zone_observation[idx * 3] = (minx - corner_min.X()) / (corner_max.X() - corner_min.X())
            hot_zone_observation[idx * 3 + 1] = (miny - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
            hot_zone_observation[idx * 3 + 2] = (minz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
        
        # start point observation
        start_observation = np.zeros(3)
        startx, starty, startz = grids.start_node.center_pnt.Coord()   
        start_observation[0] = (startx - corner_min.X()) / (corner_max.X() - corner_min.X())
        start_observation[1] = (starty - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
        start_observation[2] = (startz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
        
        # goal point observation
        goal_observation = np.zeros(3)
        goalx, goaly, goalz = grids.goal_node.center_pnt.Coord()   
        goal_observation[0] = (goalx - corner_min.X()) / (corner_max.X() - corner_min.X())
        goal_observation[1] = (goaly - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
        goal_observation[2] = (goalz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())        
        
        # current point observation 
        current_observation = np.zeros(3)   
        curx, cury, curz = cur_node.center_pnt.Coord()  
        current_observation[0] = (curx - corner_min.X()) / (corner_max.X() - corner_min.X())
        current_observation[1] = (cury - corner_min.Y()) / (corner_max.Y() - corner_min.Y())
        current_observation[2] = (curz - corner_min.Z()) / (corner_max.Z() - corner_min.Z())
        
        observation = np.concatenate((terminal_observation, hot_zone_observation, start_observation, goal_observation, current_observation), axis=0)  
        return observation

class VoxelObservation(Observation):
    def __init__(self, panel: Optional[Panel] = None) -> None:
        super().__init__()
        self.obs_dims = 81000
        self.obs_name: str = "VoxelObservation"
        self.panel: Optional[Panel] = panel
        
    def set_panel(self, panel: Panel) -> None:
        self.panel =  panel
        return
    
    def get_observation(self, cur_node: VoxelNode) -> np.ndarray:    
        grids = self.panel.grids
        observation = np.zeros((3, 30, 30, 30))
        for i in range(grids.map_size): 
            for j in range(grids.map_size):
                for k in range(grids.map_size):
                    node = grids[i, j, k]
                    if node.is_obstacle:
                        observation[0, i, j, k] = 1.0
                    elif node.is_goal_node:
                        observation[1, i, j, k] = 1.0
                    elif node == cur_node:    
                        observation[2, i, j, k] = 1.0  
        observation = observation.flatten()
        return observation

