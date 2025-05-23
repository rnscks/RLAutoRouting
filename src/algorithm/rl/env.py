from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Vec
from OCC.Core.TopoDS import TopoDS_Shape    
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Set, Dict, Any
import numpy as np
import random

from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode
from src.algorithm.rl.agent import PathFindingAgent
from src.algorithm.rl.util.routing_env_gen import RandomPanelGen
from src.cable_routing.routing_component.panel import Panel
from src.cable_routing.routing_component.terminal import Terminal
from src.cable_routing.routing_component.terminal_block import TerminalBlock
from src.cable_routing.routing_component.cable import Cable
from src.algorithm.pathfinding.pathfinding import PathFollower


class MaskableGridsEnv(gym.Env):    
    def __init__(self, 
                 map_size: int = 30,
                 max_step: int = 100,   
                 grids: Optional[VoxelGrids3D] = None):      
        super().__init__()
        self.max_step: int = max_step   
        self.map_size: int = map_size   
        self.grids: Optional[VoxelGrids3D] = grids
        self.cur_node: Optional[VoxelNode] = None
        self.agent: Optional[PathFindingAgent] = None   
        self.cur_setp: int = 0  
        self.action_space = spaces.Discrete(n = 26) 
        
    @staticmethod
    def valid_action_mask(env: 'MaskableGridsEnv') -> List[bool]:
        n_actions = 26
        possible_actions = np.arange(n_actions)
        
        x, y, z = env.cur_node.i, env.cur_node.j, env.cur_node.k
        invalid_actions = []
        for i in range(n_actions):
            nx, ny, nz = env.agent.action_table[i]
            nx, ny, nz = nx + x, ny + y, nz + z
            if 0 > nx or nx >= env.grids.map_size or\
                0 > ny or ny >= env.grids.map_size or \
                0 > nz or nz >= env.grids.map_size:
                    invalid_actions.append(i)
                    continue
            if env.grids[nx, ny, nz].is_obstacle:
                invalid_actions.append(i)
        masked_action = [action not in invalid_actions for action in possible_actions]
        return masked_action
    
    def _is_vail_action(self, action: Tuple[int, int, int]) -> bool:
        x, y, z = self.cur_node.i, self.cur_node.j, self.cur_node.k
        nx, ny, nz = action
        nx, ny, nz = nx + x, ny + y, nz + z
        if 0 > nx or nx >= self.grids.map_size or\
            0 > ny or ny >= self.grids.map_size or \
            0 > nz or nz >= self.grids.map_size:
            return False
        if self.grids[nx, ny, nz].is_obstacle:
            return False
        return True 
    
    def _has_vertical_corner(self, cur_node: VoxelNode, nxt_node: VoxelNode) -> bool: 
        if cur_node.parent == None:
            return False
        parent_node: VoxelNode = cur_node.parent   
        parent_node_center = parent_node.center_pnt 
        cur_node_center = cur_node.center_pnt   
        nxt_node_center = nxt_node.center_pnt   
        vec1 = gp_Vec(parent_node_center, cur_node_center)
        vec2 = gp_Vec(cur_node_center, nxt_node_center)
        vec2.Reverse()

        angle = vec1.Angle(vec2)
        angle = min(angle, 2*np.pi - angle)  # Get smaller angle
        if angle <= np.pi/2:  # 90 degrees in radians
            return True
        return False
        
    def _has_backward_movement(self, cur_node: VoxelNode, nxt_node: VoxelNode) -> bool: 
        old_distance = cur_node.center_pnt.Distance(self.grids.goal_node.center_pnt)   
        new_distance = nxt_node.center_pnt.Distance(self.grids.goal_node.center_pnt)
        if new_distance >= old_distance:
            return True 
        return False    
    
    def _has_redundant_node(self, path_nodes: List[VoxelNode]) -> bool: 
        visited_nodes: Set[VoxelNode] = set()   
        for node in path_nodes:
            if node in visited_nodes:
                return True
            visited_nodes.add(node) 
        return False    

    def render(self):
        return

    def close(self):
        return NotImplemented()

class PathFindingEnv(MaskableGridsEnv):
    metadata = {"render_modes": [None]}
    def __init__(self,
                available_areas: List[Bnd_Box],
                sections: List[int] = [2],    
                len_bounary: tuple[int, int] = (5, 100),
                turning_boundary: tuple[int, int] = (1, 100),  
                map_size: int = 30, 
                max_step: int = 100,
                n_frame: int = 1):
        super().__init__(
            map_size=map_size,
            max_step=max_step)
        self.available_areas: List[Bnd_Box] = available_areas
        self.cur_ep_rewards: List[List[float]] = []
        self.action_space = spaces.Discrete(n = 26)
        self.agent: PathFindingAgent = PathFindingAgent(n_frames=n_frame)
        
        dict_for_obs_space = {}
        for obs in self.agent.observations:
            dict_for_obs_space[obs.obs_name] = spaces.Box(low=-1.0, high=1.0, shape=(obs.obs_dims*n_frame,), dtype=np.float64)  
        self.observation_space = spaces.Dict(dict_for_obs_space)    
        
        self.sections: List[int] = sections 
        self.len_bounary: tuple[int, int] = len_bounary 
        self.turning_boundary: tuple[int, int] = turning_boundary
        self.path_nodes: List[VoxelNode] = []


    def step(self, action):
        reward: float = -1.0
        terminated: bool = False
        self.cur_step += 1
        action: tuple[int, int, int] = self.agent.action(action)
        if self._is_vail_action(action) == False:
            reward -= 1.0
        else:
            nxt_i, nxt_j, nxt_k = self.cur_node.i + action[0], self.cur_node.j + action[1], self.cur_node.k + action[2]
            nxt_node: VoxelNode = self.grids[nxt_i, nxt_j, nxt_k]
            if self._has_backward_movement(self.cur_node, nxt_node):
                reward -= 1.0
            if self._has_vertical_corner(self.cur_node, nxt_node):  
                reward -= 1.0   
            
            nxt_node.parent = self.cur_node
            self.cur_node = nxt_node    
            self.path_nodes.append(self.cur_node)
            for hot_zone in self.panel.hot_zones:
                max_dist: float = hot_zone.max_dist
                if hot_zone.bnd.IsOut(self.cur_node.center_pnt) == False:   
                    maxx, maxy, maxz = hot_zone.bnd.CornerMax().Coord() 
                    minx, miny, minz = hot_zone.bnd.CornerMin().Coord() 
                    x, y, z = self.cur_node.center_pnt.Coord()
                    dx = max(minx - x, 0, x - maxx)
                    dy = max(miny - y, 0, y - maxy)
                    dz = max(minz - z, 0, z - maxz)
                    dist: float = (dx**2 + dy**2 + dz**2)**0.5
                    reward += ((min(dist, max_dist) - max_dist)/max_dist)*1.0
                else:
                    reward -= 1.0
            
        if self.cur_node.is_goal_node: 
            print(f"Routing is Done!!: {self.map_size}")
            terminated = True
            reward += 10.0
            
        if self.cur_step >= self.max_step:
            print(f"Max Step: {self.map_size}")
            reward -= 10.0
            terminated = True
            
        if terminated == True and not self._has_redundant_node(self.path_nodes):
            path_node: List[VoxelNode] = PathFollower(self.grids).get_smooth_path_nodes()
            path_pnts: List[gp_Pnt] = [node.center_pnt for node in path_node]
            
            if len(path_pnts) > 2:
                cable: Cable = Cable()
                cable.init_brep_solid(path_pnts, diameter=4.0, thickness=0.2)
                terminal_block_bnds: List[Bnd_Box] = [block.bnd for block in self.panel.terminal_blocks]    
                if cable.has_collision(terminal_block_bnds) == True:
                    reward -= 5.0
                else:   
                    reward += 5.0  
                    
                hot_zone_bnds: List[Bnd_Box] = [zone.bnd for zone in self.panel.hot_zones]
                if cable.has_collision(hot_zone_bnds) == True:
                    reward -= 5.0
                else:
                    reward += 5.0  
                
                min_radius: float = cable.get_min_radius()
                safe_min_radius: float = 4.0 * 6.0
                if min_radius > safe_min_radius:
                    reward += 5.0
                elif min_radius < safe_min_radius:
                    reward -= 5.0
        
        self.cur_ep_rewards[-1].append(reward)
        return self.agent.get_observation(self.cur_node), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.cur_step = 0
        self.cur_ep_rewards.append([])
        self.path_nodes = []
        if len(self.cur_ep_rewards) > 150:
            self.cur_ep_rewards.pop(0)  
        available_area: Bnd_Box = random.choice(self.available_areas)    
        n_sections: int = self.sections[self.available_areas.index(available_area)]
        panel: Panel= RandomPanelGen().generate(
            available_area=available_area, 
            n_terminals=10,
            n_sections=n_sections,  
            len_boundary=self.len_bounary,
            turning_boundary=self.turning_boundary, 
            resolution=self.map_size)   
        self.panel: Panel = panel
        
        self.grids = panel.grids
        while True:
            terminal_block: TerminalBlock = random.choice(panel.terminal_blocks)
            if len(terminal_block.terminals) > 0:   
                break   
        src_terminal: Terminal = random.choice(terminal_block.terminals)
        dst_terminal: Terminal = src_terminal.dst_terminal
        
        start_node: VoxelNode = Terminal.search_terminal_in_grid(self.grids, src_terminal)    
        goal_node: VoxelNode = Terminal.search_terminal_in_grid(self.grids, dst_terminal) 
        self.grids.set_start_node(start_node)   
        self.grids.set_goal_node(goal_node) 
        
        self.agent.set_panel(panel)
        self.cur_node = self.grids.start_node
        self.path_nodes.append(self.cur_node) 
        return self.agent.get_observation(self.cur_node), {}
        
    def get_observation(self, cur_node: VoxelNode) -> dict:
        observations = self.agent.get_observation(cur_node)
        return observations        
    
    def get_action(self, cur_node: VoxelNode, nxt_node: VoxelNode) -> Tuple[int, int, int]:
        direction: Tuple[int, int, int] = (nxt_node.i - cur_node.i, nxt_node.j - cur_node.j, nxt_node.k - cur_node.k)   
        for i in range(len(self.agent.action_table)):   
            if self.agent.action_table[i] == direction:
                return i    
        raise ValueError(f"Invalid Action: {direction}")    

# # -20.0, -30.0
class CurriculumPathFindingEnv(PathFindingEnv):
    metadata = {"render_modes": [None]}
    def __init__(self,
                available_areas: List[Bnd_Box],  
                len_bounary: tuple[int, int] = (5, 100),
                turning_boundary: tuple[int, int] = (1, 100),  
                map_size: int = 30, 
                max_step: int = 100,
                n_frame: int = 1):
        super().__init__(
            available_areas=available_areas,
            sections=[2],
            len_bounary=len_bounary,
            turning_boundary=turning_boundary,
            map_size=map_size,
            max_step=max_step)  
        self.level: int = 1
        self.max_level: int = 5
        self.agent: PathFindingAgent = PathFindingAgent(n_frames=n_frame)
        dict_for_obs_space = {}
        for obs in self.agent.observations:
            dict_for_obs_space[obs.obs_name] = spaces.Box(low=-1.0, high=1.0, shape=(obs.obs_dims*n_frame,), dtype=np.float64)  
        self.observation_space = spaces.Dict(dict_for_obs_space)    
        self.level_to_sections: Dict[int, Any] = {
            1: [2, 3],
            2: [2, 3],
            3: [2, 3],
            4: [2, 3],
            5: [2, 3]}
        self.level_to_resolution: Dict[int, int] = {
            1: 10,
            2: 20,
            3: 30,
            4: 40,
            5: 50}
        self.level_to_score: Dict[int, float] = {
            1: -20.0,
            2: -70.0,
            3: 100.0,
            4: 100.0,
            5: 100.0}
        

    def reset(self, seed=None, options=None):
        if len(self.cur_ep_rewards) > 150:  
            self.cur_ep_rewards.pop(0)
        
        if self.is_level_completed():   
            self.level_up() 
        self.cur_step = 0
        self.cur_ep_rewards.append([])
        self.sections = self.level_to_sections[self.level]  
        self.map_size = self.level_to_resolution[self.level]    

        available_area: Bnd_Box = random.choice(self.available_areas)
        if self.available_areas.index(available_area) == 0: 
            n_sections: int = 2
        else:
            n_sections: int = random.choice(self.sections)  
        
        panel: Panel= RandomPanelGen().generate(
            available_area=available_area, 
            n_terminals=10,
            n_sections=n_sections,  
            len_boundary=self.len_bounary,
            turning_boundary=self.turning_boundary, 
            resolution=self.map_size)   
        self.panel = panel
        self.grids = panel.grids
        while True:
            terminal_block: TerminalBlock = random.choice(panel.terminal_blocks)
            if len(terminal_block.terminals) > 0:   
                break   
        src_terminal: Terminal = random.choice(terminal_block.terminals)    
        dst_terminal: Terminal = src_terminal.dst_terminal
        start_node: VoxelNode = Terminal.search_terminal_in_grid(self.grids, src_terminal)    
        goal_node: VoxelNode = Terminal.search_terminal_in_grid(self.grids, dst_terminal) 
        self.grids.set_start_node(start_node)   
        self.grids.set_goal_node(goal_node) 
        
        self.agent.set_panel(panel)
        self.cur_node = self.grids.start_node
        return self.agent.get_observation(self.cur_node), {}

    def is_level_completed(self):   
        if self.level >= self.max_level: 
            return False
        
        if len(self.cur_ep_rewards) >= 100:
            sum_rewards: List[float] = []    
            for idx in range(-1, -101, -1):  
                ep_rewards: List[float] = self.cur_ep_rewards[idx]  
                sum_rewards.append(sum(ep_rewards)) 
            mean_reward = np.mean(sum_rewards)
            if mean_reward >= self.level_to_score[self.level]:
                return True
        return False 

    def level_up(self):
        self.cur_ep_rewards.clear()
        
        self.level += 1
        if self.level > self.max_level:
            self.level = self.max_level 
            return
        return