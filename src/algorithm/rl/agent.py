from OCC.Core.Bnd import Bnd_Box    

from abc import ABC
import numpy as np
from typing import Tuple, Optional, List, Dict

from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode    
from src.algorithm.rl.observation import SensorObservation, PanelConfigObservation, ActionObservation, Observation, VoxelObservation
from src.cable_routing.routing_component.panel import Panel


class VoxelAgent(ABC):
    def __init__(self, grids: VoxelGrids3D = None) -> None:
        super().__init__()   
        self.grids: Optional[VoxelGrids3D] = grids 
        self.action_table = {
            0: (1, 0, 0),
            1: (-1, 0, 0),
            2: (0, 1, 0),
            3: (0, -1, 0),
            4: (0, 0, 1),
            5: (0, 0, -1),
            6: (1, 1, 0),
            7: (1, -1, 0),
            8: (-1, 1, 0),
            9: (-1, -1, 0),
            10: (1, 0, 1),
            11: (1, 0, -1),
            12: (-1, 0, 1),
            13: (-1, 0, -1),
            14: (0, 1, 1),
            15: (0, 1, -1),
            16: (0, -1, 1),
            17: (0, -1, -1),
            18: (1, 1, 1),
            19: (1, 1, -1),
            20: (1, -1, 1),
            21: (-1, 1, 1),
            22: (1, -1, -1),
            23: (-1, 1, -1),
            24: (-1, -1, 1),
            25: (-1, -1, -1)
        }

    def get_action(self, action: np.int64) -> Tuple[int, int, int]:
        return self.action_table[action]
    
class PathFindingAgent(VoxelAgent):
    def __init__(self, 
                panel: Optional[Panel] = None,
                n_frames: int = 4) -> None:
        super().__init__()
        self.panel: Optional[Panel] = panel
        
        self.observations: List[Observation] = [
            SensorObservation(),
            PanelConfigObservation(),
            ActionObservation(),
        ]
        if self.panel != None:
            for observation in self.observations:
                observation.set_panel(self.panel)
        self.frames = []
        self.n_frames = n_frames 
        self.obs_dims: int = 0
        for observation in self.observations:
            self.obs_dims += observation.obs_dims
        
    def set_panel(self, panel: Panel) -> None: 
        if isinstance(panel, Panel) == False:
            raise ValueError("PathFinding must set a panel")
        self.panel = panel
        for observation in self.observations:
            observation.set_panel(panel)
        return
        
    def action(self, action: np.int64) -> Tuple[int, int, int]:
        return self.get_action(action)

    def get_observation(self, cur_node: VoxelNode) -> Dict[str, np.ndarray]:
        dict_observation = {
        }   
        for observation in self.observations:
            obs = observation.get_observation(cur_node)   
            dict_observation[observation.obs_name] = obs    
            
        return dict_observation