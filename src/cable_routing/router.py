from OCC.Core.gp import gp_Pnt, gp_Vec
from typing import List, Tuple, Set, Optional, Dict, Any

from src.algorithm.pathfinding.pathfinding import PathFinding, JumpPointSearch, ThetaStar, AStar, JumpPointSearchTheta, RLPathFinding
from src.datastruct.voxel_grids import VoxelNode, VoxelGrids3D
from src.cable_routing.routing_component.panel import Panel
from src.cable_routing.routing_component.terminal import Terminal
from src.cable_routing.routing_component.cable import Cable


class CableRoutingManager: 
    def __init__(self, 
                 panel: Panel) -> None:
        self.panel: Panel = panel
        self.terminal_blocks: Set[Terminal] = set()
        for block in panel.terminal_blocks: 
            self.terminal_blocks.update(block.terminals)   
            
        self.cables: List[Cable] = []   
        self.exe_times: List[float] = []
        
    def route(self, 
            pathfinder: PathFinding, 
            is_smooth: bool = True,
            diameter: float = 2.0, 
            thickness: float = 1.0) -> List[Cable]:
        for terminal in self.terminal_blocks:
            if terminal.is_routed == True:
                continue    
            if terminal.dst_terminal is None:
                continue
            terminal_pair: Tuple[Terminal, Terminal] = (terminal, terminal.dst_terminal)
            cable_router = CableRouter()  
            cable = cable_router.route(
                panel=self.panel,
                pathfinder=pathfinder,
                terminal_pair=terminal_pair,
                is_smooth=is_smooth,
                diameter=diameter,
                thickness=thickness)
            if isinstance(cable, ValueError):
                continue
            self.exe_times.append(cable_router.exe_time)    
            self.cables.append(cable)
        return self.cables    

class CableRouter:
    def __init__(self) -> None:        
        self.exe_time: float = 0.0  
        
        
    def route(self,
            panel: Panel,
            pathfinder: PathFinding,
            terminal_pair: Tuple[Terminal, Terminal],
            is_smooth: bool = True,
            diameter: float=2.0,
            thickness: float=1.0) -> Optional[Cable]: 
        src_terminal, dst_terminal = terminal_pair 
        if panel.grids == None:
            return None
        panel.grids.reset()
        
        src_node: VoxelNode = Terminal.search_terminal_in_grid(panel.grids, src_terminal)
        dst_node: VoxelNode = Terminal.search_terminal_in_grid(panel.grids, dst_terminal)    
        panel.grids.set_start_node(src_node)  
        panel.grids.set_goal_node(dst_node) 
        
        if isinstance(pathfinder, RLPathFinding):
            pathfinder.set_panel(panel)
        
        if pathfinder.search(grid=panel.grids):
            self.exe_time = pathfinder.exe_time
            if is_smooth:
                path_nodes: List[VoxelNode] =  pathfinder.get_smooth_path_nodes(panel.grids)
            else:
                path_nodes: List[VoxelNode] = pathfinder.get_path_nodes(panel.grids)
            path_pnts: List[gp_Pnt] = [node.center_pnt for node in path_nodes]  
            cable: Cable = CableModeler().modeling(
                path_pnts=path_pnts,
                src_terminal=src_terminal,
                dst_terminal=dst_terminal,
                diameter=diameter,
                thickness=thickness)    
            return cable
        else:
            return None

class CableModeler:
    def __init__(self) -> None:
        pass
    
    
    def modeling(self, 
                    path_pnts: List[gp_Pnt], 
                    src_terminal: Terminal,
                    dst_terminal: Terminal,
                    diameter: float=2.0,
                    thickness: float=1.0) -> Optional[Cable]:
        start_front_pnt = src_terminal.front_pnt
        start_terminal_pnt = src_terminal.terminal_pnt
        input_pnts: List[gp_Pnt] = [start_terminal_pnt, start_front_pnt]    
        if src_terminal.terminal_pnt.Distance(src_terminal.front_pnt) <= 1:
            input_pnts.remove(start_front_pnt)

        end_front_pnt = dst_terminal.front_pnt
        end_terminal_pnt = dst_terminal.terminal_pnt
        output_pnts: List[gp_Pnt] = [end_front_pnt, end_terminal_pnt]   
        if dst_terminal.terminal_pnt.Distance(dst_terminal.front_pnt) <= 1:
            output_pnts.remove(end_front_pnt)

        cable_pnts: List[gp_Pnt] = input_pnts + path_pnts + output_pnts
        if len(cable_pnts) <= 2:
            print("Number of Path points is less than 2")
            return None
        
        cable = Cable()
        cable.init_brep_solid(
            gp_pnts=cable_pnts,
            diameter=diameter,
            thickness=thickness)    
        return cable    

if __name__ == "__main__":
    from src.display.scene import Scene
    from src.algorithm.rl.util.routing_env_gen import RandomPanelGen
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy

    def test_routing_example(pathfinding: PathFinding=AStar()) -> None: 
        panel = Panel.read_panel("data/패널_3번_관련_파일_정보.json")
        panel.init_brep_solid()
        panel.init_voxel_grids(resolution=30)
        
        manager = CableRoutingManager(panel=panel)
        cables: List[Cable] = manager.route(pathfinder=pathfinding, is_smooth=True)
        import pandas as pd 
        df = pd.DataFrame({'length': [cable.get_linear_length() for cable in cables],   
                           'min_radius': [cable.get_min_inner_radius() for cable in cables],
                           'exe_time': manager.exe_times})  
        df.to_excel('cable_routing_result.xlsx', index=False)  
        hot_zone_bnds = [hot_zone.bnd for hot_zone in panel.hot_zones]  
        n_hot_zone_collision: int = 0
        for cable in cables:    
            if cable.has_inner_collision(hot_zone_bnds):
                n_hot_zone_collision += 1   
        print(f"Number of Hot Zone Collision: {n_hot_zone_collision}")  
        n_obstacle_collision: int = 0   
        termainal_bnds  = [terminal_block.bnd for terminal_block in panel.terminal_blocks]  
        for cable in cables:    
            if cable.has_inner_collision(termainal_bnds):
                n_obstacle_collision += 1   
        
        print(f"Number of Obstacle Collision: {n_obstacle_collision}")

        avg_cable_length = sum([cable.get_linear_length() for cable in cables]) / len(cables)
        avg_cable_min_radius = sum([cable.get_min_inner_radius() for cable in cables]) / len(cables)
        print(f"Average Cable Length: {avg_cable_length:.2f}")  
        print(f"Average Cable Min Radius: {avg_cable_min_radius:.2f}")  
        print(f"Average Execution Time: {sum(manager.exe_times)/len(manager.exe_times):.2f}")     
        # scene = Scene()
        # scene.add_entities(cables)
        # scene.add_entity(panel)
        # scene.display()
        
    def random_routing_example(panel: Panel, pathfinding: PathFinding=AStar()) -> None:
        # panel.init_brep_solid()
        # panel.init_voxel_grids(resolution=10)   
        
        manager = CableRoutingManager(panel=panel)
        cables: List[Cable] = manager.route(pathfinder=pathfinding, is_smooth=True)
        import pandas as pd 
        df = pd.DataFrame({'length': [cable.get_linear_length() for cable in cables],   
                           'min_radius': [cable.get_min_inner_radius() for cable in cables],
                           'exe_time': manager.exe_times})  
        df.to_excel('cable_routing_result.xlsx', index=False)  
        avg_cable_length = sum([cable.get_linear_length() for cable in cables]) / len(cables)
        avg_cable_min_radius = sum([cable.get_min_inner_radius() for cable in cables]) / len(cables)
        
        
        hot_zone_bnds = [hot_zone.bnd for hot_zone in panel.hot_zones]  
        n_hot_zone_collision: int = 0
        for cable in cables:    
            if cable.has_inner_collision(hot_zone_bnds):
                n_hot_zone_collision += 1   
        print(f"Number of Hot Zone Collision: {n_hot_zone_collision}")  
        n_obstacle_collision: int = 0   
        termainal_bnds  = [terminal_block.bnd for terminal_block in panel.terminal_blocks]  
        for cable in cables:    
            if cable.has_inner_collision(termainal_bnds):
                n_obstacle_collision += 1   
        
        print(f"Number of Obstacle Collision: {n_obstacle_collision}")

        print(f"Average Cable Length: {avg_cable_length:.2f}")  
        print(f"Average Cable Min Radius: {avg_cable_min_radius:.2f}")  
        print(f"Average Execution Time: {sum(manager.exe_times)/len(manager.exe_times):.2f}")     
    
    
    model = MaskablePPO.load('data/강화학습_모델/M10_S500_l0_100.zip')
    # env = init_pathfinding_env(map_size=30)
    # print(evaluate_policy(model, env, n_eval_episodes=10, deterministic=False))  
    
    real_panel: Panel = Panel.read_panel("data/패널_3번_관련_파일_정보.json")   
    panel: Panel = RandomPanelGen().generate(
        available_area=real_panel.bnd,
        n_sections=2,
        n_terminals=10,
        resolution=10
    )

    path_finder = RLPathFinding(model)
    random_routing_example(panel, path_finder)
    random_routing_example(panel) 