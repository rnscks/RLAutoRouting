from sb3_contrib import MaskablePPO
from typing import List, Optional, Tuple, Set
from itertools import product
from abc import ABC, abstractmethod 
import heapq
import numpy as np

from src.algorithm.rl.agent import PathFindingAgent  
from src.datastruct.voxel_grids import VoxelNode, VoxelGrids3D
from src.cable_routing.routing_component.panel import Panel

class PathFollower:
    def __init__(self, grids: VoxelGrids3D) -> None:
        self.grids: VoxelGrids3D = grids
        pass
    
    
    def get_path_nodes(self) -> List[VoxelNode]:
        path_nodes: List[VoxelNode] = []
        cur_node: VoxelNode = self.grids.goal_node
        self.grids.start_node.parent = None
        while cur_node:
            path_nodes.append(cur_node)
            if isinstance(cur_node.parent, VoxelNode):
                cur_node = cur_node.parent
            else:
                break
        
        return path_nodes[::-1]
    
    def get_smooth_path_nodes(self) -> List[VoxelNode]:
        path_nodes: List[VoxelNode] = []
        cur_node: VoxelNode = self.grids.goal_node
        self.grids.start_node.parent = None
        while cur_node:
            path_nodes.append(cur_node)
            if isinstance(cur_node.parent, VoxelNode):
                while cur_node.parent.parent != None:
                    if self._has_line_of_sight(cur_node, cur_node.parent.parent, self.grids):  
                        cur_node.parent = cur_node.parent.parent
                    else:
                        break
                cur_node = cur_node.parent
            else:
                break
        return path_nodes[::-1]
    
    def _has_line_of_sight(self, 
                        src_node: VoxelNode, 
                        dst_node: VoxelNode, 
                        grids: VoxelGrids3D) -> bool:
        x1, y1, z1 = src_node.i, src_node.j, src_node.k
        x2, y2, z2 = dst_node.i, dst_node.j, dst_node.k

        if (x2 > x1):
            xs = 1
            dx = x2 - x1
        else:
            xs = -1
            dx = x1 - x2

        if (y2 > y1):
            ys = 1
            dy = y2 - y1
        else:
            ys = -1
            dy = y1 - y2

        if (z2 > z1):
            zs = 1
            dz = z2 - z1
        else:
            zs = -1
            dz = z1 - z2

        if (dx >= dy and dx >= dz):
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while (x1 != x2):
                x1 += xs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dx
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                if grids[x1, y1, z1].is_obstacle: 
                    return False

        elif (dy >= dx and dy >= dz):
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while (y1 != y2):
                y1 += ys
                if (p1 >= 0):
                    x1 += xs
                    p1 -= 2 * dy
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                if grids[x1, y1, z1].is_obstacle:  
                    return False
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while (z1 != z2):
                z1 += zs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dz
                if (p2 >= 0):
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                if grids[x1, y1, z1].is_obstacle:  
                    return False
        return True

class PathFinding(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    
    @abstractmethod
    def search(self, grid: VoxelGrids3D) -> bool:
        pass    
    
    def get_path_nodes(self, grids: VoxelGrids3D) -> List[VoxelNode]:
        return PathFollower(grids).get_path_nodes()    
    
    def get_smooth_path_nodes(self, grids: VoxelGrids3D) -> List[VoxelNode]:
        return PathFollower(grids).get_smooth_path_nodes() 
    
    def _get_neighbors(self, grids: VoxelGrids3D, node: VoxelNode) -> List[VoxelNode]:
        neighbors: List[VoxelNode] = []  
        
        for dx, dy, dz in product([-1, 0, 1], repeat=3):
            if dx == dy == dz == 0:
                continue
            nxt_i, nxt_j, nxt_k = node.i + dx, node.j + dy, node.k + dz
            if not self._is_valid(grids, nxt_i, nxt_j, nxt_k): 
                continue
            neighbors.append(grids[nxt_i, nxt_j, nxt_k])
        return neighbors
    
    def _is_valid(self, grids: VoxelGrids3D, nxt_i: int, nxt_j: int, nxt_k: int) -> bool:
        if not (0 <= nxt_i < grids.map_size):
            return False    
        if not (0 <= nxt_j < grids.map_size):
            return False
        if not (0 <= nxt_k < grids.map_size):
            return False
        nei_node: Optional[VoxelNode] = grids[nxt_i, nxt_j, nxt_k]    
        if nei_node == None:
            print("PathFinding: Neighbor node is None")
            return False
        if nei_node.is_obstacle == True:    
            return False    
        return True

class RLPathFinding(PathFinding):   
    def __init__(self, model: MaskablePPO) -> None:
        super().__init__()
        self.model = model  
        self.exe_time = 0.0 
        
    def search(self, grid: VoxelGrids3D) -> bool:
        print("INPUT GRID DOES NOT USE FOR PATHFINDING")
        if self.panel.grids == None or self.panel.grids.start_node == None:
            raise ValueError("Panel is not set")
        cur_node: VoxelNode = self.panel.grids.start_node
        agent = PathFindingAgent(panel=self.panel, n_frames=1)
        
        done = False
        obs = agent.get_observation(cur_node)   
        path_nodes: List[VoxelNode] = []    
        
        max_steps = 100
        step = 0
        import time
        start_time = time.time()
        
        while not done:
            action, _ = self.model.predict(obs, action_masks=self._valid_action_mask(agent, cur_node, self.panel.grids), deterministic=False)
            dir_i, dir_j, dir_k = agent.action(int(action))
            next_node = self.panel.grids[cur_node.i + dir_i, cur_node.j + dir_j, cur_node.k + dir_k] 
            path_nodes.append(next_node)    
            if next_node.is_goal_node:
                next_node.parent = cur_node 
                break
            next_node.parent = cur_node
            obs = agent.get_observation(next_node)
            cur_node = next_node
            step += 1   
            if step > max_steps:
                return False   
        self.exe_time = time.time() - start_time
        path_nodes = self._remove_duplicated_node(path_nodes)   
        for idx in range(len(path_nodes) - 1, 0, -1):       
            prev_node = path_nodes[idx - 1]    
            cur_node = path_nodes[idx]  
            cur_node.parent = prev_node
        
        return True 
    
    def set_panel(self, panel: Panel) -> None:  
        self.panel = panel  
        return
    
    def _valid_action_mask(self, agent: PathFindingAgent, cur_node: VoxelNode, grids: VoxelGrids3D) -> np.ndarray:
        n_actions = 26
        possible_actions = np.arange(n_actions)

        x, y, z = cur_node.i, cur_node.j, cur_node.k
        invalid_actions = []
        for i in range(n_actions):
            nx, ny, nz = agent.action_table[i]
            nx, ny, nz = nx + x, ny + y, nz + z
            if 0 > nx or nx >= grids.map_size or\
                0 > ny or ny >= grids.map_size or \
                0 > nz or nz >= grids.map_size:
                    invalid_actions.append(i)
                    continue
            if grids[nx, ny, nz].is_obstacle:
                invalid_actions.append(i)
        masked_action = [action not in invalid_actions for action in possible_actions]
        return masked_action
    
    def _remove_duplicated_node(self, path_nodes: List[VoxelNode]) -> List[VoxelNode]:
        new_path_nodes: List[VoxelNode] = []
        for node in path_nodes:
            if node not in new_path_nodes:
                new_path_nodes.append(node)
        return new_path_nodes   

class AStar(PathFinding):
    def __init__(self) -> None:
        super().__init__()
        self.exe_time = 0.0 
        
        
    def search(self, grid: VoxelGrids3D) -> bool:
        import time
        start_time = time.time()    
        if grid.start_node == None or grid.goal_node == None:   
            raise ValueError("Start or goal node is not set")   
        if grid == None:
            raise ValueError("Grids is not set")    
        
        open_list: List[VoxelNode] = []  
        closed_list: set[VoxelNode] = set() 
        
        start_node: VoxelNode = grid.start_node    
        heapq.heappush(open_list, start_node)   
        while open_list:
            cur_node: VoxelNode  = heapq.heappop(open_list)    
            closed_list.add(cur_node)    

            if cur_node.is_goal_node:
                self.exe_time = time.time() - start_time    
                return True 
            
            neighbors: List[VoxelNode] = self._get_neighbors(grid, cur_node)   
            for successor in neighbors:  
                if successor in closed_list:
                    continue
                
                ng = cur_node.g + cur_node.center_pnt.Distance(successor.center_pnt)
                h = successor.center_pnt.Distance(grid.goal_node.center_pnt)
                if successor.parent == None or successor.g > ng:
                    successor.g = ng
                    successor.f = h + ng
                    successor.parent = cur_node
                    heapq.heappush(open_list, successor) 
        return False    

class ThetaStar(PathFinding):
    def __init__(self) -> None:
        super().__init__()        
    
    
    def search(self, grids: VoxelGrids3D) -> bool:
        if grids.start_node == None or grids.goal_node == None:   
            raise ValueError("Start or goal node is not set")   
        if grids == None:
            raise ValueError("Grids is not set")    

        open_list: List[VoxelNode] = []  
        closed_list: set[VoxelNode] = set()
        
        start_node: VoxelNode = grids.start_node    
        heapq.heappush(open_list, start_node)   
        while open_list:
            cur_node: VoxelNode = heapq.heappop(open_list)   
            closed_list.add(cur_node)    

            if cur_node.is_goal_node:   
                return True 
            
            neighbors: List[VoxelNode] = self._get_neighbors(grids, cur_node)   
            for successor in neighbors:
                if successor in closed_list:
                    continue
                
                h = successor.center_pnt.Distance(grids.goal_node.center_pnt)   
                if cur_node.parent != None and PathFollower(grids)._has_line_of_sight(cur_node.parent, successor, grids):
                    ng: float = cur_node.parent.g + cur_node.parent.center_pnt.Distance(successor.center_pnt) 
                    if successor.parent == None or ng < successor.g:
                        successor.g = ng
                        successor.f = h + ng
                        successor.parent = cur_node.parent
                        heapq.heappush(open_list, successor)    
                else:
                    ng = cur_node.g + cur_node.center_pnt.Distance(successor.center_pnt)
                    if successor.parent == None or successor.g > ng:
                        successor.g = ng
                        successor.f = h + ng
                        successor.parent = cur_node
                        heapq.heappush(open_list, successor)    
        return False    

class JumpPointSearch(PathFinding):
    def __init__(self):
        self._neighbor_dirs = list(product([-1, 0, 1], repeat=3))
        self._neighbor_dirs.remove((0, 0, 0))
        self._neighbor_dirs = np.array(list(product([-1, 0, 1], repeat=3)))
        self._neighbor_dirs = self._neighbor_dirs[np.any(self._neighbor_dirs != 0, axis=1)]  # (0,0,0) 제거

    
    def search(self, grids: VoxelGrids3D) -> bool:
        self.valid_range = range(grids.map_size)
        open_list: List[VoxelNode] = []  
        closed_list: Set[VoxelNode] = set()
        start_node: VoxelNode = grids.start_node
        heapq.heappush(open_list, start_node)
        
        while open_list:
            cur_node: VoxelNode = heapq.heappop(open_list)
            closed_list.add(cur_node)

            if cur_node.is_goal_node:
                return True
            
            successors: List[VoxelNode] = self._get_successors(cur_node, grids)   
            for successor in successors:
                if successor in closed_list:
                    continue
                
                ng = cur_node.g + cur_node.center_pnt.Distance(successor.center_pnt)
                h = successor.center_pnt.Distance(grids.goal_node.center_pnt)   

                if successor.parent == None or successor.g > ng:
                    successor.g = ng
                    successor.f = h + ng
                    successor.parent = cur_node
                    heapq.heappush(open_list, successor) 
        return False
    
    def _get_successors(self, node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:    
        successors: List[VoxelNode] = []
        for neighbor in self._prune(node, grids):
            jump_node: Optional[VoxelNode] = self._jump(
                node, 
                grids, 
                neighbor.i - node.i,
                neighbor.j - node.j,
                neighbor.k - node.k)
            
            if jump_node:
                successors.append(jump_node)
        return successors
    
    def _jump(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> Optional[VoxelNode]:        
        nxt_i, nxt_j, nxt_k = node.i + dir_i, node.j + dir_j, node.k + dir_k

        if not self._is_valid(node, grids, dir_i, dir_j, dir_k): 
            return None 
        nxt_node: VoxelNode = grids[nxt_i, nxt_j, nxt_k]
        if nxt_node.is_obstacle:    
            return None 
        
        nxt_node.scaning_parent = node
        
        if nxt_node.is_goal_node:
            nxt_node.scaning_parent = node
            return nxt_node
        
        if self._get_forced_neighbors_all(nxt_node, node, grids, dir_i, dir_j, dir_k) != []:   
            return nxt_node
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 2:
            dirs = [(0, dir_j, 0), (dir_i, 0, 0), (0, 0, dir_k)]
            for dir in dirs:
                if dir == (0, 0, 0):    
                    continue
                if self._jump(nxt_node, grids, *dir) != None:
                    return nxt_node
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 3:
            dirs = [(0, dir_j, dir_k), (dir_i, 0, dir_k), (dir_i, dir_j, 0), (dir_i, 0, 0), (0, dir_j, 0), (0, 0, dir_k)]   
            for dir in dirs:
                if dir == (0, 0, 0):    
                    continue
                if self._jump(nxt_node, grids, *dir) != None:
                    return nxt_node
        
        return self._jump(nxt_node, grids, dir_i, dir_j, dir_k)    

    def _prune(self, cur_node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:   
        parent: Optional[VoxelNode] = cur_node.scaning_parent
        if parent == None:
            return self._get_neighbors(cur_node, grids)
        
        dir_i, dir_j, dir_k = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        canonical_neighbours: List[VoxelNode] = self._get_canonical_neighbours(cur_node, grids, dir_i, dir_j, dir_k)  
        forced_neighbors: List[VoxelNode] = self._get_forced_neighbors_all(cur_node, parent, grids, dir_i, dir_j, dir_k)    
        return canonical_neighbours + forced_neighbors
    
    def _get_canonical_neighbours(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> List[VoxelNode]:        
        canonical_neighbours: Set[VoxelNode] = set()
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 1:
            dirs = [(dir_i, dir_j, dir_k)]
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 2:
            if dir_i != 0 and dir_j != 0 and dir_k == 0:
                dirs = [
                    (dir_i, dir_j, 0),  # 현재 대각선 방향
                    (dir_i, 0, 0),      # i축 방향
                    (0, dir_j, 0)       # j축 방향
                ]
            elif dir_i != 0 and dir_k != 0 and dir_j == 0:
                dirs = [
                    (dir_i, 0, dir_k),  # 현재 대각선 방향
                    (dir_i, 0, 0),      # i축 방향
                    (0, 0, dir_k)       # k축 방향
                ]
            elif dir_j != 0 and dir_k != 0 and dir_i == 0:
                dirs = [
                    (0, dir_j, dir_k),  # 현재 대각선 방향
                    (0, dir_j, 0),      # j축 방향
                    (0, 0, dir_k)       # k축 방향
                ]
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 3:
            dirs = [
                (dir_i, dir_j, dir_k),  # 현재 3D 대각선 방향
                (dir_i, dir_j, 0),      # xy 평면
                (dir_i, 0, dir_k),      # xz 평면
                (0, dir_j, dir_k),      # yz 평면
                (dir_i, 0, 0),          # x축
                (0, dir_j, 0),          # y축
                (0, 0, dir_k)           # z축
            ]
        
        for dir in dirs:
            if dir == (0, 0, 0):
                continue
            
            nxt_i, nxt_j, nxt_k = node.i + dir[0], node.j + dir[1], node.k + dir[2]
            if not self._is_valid(node, grids, *dir):
                continue
            canonical_neighbours.add(grids[nxt_i, nxt_j, nxt_k])
        return list(canonical_neighbours)

    def _get_forced_neighbors_all(self, node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> List[VoxelNode]:        
        if node.is_checked_forced_neighbours == True:
            return node.forced_neighbours
        else:
            node.is_checked_forced_neighbours = True    
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 1:
            node.forced_neighbours = self._get_forced_neighbor_orthogonal(node, parent, grids)   
            return node.forced_neighbours   
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 2: 
            node.forced_neighbours = self._get_forced_neighbor_diagonal2d(node, parent, grids)
            return node.forced_neighbours   
        else:
            node.forced_neighbours = self._get_forced_neighbor_diagonal3d(node, parent, grids)
            return node.forced_neighbours

    def _get_forced_neighbors(self, cur_node: VoxelNode, 
                            parent: VoxelNode, 
                            grids: VoxelGrids3D,
                            block_dir: Tuple[int, int, int],
                            free_dirs: List[Tuple[int, int, int]],
                            forced_neighbors: Set[VoxelNode]) -> List[VoxelNode]:
        """
        Block_dir: (i, j, k) 는 parent 기준으로 block node의 방향을 나타냄 
        free_dirs: (i, j, k) 는 cur_node 기준으로 free node의 방향을 나타냄  
        """
        if self._is_valid(parent, grids, block_dir[0], block_dir[1], block_dir[2]) == True:
            block: VoxelNode = grids[parent.i + block_dir[0], parent.j + block_dir[1], parent.k + block_dir[2]]
            if block.is_obstacle == True:
                frees: Set[VoxelNode] = self._get_next_nodes(cur_node, 
                                                        grids,
                                                        free_dirs)
                forced_neighbors.update([free for free in frees if free.is_obstacle == False])
        return forced_neighbors

    def _get_forced_neighbor_orthogonal(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        
        if abs(di) == 1:
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, 0),
                                free_dirs=[(0, 1, 0), (0, 1, 1), (0, 1, -1), (di, 1, 0), (di, 1, 1), (di, 1, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, -1, 0),
                                free_dirs=[(0, -1, 0), (0, -1, 1), (0, -1, -1), (di, -1, 0), (di, -1, 1), (di, -1, -1)],
                                forced_neighbors=forced_neighbors)    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(0, 1, 1), (0, -1, 1), (0, 0, 1), (di, 1, 1), (di, -1, 1), (di, 0, 1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, -1),
                                free_dirs=[(0, 1, -1), (0, -1, -1), (0, 0, -1), (di, 1, -1), (di, -1, -1), (di, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, 1),
                                free_dirs=[(0, 1, 1), (di, 1, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, -1),
                                free_dirs=[(0, 1, -1), (di, 1, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, -1, 1),
                                free_dirs=[(0, -1, 1), (di, -1, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, -1),
                                free_dirs=[(0, -1, -1), (di, -1, -1)],
                                forced_neighbors=forced_neighbors)  
            
        if abs(dj) == 1:    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, 0, 0), (1, 0, 1), (1, 0, -1), (1, dj, 0), (1, dj, 1), (1, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, 0, 0), (-1, 0, 1), (-1, 0, -1), (-1, dj, 0), (-1, dj, 1), (-1, dj, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(1, 0, 1), (-1, 0, 1), (0, 0, 1), (1, dj, 1), (-1, dj, 1), (0, dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, -1),
                                free_dirs=[(1, 0, -1), (-1, 0, -1), (0, 0, -1), (1, dj, -1), (-1, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 1),
                                free_dirs=[(1, 0, 1), (1, dj, 1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 0, -1),
                                free_dirs=[(1, 0, -1), (1, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 1),
                                free_dirs=[(-1, 0, 1), (-1, dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 0, -1),
                                free_dirs=[(-1, 0, -1), (-1, dj, -1)],
                                forced_neighbors=forced_neighbors)
        
        if abs(dk) == 1:    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 0, dk), (1, 1, dk), (1, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, 0, 0), (-1, 1, 0), (-1, -1, 0), (-1, 0, dk), (-1, 1, dk), (-1, -1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, 0),
                                free_dirs=[(1, 1, 0), (-1, 1, 0), (0, 1, 0), (1, 1, dk), (-1, 1, dk), (0, 1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, 0),
                                free_dirs=[(1, -1, 0), (-1, -1, 0), (0, -1, 0), (1, -1, dk), (-1, -1, dk), (0, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 1, 0),
                                free_dirs=[(1, 1, 0), (1, 1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, -1, 0),
                                free_dirs=[(1, -1, 0), (1, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 1, 0),
                                free_dirs=[(-1, 1, 0), (-1, 1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, -1, 0),
                                free_dirs=[(-1, -1, 0), (-1, -1, dk)],
                                forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)  

    def _get_forced_neighbor_diagonal2d(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:   
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        
        if abs(di) == 1 and abs(dj) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 0),
                                free_dirs=[(-di, dj, 0),
                                        (-di, dj, 1), (-di, 0, 1), (0, dj, 1), (0, 0, 1),
                                        (-di, dj, -1), (-di, 0, -1), (0, dj, -1), (0, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 0, 0),
                                free_dirs=[(di, -dj, 0), 
                                        (di, -dj, 1), (0, -dj, 1), (di, 0, 1), (0, 0, 1),
                                        (di, -dj, -1), (0, -dj, -1), (di, 0 , -1), (0, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(0, -dj, 1), (-di, 0, 1), (0, 0, 1), (di, 0, 1), (0, dj, 1), (di, dj, 1), (-di, dj, 1), (di, -dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, -1),
                                free_dirs=[(-di, 0, -1), (0, -dj, -1), (0, 0, -1), (di, 0, -1), (0, dj, -1), (di, dj, -1), (-di, dj, -1), (di, -dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 1),
                                free_dirs=[(-di, dj, 1), (0, dj, 1), (0, 0, 1), (di, dj, 1), (di, 0, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, dj, -1),
                                free_dirs=[(-di, dj, -1), (0, dj, -1), (0, 0, -1), (di, dj, -1), (di, 0, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 0, 1),
                                free_dirs=[(di, -dj, 1), (di, 0, -1), (0, 0, -1), (di, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(di, 0, -1),
                                free_dirs=[(di, -dj, -1), (di, 0, -1), (0, 0, -1), (di, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)
        
        if abs(dj) == 1 and abs(dk) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, dk),
                                free_dirs=[(0, -dj, dk),
                                        (1, -dj, dk), (1, -dj, 0), (1, 0, dk), (1, 0, 0),
                                        (-1, -dj, dk), (-1, -dj, 0), (-1, 0, dk), (-1, 0, 0)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 0),
                                free_dirs=[(0, dj, -dk), 
                                        (1, dj, -dk), (1, 0, -dk), (1, dj, 0), (1, 0, 0),
                                        (-1, dj, -dk), (-1, 0, -dk), (-1, dj, 0), (-1, 0, 0)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, -dj, 0), (1, 0, -dk), (1, 0, 0), (1, dj, dk), (1, dj, 0), (1, 0, dk), (1, -dj, dk), (1, dj, -dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, -dj, 0), (-1, 0, -dk), (-1, 0, 0), (-1, dj, dk), (-1, dj, 0), (-1, 0, dk), (-1, -dj, dk), (-1, dj, -dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 0, dk),
                                free_dirs=[(1, -dj, dk), (1, 0, dk), (1, 0, 0), (1, dj, 0), (1, dj, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, dk),
                                free_dirs=[(-1, -dj, dk), (-1, 0, dk), (-1, 0, 0), (-1, dj, 0), (-1, dj, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, dj, 0),
                                free_dirs=[(1, dj, -dk), (1, dj, 0), (1, 0, 0), (1, 0, dk), (1, dj, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, dj, 0),
                                free_dirs=[(-1, dj, -dk), (-1, dj, 0), (-1, 0, 0), (-1, 0, dk), (-1, dj, dk)],   
                                forced_neighbors=forced_neighbors)

        if abs(dk) == 1 and abs(di) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(di, 0, 0),
                                free_dirs=[(di, 0, -dk),
                                        (di, 1, -dk), (0, 1, -dk), (di, 1, 0), (0, 1, 0),
                                        (di, -1, -dk), (0, -1, -dk), (di, -1, 0), (0, -1, 0)],   
                                forced_neighbors=forced_neighbors)         
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, dk),   
                                free_dirs=[(-di, 0, dk), 
                                        (-di, 1, dk), (-di, 1, 0), (0, 1, dk), (0, 1, 0),
                                        (-di, -1, dk), (-di, -1, 0), (0, -1, dk), (0, -1, 0)],     
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, 0),
                                free_dirs=[(0, 1, -dk), (-di, 1, 0), (0, 1, 0), (di, 1, dk), (0, 1, dk), (di, 1, 0), (-di, 1, dk), (di, 1, -dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, 0),   
                                free_dirs=[(-di, -1, 0), (0, -1, -dk), (0, -1, 0), (di, -1, dk), (0, -1, dk), (di, -1, 0), (-di, -1, dk), (di, -1, -dk)],      
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 1, 0),
                                free_dirs=[(di, 1, -dk), (di, 1, 0), (0, 1, 0), (0, 1, dk), (di, 1, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, -1, 0),
                                free_dirs=[(di, -1, -dk), (di, -1, 0), (0, -1, 0), (0, -1, dk), (di, -1, dk)],  
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, dk),
                                free_dirs=[(-di, 1, dk), (0, 1, 0), (0, 1, dk), (di, 1, 0), (di, 1, dk)],    
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, dk),
                                free_dirs=[(-di, -1, dk), (0, -1, 0), (0, -1, dk), (di, -1, 0), (di, -1, dk)],    
                                forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)   

    def _get_forced_neighbor_diagonal3d(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]: 
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k    
        self._get_forced_neighbors(cur_node, parent, grids,    
                            block_dir=(di, dj, 0),
                            free_dirs=[(di, dj, -dk), (0, dj, -dk), (di, 0, -dk)],
                            forced_neighbors=forced_neighbors)      
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(di, 0, 0),
                            free_dirs=[(di, -dj, -dk), (0, 0, -dk), (di, 0, -dk), (0, dj, -dk), (di, dj, -dk),
                                    (di, -dj, 0), (0, -dj, 0)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids, 
                            block_dir=(0, dj, 0),
                            free_dirs=[(-di, dj, -dk), (0, 0, -dk), (0, dj, -dk), (di, 0, -dk), (di, dj, -dk),
                                    (-di, dj, 0), (-di, 0, 0)],  
                            forced_neighbors=forced_neighbors)        
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(0, 0, dk),
                            free_dirs=[(-di, 0, 0), (-di, dj, 0), (0, -dj, 0), (di, -dj, 0), 
                                    (-di, -dj, dk), (0, -dj, dk), (di, -dj, dk), (-di, 0, dk), (-di, dj, dk)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids,    
                            block_dir=(0, dj, dk),
                            free_dirs=[(-di, dj, 0),
                                    (-di, dj, dk), (-di, 0, dk)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(di, 0, dk),
                            free_dirs=[(di, -dj, 0),
                                    (di, -dj, dk), (0, -dj, dk)],   
                            forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)   

    def _is_valid(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> bool:
        nxt_i, nxt_j, nxt_k = node.i + dir_i, node.j + dir_j, node.k + dir_k
        return nxt_i in self.valid_range and nxt_j in self.valid_range and nxt_k in self.valid_range
    
    def _get_next_nodes(self, cur_node: VoxelNode, grids: VoxelGrids3D, directions: Tuple[int, int, int]) -> Set[VoxelNode]:
        nxt_nodes = set()
        for direction in directions:
            dir_i, dir_j, dir_k = direction 
            if self._is_valid(cur_node, grids, dir_i, dir_j, dir_k) == False:  
                continue
            nxt_nodes.add(grids[cur_node.i + dir_i, cur_node.j + dir_j, cur_node.k + dir_k])
        return nxt_nodes

    def _get_neighbors(self, node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:
        neighbors = []
        neighbor_positions = self._neighbor_dirs + np.array([node.i, node.j, node.k])  # numpy 연산 활용
        valid_mask = np.all((neighbor_positions >= 0) & (neighbor_positions < grids.map_size), axis=1)
        for pos in neighbor_positions[valid_mask]:
            neighbors.append(grids[pos[0], pos[1], pos[2]])
        return neighbors

class JumpPointSearchTheta(PathFinding):
    def __init__(self):
        self._neighbor_dirs = list(product([-1, 0, 1], repeat=3))
        self._neighbor_dirs.remove((0, 0, 0))
        self._neighbor_dirs = np.array(list(product([-1, 0, 1], repeat=3)))
        self._neighbor_dirs = self._neighbor_dirs[np.any(self._neighbor_dirs != 0, axis=1)]  # (0,0,0) 제거

    
    def search(self, grids: VoxelGrids3D) -> bool:
        self.valid_range = range(grids.map_size)
        open_list: List[VoxelNode] = []  
        closed_list: Set[VoxelNode] = set()
        start_node: VoxelNode = grids.start_node
        heapq.heappush(open_list, start_node)
        
        while open_list:
            cur_node: VoxelNode = heapq.heappop(open_list)
            closed_list.add(cur_node)

            if cur_node.is_goal_node:
                return True
            
            successors: List[VoxelNode] = self._get_successors(cur_node, grids)   
            for successor in successors:
                if successor in closed_list:
                    continue
                
                h = successor.center_pnt.Distance(grids.goal_node.center_pnt)   
                ng = cur_node.g + cur_node.center_pnt.Distance(successor.center_pnt)
                if cur_node.parent != None and PathFollower(grids)._has_line_of_sight(cur_node.parent, successor, grids):
                    ng: float = cur_node.parent.g + cur_node.parent.center_pnt.Distance(successor.center_pnt) 
                    if successor.parent == None or ng < successor.g:
                        successor.g = ng
                        successor.f = h + ng
                        successor.parent = cur_node.parent
                        heapq.heappush(open_list, successor)    
                elif successor.parent == None or successor.g > ng:
                    successor.g = ng
                    successor.f = h + ng
                    successor.parent = cur_node
                    heapq.heappush(open_list, successor)    
        return False
    
    def _get_successors(self, node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:    
        successors: List[VoxelNode] = []
        for neighbor in self._prune(node, grids):
            jump_node: Optional[VoxelNode] = self._jump(
                node, 
                grids, 
                neighbor.i - node.i,
                neighbor.j - node.j,
                neighbor.k - node.k)
            
            if jump_node:
                successors.append(jump_node)
        return successors
    
    def _jump(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> Optional[VoxelNode]:        
        nxt_i, nxt_j, nxt_k = node.i + dir_i, node.j + dir_j, node.k + dir_k

        if not self._is_valid(node, grids, dir_i, dir_j, dir_k): 
            return None 
        nxt_node: VoxelNode = grids[nxt_i, nxt_j, nxt_k]
        if nxt_node.is_obstacle:    
            return None 
        
        nxt_node.scaning_parent = node
        if nxt_node.is_goal_node:
            return nxt_node
        
        if self._get_forced_neighbors_all(nxt_node, node, grids, dir_i, dir_j, dir_k) != []:   
            return nxt_node
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 2:
            dirs = [(0, dir_j, 0), (dir_i, 0, 0), (0, 0, dir_k)]
            for dir in dirs:
                if dir == (0, 0, 0):    
                    continue
                if self._jump(nxt_node, grids, *dir) != None:
                    return nxt_node
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 3:
            dirs = [(0, dir_j, dir_k), (dir_i, 0, dir_k), (dir_i, dir_j, 0), (dir_i, 0, 0), (0, dir_j, 0), (0, 0, dir_k)]   
            for dir in dirs:
                if dir == (0, 0, 0):    
                    continue
                if self._jump(nxt_node, grids, *dir) != None:
                    return nxt_node
        
        return self._jump(nxt_node, grids, dir_i, dir_j, dir_k)    

    def _prune(self, cur_node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:   
        parent: Optional[VoxelNode] = cur_node.scaning_parent
        if parent == None:
            return self._get_neighbors(cur_node, grids)
        
        dir_i, dir_j, dir_k = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        canonical_neighbours: List[VoxelNode] = self._get_canonical_neighbours(cur_node, grids, dir_i, dir_j, dir_k)  
        forced_neighbors: List[VoxelNode] = self._get_forced_neighbors_all(cur_node, parent, grids, dir_i, dir_j, dir_k)    
        return canonical_neighbours + forced_neighbors
    
    def _get_canonical_neighbours(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> List[VoxelNode]:
        canonical_neighbours: Set[VoxelNode] = set()
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 1:
            dirs = [(dir_i, dir_j, dir_k)]
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 2:
            if dir_i != 0 and dir_j != 0 and dir_k == 0:
                dirs = [
                    (dir_i, dir_j, 0),  # 현재 대각선 방향
                    (dir_i, 0, 0),      # i축 방향
                    (0, dir_j, 0)       # j축 방향
                ]
            elif dir_i != 0 and dir_k != 0 and dir_j == 0:
                dirs = [
                    (dir_i, 0, dir_k),  # 현재 대각선 방향
                    (dir_i, 0, 0),      # i축 방향
                    (0, 0, dir_k)       # k축 방향
                ]
            elif dir_j != 0 and dir_k != 0 and dir_i == 0:
                dirs = [
                    (0, dir_j, dir_k),  # 현재 대각선 방향
                    (0, dir_j, 0),      # j축 방향
                    (0, 0, dir_k)       # k축 방향
                ]
        
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 3:
            dirs = [
                (dir_i, dir_j, dir_k),  # 현재 3D 대각선 방향
                (dir_i, dir_j, 0),      # xy 평면
                (dir_i, 0, dir_k),      # xz 평면
                (0, dir_j, dir_k),      # yz 평면
                (dir_i, 0, 0),          # x축
                (0, dir_j, 0),          # y축
                (0, 0, dir_k)           # z축
            ]
        
        for dir in dirs:
            if dir == (0, 0, 0):
                continue
            
            nxt_i, nxt_j, nxt_k = node.i + dir[0], node.j + dir[1], node.k + dir[2]
            if not self._is_valid(node, grids, *dir):
                continue
            canonical_neighbours.add(grids[nxt_i, nxt_j, nxt_k])
        return list(canonical_neighbours)

    def _get_forced_neighbors_all(self, node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> List[VoxelNode]:        
        if node.is_checked_forced_neighbours == True:
            return node.forced_neighbours
        else:
            node.is_checked_forced_neighbours = True    
        
        if abs(dir_i) + abs(dir_j) + abs(dir_k) == 1:
            node.forced_neighbours = self._get_forced_neighbor_orthogonal(node, parent, grids)   
            return node.forced_neighbours   
        elif abs(dir_i) + abs(dir_j) + abs(dir_k) == 2: 
            node.forced_neighbours = self._get_forced_neighbor_diagonal2d(node, parent, grids)
            return node.forced_neighbours   
        else:
            node.forced_neighbours = self._get_forced_neighbor_diagonal3d(node, parent, grids)
            return node.forced_neighbours

    def _get_forced_neighbors(self, cur_node: VoxelNode, 
                            parent: VoxelNode, 
                            grids: VoxelGrids3D,
                            block_dir: Tuple[int, int, int],
                            free_dirs: List[Tuple[int, int, int]],
                            forced_neighbors: Set[VoxelNode]) -> List[VoxelNode]:
        """
        Block_dir: (i, j, k) 는 parent 기준으로 block node의 방향을 나타냄 
        free_dirs: (i, j, k) 는 cur_node 기준으로 free node의 방향을 나타냄  
        """
        if self._is_valid(parent, grids, block_dir[0], block_dir[1], block_dir[2]) == True:
            block: VoxelNode = grids[parent.i + block_dir[0], parent.j + block_dir[1], parent.k + block_dir[2]]
            if block.is_obstacle == True:
                frees: Set[VoxelNode] = self._get_next_nodes(cur_node, 
                                                        grids,
                                                        free_dirs)
                forced_neighbors.update([free for free in frees if free.is_obstacle == False])
        return forced_neighbors

    def _get_forced_neighbor_orthogonal(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        
        if abs(di) == 1:
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, 0),
                                free_dirs=[(0, 1, 0), (0, 1, 1), (0, 1, -1), (di, 1, 0), (di, 1, 1), (di, 1, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, -1, 0),
                                free_dirs=[(0, -1, 0), (0, -1, 1), (0, -1, -1), (di, -1, 0), (di, -1, 1), (di, -1, -1)],
                                forced_neighbors=forced_neighbors)    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(0, 1, 1), (0, -1, 1), (0, 0, 1), (di, 1, 1), (di, -1, 1), (di, 0, 1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, -1),
                                free_dirs=[(0, 1, -1), (0, -1, -1), (0, 0, -1), (di, 1, -1), (di, -1, -1), (di, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, 1),
                                free_dirs=[(0, 1, 1), (di, 1, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 1, -1),
                                free_dirs=[(0, 1, -1), (di, 1, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, -1, 1),
                                free_dirs=[(0, -1, 1), (di, -1, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, -1),
                                free_dirs=[(0, -1, -1), (di, -1, -1)],
                                forced_neighbors=forced_neighbors)  
            
        if abs(dj) == 1:    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, 0, 0), (1, 0, 1), (1, 0, -1), (1, dj, 0), (1, dj, 1), (1, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, 0, 0), (-1, 0, 1), (-1, 0, -1), (-1, dj, 0), (-1, dj, 1), (-1, dj, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(1, 0, 1), (-1, 0, 1), (0, 0, 1), (1, dj, 1), (-1, dj, 1), (0, dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, -1),
                                free_dirs=[(1, 0, -1), (-1, 0, -1), (0, 0, -1), (1, dj, -1), (-1, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 1),
                                free_dirs=[(1, 0, 1), (1, dj, 1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 0, -1),
                                free_dirs=[(1, 0, -1), (1, dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 1),
                                free_dirs=[(-1, 0, 1), (-1, dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 0, -1),
                                free_dirs=[(-1, 0, -1), (-1, dj, -1)],
                                forced_neighbors=forced_neighbors)
        
        if abs(dk) == 1:    
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 0, dk), (1, 1, dk), (1, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, 0, 0), (-1, 1, 0), (-1, -1, 0), (-1, 0, dk), (-1, 1, dk), (-1, -1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, 0),
                                free_dirs=[(1, 1, 0), (-1, 1, 0), (0, 1, 0), (1, 1, dk), (-1, 1, dk), (0, 1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, 0),
                                free_dirs=[(1, -1, 0), (-1, -1, 0), (0, -1, 0), (1, -1, dk), (-1, -1, dk), (0, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 1, 0),
                                free_dirs=[(1, 1, 0), (1, 1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, -1, 0),
                                free_dirs=[(1, -1, 0), (1, -1, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 1, 0),
                                free_dirs=[(-1, 1, 0), (-1, 1, dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, -1, 0),
                                free_dirs=[(-1, -1, 0), (-1, -1, dk)],
                                forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)  

    def _get_forced_neighbor_diagonal2d(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:   
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k
        
        if abs(di) == 1 and abs(dj) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 0),
                                free_dirs=[(-di, dj, 0),
                                        (-di, dj, 1), (-di, 0, 1), (0, dj, 1), (0, 0, 1),
                                        (-di, dj, -1), (-di, 0, -1), (0, dj, -1), (0, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 0, 0),
                                free_dirs=[(di, -dj, 0), 
                                        (di, -dj, 1), (0, -dj, 1), (di, 0, 1), (0, 0, 1),
                                        (di, -dj, -1), (0, -dj, -1), (di, 0 , -1), (0, 0, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, 1),
                                free_dirs=[(0, -dj, 1), (-di, 0, 1), (0, 0, 1), (di, 0, 1), (0, dj, 1), (di, dj, 1), (-di, dj, 1), (di, -dj, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, -1),
                                free_dirs=[(-di, 0, -1), (0, -dj, -1), (0, 0, -1), (di, 0, -1), (0, dj, -1), (di, dj, -1), (-di, dj, -1), (di, -dj, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 1),
                                free_dirs=[(-di, dj, 1), (0, dj, 1), (0, 0, 1), (di, dj, 1), (di, 0, 1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, dj, -1),
                                free_dirs=[(-di, dj, -1), (0, dj, -1), (0, 0, -1), (di, dj, -1), (di, 0, -1)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 0, 1),
                                free_dirs=[(di, -dj, 1), (di, 0, -1), (0, 0, -1), (di, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(di, 0, -1),
                                free_dirs=[(di, -dj, -1), (di, 0, -1), (0, 0, -1), (di, dj, -1), (0, dj, -1)],
                                forced_neighbors=forced_neighbors)
        
        if abs(dj) == 1 and abs(dk) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, 0, dk),
                                free_dirs=[(0, -dj, dk),
                                        (1, -dj, dk), (1, -dj, 0), (1, 0, dk), (1, 0, 0),
                                        (-1, -dj, dk), (-1, -dj, 0), (-1, 0, dk), (-1, 0, 0)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(0, dj, 0),
                                free_dirs=[(0, dj, -dk), 
                                        (1, dj, -dk), (1, 0, -dk), (1, dj, 0), (1, 0, 0),
                                        (-1, dj, -dk), (-1, 0, -dk), (-1, dj, 0), (-1, 0, 0)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, 0, 0),
                                free_dirs=[(1, -dj, 0), (1, 0, -dk), (1, 0, 0), (1, dj, dk), (1, dj, 0), (1, 0, dk), (1, -dj, dk), (1, dj, -dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, 0, 0),
                                free_dirs=[(-1, -dj, 0), (-1, 0, -dk), (-1, 0, 0), (-1, dj, dk), (-1, dj, 0), (-1, 0, dk), (-1, -dj, dk), (-1, dj, -dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(1, 0, dk),
                                free_dirs=[(1, -dj, dk), (1, 0, dk), (1, 0, 0), (1, dj, 0), (1, dj, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(-1, 0, dk),
                                free_dirs=[(-1, -dj, dk), (-1, 0, dk), (-1, 0, 0), (-1, dj, 0), (-1, dj, dk)],
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(1, dj, 0),
                                free_dirs=[(1, dj, -dk), (1, dj, 0), (1, 0, 0), (1, 0, dk), (1, dj, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(-1, dj, 0),
                                free_dirs=[(-1, dj, -dk), (-1, dj, 0), (-1, 0, 0), (-1, 0, dk), (-1, dj, dk)],   
                                forced_neighbors=forced_neighbors)

        if abs(dk) == 1 and abs(di) == 1:   
            self._get_forced_neighbors(cur_node, parent, grids,
                                block_dir=(di, 0, 0),
                                free_dirs=[(di, 0, -dk),
                                        (di, 1, -dk), (0, 1, -dk), (di, 1, 0), (0, 1, 0),
                                        (di, -1, -dk), (0, -1, -dk), (di, -1, 0), (0, -1, 0)],   
                                forced_neighbors=forced_neighbors)         
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 0, dk),   
                                free_dirs=[(-di, 0, dk), 
                                        (-di, 1, dk), (-di, 1, 0), (0, 1, dk), (0, 1, 0),
                                        (-di, -1, dk), (-di, -1, 0), (0, -1, dk), (0, -1, 0)],     
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, 0),
                                free_dirs=[(0, 1, -dk), (-di, 1, 0), (0, 1, 0), (di, 1, dk), (0, 1, dk), (di, 1, 0), (-di, 1, dk), (di, 1, -dk)],
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, 0),   
                                free_dirs=[(-di, -1, 0), (0, -1, -dk), (0, -1, 0), (di, -1, dk), (0, -1, dk), (di, -1, 0), (-di, -1, dk), (di, -1, -dk)],      
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, 1, 0),
                                free_dirs=[(di, 1, -dk), (di, 1, 0), (0, 1, 0), (0, 1, dk), (di, 1, dk)],   
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(di, -1, 0),
                                free_dirs=[(di, -1, -dk), (di, -1, 0), (0, -1, 0), (0, -1, dk), (di, -1, dk)],  
                                forced_neighbors=forced_neighbors)  
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, 1, dk),
                                free_dirs=[(-di, 1, dk), (0, 1, 0), (0, 1, dk), (di, 1, 0), (di, 1, dk)],    
                                forced_neighbors=forced_neighbors)
            self._get_forced_neighbors(cur_node, parent, grids,    
                                block_dir=(0, -1, dk),
                                free_dirs=[(-di, -1, dk), (0, -1, 0), (0, -1, dk), (di, -1, 0), (di, -1, dk)],    
                                forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)   

    def _get_forced_neighbor_diagonal3d(self, cur_node: VoxelNode, parent: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]: 
        forced_neighbors = set()
        di, dj, dk = cur_node.i - parent.i, cur_node.j - parent.j, cur_node.k - parent.k    
        self._get_forced_neighbors(cur_node, parent, grids,    
                            block_dir=(di, dj, 0),
                            free_dirs=[(di, dj, -dk), (0, dj, -dk), (di, 0, -dk)],
                            forced_neighbors=forced_neighbors)      
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(di, 0, 0),
                            free_dirs=[(di, -dj, -dk), (0, 0, -dk), (di, 0, -dk), (0, dj, -dk), (di, dj, -dk),
                                    (di, -dj, 0), (0, -dj, 0)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids, 
                            block_dir=(0, dj, 0),
                            free_dirs=[(-di, dj, -dk), (0, 0, -dk), (0, dj, -dk), (di, 0, -dk), (di, dj, -dk),
                                    (-di, dj, 0), (-di, 0, 0)],  
                            forced_neighbors=forced_neighbors)        
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(0, 0, dk),
                            free_dirs=[(-di, 0, 0), (-di, dj, 0), (0, -dj, 0), (di, -dj, 0), 
                                    (-di, -dj, dk), (0, -dj, dk), (di, -dj, dk), (-di, 0, dk), (-di, dj, dk)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids,    
                            block_dir=(0, dj, dk),
                            free_dirs=[(-di, dj, 0),
                                    (-di, dj, dk), (-di, 0, dk)],
                            forced_neighbors=forced_neighbors)
        self._get_forced_neighbors(cur_node, parent, grids,
                            block_dir=(di, 0, dk),
                            free_dirs=[(di, -dj, 0),
                                    (di, -dj, dk), (0, -dj, dk)],   
                            forced_neighbors=forced_neighbors)  
        return list(forced_neighbors)   

    def _is_valid(self, node: VoxelNode, grids: VoxelGrids3D, dir_i: int, dir_j: int, dir_k: int) -> bool:
        nxt_i, nxt_j, nxt_k = node.i + dir_i, node.j + dir_j, node.k + dir_k
        return nxt_i in self.valid_range and nxt_j in self.valid_range and nxt_k in self.valid_range
    
    def _get_next_nodes(self, cur_node: VoxelNode, grids: VoxelGrids3D, directions: Tuple[int, int, int]) -> Set[VoxelNode]:
        nxt_nodes = set()
        for direction in directions:
            dir_i, dir_j, dir_k = direction 
            if self._is_valid(cur_node, grids, dir_i, dir_j, dir_k) == False:  
                continue
            nxt_nodes.add(grids[cur_node.i + dir_i, cur_node.j + dir_j, cur_node.k + dir_k])
        return nxt_nodes

    def _get_neighbors(self, node: VoxelNode, grids: VoxelGrids3D) -> List[VoxelNode]:
        neighbors = []
        neighbor_positions = self._neighbor_dirs + np.array([node.i, node.j, node.k])  # numpy 연산 활용
        valid_mask = np.all((neighbor_positions >= 0) & (neighbor_positions < grids.map_size), axis=1)
        for pos in neighbor_positions[valid_mask]:
            neighbors.append(grids[pos[0], pos[1], pos[2]])
        return neighbors

