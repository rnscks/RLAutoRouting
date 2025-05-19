import random
from typing import List, Tuple, Dict, Any
import os
from imitation.data.types import Trajectory
from imitation.data.types import DictObs
import numpy as np  
from tqdm import tqdm
from src.algorithm.rl.env import PathFindingEnv
from src.algorithm.pathfinding.pathfinding import AStar
from src.datastruct.voxel_grids import VoxelGrids3D, VoxelNode
from initialize_env import init_pathfinding_env, init_maskable_ppo_model, init_maskable_ppo_model_mlp

def get_trajectories(env: PathFindingEnv, n_sample: int) -> Trajectory:
    trajs = []
    for _ in tqdm(range(n_sample)):
        obs, _ = env.reset()
        grid: VoxelGrids3D = env.grids
        free_nodes = [node for node in grid if node.is_obstacle == False]   
        obs_list, act_list = [], []

        grid.reset()
        start_node = random.choice(free_nodes)
        goal_node = random.choice(free_nodes)
        if start_node == goal_node: 
            continue    
        
        grid.set_goal_node(goal_node)
        grid.set_start_node(start_node)
        ast = AStar()
        ast.search(grid)
        path: List[VoxelNode] = ast.get_path_nodes(grid)
        
        if len(path) <= 2:  
            continue    
        
        for i in range(len(path) - 2):  
            state = env.get_observation(path[i])    
            action = env.get_action(path[i], path[i + 1]) 
            obs_list.append(state)  
            act_list.append(action)
        
        obs_list.append(env.get_observation(path[-1]))
        # ndarray로 변환
        # obs_array = np.array(obs_list)       # shape: (T+1, obs_dim)
        obs_array = DictObs.from_obs_list(obs_list)  # shape: (T+1, obs_dim)
        acts_array = np.array(act_list)      # shape: (T,)
        traj = Trajectory(obs=obs_array,
                        acts=acts_array,
                        infos=None,
                        terminal=True)
        trajs.append(traj)  
        
    return trajs 


if __name__ == "__main__":
    import glob
    from imitation.algorithms.bc import BC
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
        
    def run_bc_process(trajs: List[Trajectory], model_path: str = 'bc_trained_ppo_model', map_size: int = 10):
        def make_env():
            return init_pathfinding_env(map_size=map_size, max_step=500)
        
        env = DummyVecEnv([make_env, make_env, make_env])
        model = init_maskable_ppo_model(env)
        
        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            policy=model.policy,    
            demonstrations=trajs,
            rng=0)
        
        import time
        start_time = time.time()    
        bc_trainer.train(n_epochs=10)    
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        print(f"After BC: {evaluate_policy(bc_trainer.policy, env, n_eval_episodes=100, render=False, deterministic=False)}")
        
        trained_env = init_pathfinding_env(map_size=map_size, max_step=500)   


        model = init_maskable_ppo_model(trained_env)    
        
        model.policy.load_state_dict(bc_trainer.policy.state_dict())    
        # model.learn(total_timesteps=200_000, tb_log_name=model_path)
        model.save(model_path)
        return
    
    def run_bc_process_mlp(trajs: List[Trajectory], model_path: str = 'bc_trained_ppo_model_mlp', map_size: int = 10):
        def make_env():
            return init_pathfinding_env(map_size=map_size, max_step=500)
        
        env = DummyVecEnv([make_env, make_env, make_env])
        model = init_maskable_ppo_model_mlp(env)
        
        bc_trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            policy=model.policy,    
            demonstrations=trajs,
            rng=0)
        
        
        import time
        start_time = time.time()    
        bc_trainer.train(n_epochs=10)    
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        print(f"After BC: {evaluate_policy(bc_trainer.policy, env, n_eval_episodes=100, render=False, deterministic=False)}")
        
        trained_env = init_pathfinding_env(map_size=map_size, max_step=500)   


        model = init_maskable_ppo_model_mlp(trained_env)    
        
        model.policy.load_state_dict(bc_trainer.policy.state_dict())
        model.learn(total_timesteps=200_000, tb_log_name=model_path)
        model.save(model_path)
        return
    
    def save_trajectory(trajs: List[Trajectory], file_name: str = 'trajectories.pkl'):
        from imitation.data import serialize
        import os
        dir_path = 'data/궤적'
        file_path = os.path.join(dir_path, file_name)  
        serialize.save(file_path, trajs)
            
    def load_trajectory(file_name: str) -> List[Trajectory]: 
        dir_path = 'data/궤적'   
        
        import os
        from imitation.data import serialize    
        file_path = os.path.join(dir_path, file_name)                  
        return serialize.load(file_path)    
    
    def save_trajectory_data(trajs: List[Trajectory], save_path="data/궤적/map30"):
        os.makedirs(save_path, exist_ok=True)

        for idx, traj in enumerate(trajs):
            # 각 trajectory의 관측값과 행동을 저장
            obs_dict = traj.obs  # DictObs
            actions = traj.acts  # np.ndarray of shape (T,)

            # DictObs 내부 딕셔너리 구조로 변환 (예: {"sensor": [...], "panel_config": [...], ...})
            obs_np = {k: np.array(v) for k, v in obs_dict.items()}  # T+1, dim

            # 저장
            np.savez(os.path.join(save_path, f"traj_{idx}.npz"), **obs_np, actions=actions)

    def load_npz_as_trajectory(npz_path: str) -> Trajectory:
        data = np.load(npz_path)
        
        # actions 추출
        actions = data["actions"]
        
        # 관측값 딕셔너리 구성
        obs_dict = {k: data[k] for k in data.files if k != "actions"}
        
        # DictObs로 변환 (T+1, obs_dim)
        obs = DictObs(obs_dict)
        
        # Trajectory 구성 (infos, terminal은 None, True로 고정)
        traj = Trajectory(obs=obs, acts=actions, infos=None, terminal=True)
        return traj
    
    def load_all_trajectories_from_dir(folder_path: str):
        traj_files = sorted(glob.glob(f"{folder_path}/traj_*.npz"))
        return [load_npz_as_trajectory(fp) for fp in traj_files]

    # sampled_env = init_pathfinding_env(map_size=30, max_step=500)
    # trajs: List[Trajectory] = get_trajectories(sampled_env, n_sample=1000)
    # save_trajectory_data(trajs, 'data/궤적/map30')
    trajs = load_all_trajectories_from_dir('data/궤적/map10')
    # trajs: List[Trajectory] = load_trajectory('map10.pkl') 
    # run_bc_process(trajs, model_path='map10_bc_trained_ppo_20', map_size=10)
    run_bc_process_mlp(trajs, model_path='map30_bc_trained_ppo', map_size=10)