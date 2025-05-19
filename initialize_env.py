import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import PPO   
from stable_baselines3.common.env_checker import check_env  
from typing import List, Tuple, Set, Dict
import torch
from src.algorithm.rl.env import PathFindingEnv
from src.algorithm.rl.feature_extractor import PathFindingFeatureExtractor, NormalExtractor
from read_file import read_panel_bnd, read_stp_file, read_routing_json


def test_random_action(env: gym.Env) -> None:
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        if done:
            env.reset()
    return  

def init_pathfinding_env(
    sections: List[int] = [2, 2, 3],
    map_size: int = 10,
    max_step: int = 500,
    len_boundary: Tuple[int, int] = (0, 100),
    turning_boundary: Tuple[int, int] = (0, 100)) -> PathFindingEnv:    
    routing_jsons_files: List[str] = [
        'data/패널_1번_관련_파일_정보.json',
        'data/패널_2번_관련_파일_정보.json',
        'data/패널_3번_관련_파일_정보.json']
    routing_jsons = [read_routing_json(routing_json) for routing_json in routing_jsons_files]
        
    env = PathFindingEnv(
        available_areas=[
            read_panel_bnd(routing_jsons[0]['excel_file']),
            read_panel_bnd(routing_jsons[1]['excel_file']),
            read_panel_bnd(routing_jsons[2]['excel_file'])], 
        sections=sections,
        len_bounary=len_boundary,
        turning_boundary=turning_boundary,  
        map_size=map_size,
        max_step=max_step)
    
    env = ActionMasker(env, lambda s: env.valid_action_mask())   
    
    try:
        check_env(env)
        test_random_action(env) 
    except Exception as e:
        raise e
    return env

def init_maskable_ppo_model(
    env: PathFindingEnv) -> MaskablePPO:
    policy_kwargs = dict(
        features_extractor_class=NormalExtractor,
        activation_fn = torch.nn.LeakyReLU, 
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    model = MaskablePPO(
                policy = MaskableActorCriticPolicy, 
                env = env, 
                verbose=1, 
                device="cuda", 
                batch_size= 1024, 
                learning_rate= 0.00005,
                gae_lambda=0.95,
                gamma = 0.95, 
                ent_coef= 5.0e-3, 
                n_steps=10240, 
                clip_range= 0.1, 
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs/")
    return model    



def init_maskable_ppo_model_mlp(
    env: PathFindingEnv) -> MaskablePPO:
    policy_kwargs = dict(
        features_extractor_class=PathFindingFeatureExtractor,
        activation_fn = torch.nn.LeakyReLU, 
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    model = MaskablePPO(
                policy = MaskableActorCriticPolicy, 
                env = env, 
                verbose=1, 
                device="cuda", 
                batch_size= 1024, 
                learning_rate= 0.00005,
                gae_lambda=0.95,
                gamma = 0.95, 
                ent_coef= 5.0e-3, 
                n_steps=10240, 
                clip_range= 0.1, 
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs/")
    return model    


def init_ppo_model(
    env: PathFindingEnv) -> MaskablePPO:
    model = PPO(
                policy = "MlpPolicy", 
                env = env, 
                verbose=1, 
                device="cuda", 
                batch_size= 1024, 
                learning_rate= 0.00005,
                gae_lambda=0.95,
                gamma = 0.95, 
                ent_coef= 5.0e-3, 
                n_steps=10240, 
                clip_range= 0.1, 
                tensorboard_log="logs/")
    
    return model    