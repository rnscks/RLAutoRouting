import gymnasium as gym
import torch
from stable_baselines3.common.env_checker import check_env  
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from typing import List, Tuple, Set, Dict
import json
import pandas as pd 

from src.algorithm.rl.env import PathFindingEnv, CurriculumPathFindingEnv
from src.algorithm.rl.feature_extractor import PathFindingFeatureExtractor, NormalExtractor
from src.cable_routing.routing_component.panel import Panel
from src.algorithm.rl.util.callback import CurriculumCallBack
from read_file import read_panel_bnd, read_stp_file, read_routing_json


def test_random_action(env: gym.Env) -> None:
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)
        if done:
            env.reset()
    return  

def run_maskable_learning_process(
    sections: List[int] = [2, 2, 3],
    map_size: int = 10,
    max_step: int = 500,
    total_timesteps:int = 640_000,
    len_boundary: Tuple[int, int] = (0, 100),
    turning_boundary: Tuple[int, int] = (0, 100),
    model_name: str = '') -> None:    
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
    env = ActionMasker(env, env.valid_action_mask)
    
    try:
        check_env(env)
        test_random_action(env) 
    except Exception as e:
        raise e
        
    if model_name == '':
        policy_kwargs = dict(
            features_extractor_class=NormalExtractor,
            activation_fn = torch.nn.ReLU, 
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
    else:
        model = MaskablePPO.load(model_name, env=env)
        
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=4, 
        tb_log_name=f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}")
    model.save(f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}.zip")
    return

def run_curriculum_learning_process(
    map_size: int = 10,
    max_step: int = 500,
    total_timesteps:int = 640_000,
    len_boundary: Tuple[int, int] = (5, 100),
    turning_boundary: Tuple[int, int] = (1, 100),
    model_name: str = '') -> None:    
    routing_jsons_files: List[str] = [
        'data/패널_1번_관련_파일_정보.json',
        'data/패널_2번_관련_파일_정보.json',
        'data/패널_3번_관련_파일_정보.json']
    routing_jsons = [read_routing_json(routing_json) for routing_json in routing_jsons_files]
        
    env = CurriculumPathFindingEnv(
        available_areas=[
            read_panel_bnd(routing_jsons[0]['excel_file']),
            read_panel_bnd(routing_jsons[1]['excel_file']),
            read_panel_bnd(routing_jsons[2]['excel_file'])], 
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
    
    if model_name == '':
        policy_kwargs = dict(
            features_extractor_class=NormalExtractor,
            activation_fn = torch.nn.ReLU, 
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
    else:
        model = MaskablePPO.load(model_name, env=env)
    
    curriculum_callback = CurriculumCallBack(   
        log=f'M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}')
    model.learn(
        callback=curriculum_callback,
        total_timesteps=total_timesteps, 
        log_interval=4, 
        tb_log_name=f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}")
    model.save(f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}.zip")
    return

def run_maskable_learning_process_mlp(
    sections: List[int] = [2, 2, 3],
    map_size: int = 10,
    max_step: int = 500,
    total_timesteps:int = 640_000,
    len_boundary: Tuple[int, int] = (0, 100),
    turning_boundary: Tuple[int, int] = (0, 100),
    model_name: str = '') -> None:    
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
        
    if model_name == '':
        policy_kwargs = dict(
            features_extractor_class=PathFindingFeatureExtractor,
            activation_fn = torch.nn.ReLU, 
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
    else:
        model = MaskablePPO.load(model_name, env=env)
        
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=4, 
        tb_log_name=f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}_mlp")
    model.save(f"M{map_size}_S{max_step}_l{len_boundary[0]}_{len_boundary[1]}_mlp.zip")
    return


if __name__ == "__main__":
    run_maskable_learning_process(
        model_name='',
        map_size=10, 
        total_timesteps=100_000,
        max_step=500)