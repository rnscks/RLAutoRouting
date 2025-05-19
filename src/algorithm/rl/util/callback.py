from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from collections import deque
from typing import List, Tuple, Dict, cast
import numpy as np
import os

from src.cable_routing.routing_component.panel import Panel

class CurriculumCallBack(BaseCallback):
    def __init__(self,
                log: str = "test",
                verbose: int = 1) -> None:
        super().__init__(verbose)
        
        self.log = f"{log}_1"
        self.current_level: int = 1
        
        if os.path.exists(f'rl_models/{self.log}'):
            for i in range(2, 100):
                if not os.path.exists(f'rl_models/{log}_{i}'):
                    self.log = f'{log}_{i}'
                    break
        if not os.path.exists(f'rl_models/{self.log}'):
            os.makedirs(f'rl_models/{self.log}')  
        
        
    def _on_step(self) -> bool:
        return super()._on_step()
        
    def _on_rollout_end(self) -> None:
        env = self.training_env.envs[0]

        self.logger.record("train/level", env.level)    
        if len(env.cur_ep_rewards) >= 100:   
            sum_rewards = []  
            for idx in range(-1, -101, -1):
                sum_rewards.append(np.sum(env.cur_ep_rewards[idx])) 
            mean_ep_reward = np.mean(sum_rewards)    
            self.logger.record("train/cur_ep_rew_mean", mean_ep_reward)
        # 현재 레벨 관련 정보 접근
        current_level = env.level
        if current_level != self.current_level:
            self.current_level = current_level
            self._save_model()  
        return
    
    def _save_model(self) -> None:
        model: MaskablePPO = self.model
        model.save(f"rl_models/{self.log}/level_{self.current_level}.zip")  
        return  
    

