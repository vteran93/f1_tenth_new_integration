from abc import ABC, abstractmethod
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
# from ray.rllib.algorithms.td3 import TD3Config  # Comentado debido a módulo faltante
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

class RLAlgorithm(ABC):
    """Clase base abstracta para algoritmos de aprendizaje por refuerzo."""

    def __init__(self, config, env_name, policies):
        self.config = config
        self.env_name = env_name
        self.policies = policies
        self.algo_config = None
        self.algo = None

    @abstractmethod
    def setup_config(self):
        """Configura el algoritmo RLlib con parámetros del archivo config."""
        pass

    def build(self, logger_creator):
        """Construye el algoritmo con el logger especificado."""
        self.algo = self.algo_config.build(logger_creator=logger_creator)
        return self.algo

    def train(self):
        """Entrena el algoritmo por una iteración."""
        return self.algo.train()

    def save(self, checkpoint_dir):
        """Guarda un checkpoint del algoritmo."""
        return self.algo.save(checkpoint_dir)

    def restore(self, checkpoint_path):
        """Restaura el algoritmo desde un checkpoint."""
        self.algo.restore(checkpoint_path)

    def compute_single_action(self, observation, policy_id, explore=False):
        """Calcula una acción para una observación dada."""
        return self.algo.compute_single_action(observation, policy_id=policy_id, explore=explore)

    def stop(self):
        """Para el algoritmo y libera recursos."""
        self.algo.stop()

class PPOAlgorithm(RLAlgorithm):
    """Implementación del algoritmo PPO (Proximal Policy Optimization)."""

    def __init__(self, config, env_name, policies):
        super().__init__(config, env_name, policies)
        self.ppo_config = config.get("algorithm", {}).get("ppo", {})

    def setup_config(self):
        """Configura PPO con parámetros del archivo config."""
        self.algo_config = (PPOConfig()
                            .environment(self.env_name, env_config=self.config.get("environment", {}))
                            .framework("torch")
                            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                            .env_runners(num_env_runners=0)
                            .multi_agent(
                                policies=self.policies,
                                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
                            )
                            .training(
                                lr=self.ppo_config.get("learning_rate", 0.0003),
                                gamma=self.ppo_config.get("gamma", 0.99),
                                train_batch_size=self.ppo_config.get("train_batch_size", 4000)
                            ))
        return self.algo_config

class SACAlgorithm(RLAlgorithm):
    """Implementación del algoritmo SAC (Soft Actor-Critic)."""

    def __init__(self, config, env_name, policies):
        super().__init__(config, env_name, policies)
        self.sac_config = config.get("algorithm", {}).get("sac", {})

    def setup_config(self):
        """Configura SAC con parámetros del archivo config, incluyendo replay buffer."""
        replay_buffer_config = self.config.get("training", {}).get("replay_buffer_config", {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "capacity": 100000
        })
        
        self.algo_config = (SACConfig()
                            .environment(self.env_name, env_config=self.config.get("environment", {}))
                            .framework("torch")
                            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                            .env_runners(num_env_runners=0)
                            .multi_agent(
                                policies=self.policies,
                                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
                            )
                            .training(
                                lr=self.sac_config.get("learning_rate", 0.0003),
                                gamma=self.sac_config.get("gamma", 0.99),
                                train_batch_size=self.sac_config.get("train_batch_size", 256),
                                tau=self.sac_config.get("tau", 0.005),
                                replay_buffer_config=replay_buffer_config
                            ))
        return self.algo_config

class DDPGAlgorithm(RLAlgorithm):
    """Implementación del algoritmo DDPG (Deep Deterministic Policy Gradient)."""
    
    def __init__(self, config, env_name, policies):
        super().__init__(config, env_name, policies)
        self.ddpg_config = config.get("algorithm", {}).get("ddpg", {})

    def setup_config(self):
        """Configura DDPG con parámetros del archivo config."""
        raise NotImplementedError("DDPG is not yet implemented.")

class TD3Algorithm(RLAlgorithm):
    """Implementación del algoritmo TD3 (Twin Delayed DDPG)."""

    def __init__(self, config, env_name, policies):
        super().__init__(config, env_name, policies)
        self.td3_config = config.get("algorithm", {}).get("td3", {})

    def setup_config(self):
        """Configura TD3 con parámetros del archivo config."""
        raise NotImplementedError("TD3Config is not available due to missing module.")

def get_algorithm(config, env_name, policies):
    """Función fábrica para crear el algoritmo RL especificado."""
    algo_type = config.get("algorithm", {}).get("type")
    algo_map = {
        "ppo": PPOAlgorithm,
        "sac": SACAlgorithm,
        "ddpg": DDPGAlgorithm,
        "td3": TD3Algorithm
    }
    algo_class = algo_map.get(algo_type)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm type: {algo_type}. Supported: {list(algo_map.keys())}")
    return algo_class(config, env_name, policies)