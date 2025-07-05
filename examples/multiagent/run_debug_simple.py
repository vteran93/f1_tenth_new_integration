import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Versión ULTRA SIMPLE para debugging - mínima configuración
    """
    
    # Configuración mínima de Ray
    ray.init(
        local_mode=True,  # Modo local sin distribución
        ignore_reinit_error=True,
        log_to_driver=False
    )
    
    logger.info("🚀 Ray en modo local iniciado")
    
    try:
        # Configuración MÍNIMA de SAC
        config = (
            SACConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .env_runners(
                num_env_runners=0,  # SIN workers remotos
                create_local_env_runner=True,
            )
            .training(
                train_batch_size=64,  # Muy pequeño
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 10000,  # Muy pequeño
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
                lr=3e-4,
            )
            .debugging(seed=42)
        )
        
        # Entrenamiento ULTRA corto
        results = tune.run(
            "SAC",
            config=config.to_dict(),
            stop={"timesteps_total": 5000},  # MUY CORTO
            storage_path="/tmp/ray_debug_simple",
            name="sac_simple_test",
            verbose=2,
        )
        
        logger.info("✅ Entrenamiento simple completado!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
