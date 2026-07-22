import ray
from ray.rllib.algorithms.sac import SAC, SACConfig
import gymnasium as gym

def main():
    """
    Versión BÁSICA sin Ray Tune - solo algoritmo directo
    """
    print("🧪 Probando SAC básico sin Ray Tune...")
    
    # Sin Ray distribuido - solo local
    ray.init(local_mode=True, ignore_reinit_error=True)
    
    try:
        # Configuración SAC básica
        config = (
            SACConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .training(
                train_batch_size=32,
                actor_lr=0.001,
                critic_lr=0.001,
                alpha_lr=0.001,
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 10000,
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
            )
            .env_runners(num_env_runners=0)
        )
        
        print("✅ Configuración creada")
        
        # Crear algoritmo directamente
        algo = config.build()
        print("✅ Algoritmo creado")
        
        # Entrenar por pocas iteraciones
        for i in range(5):
            result = algo.train()
            print(f"Iteración {i+1}: Reward = {result.get('env_runners/episode_return_mean', 0):.2f}")
        
        print("🎉 ¡Entrenamiento básico completado!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
