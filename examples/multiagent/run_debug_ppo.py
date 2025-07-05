import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
import os

# FORZAR CPU completamente
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RAY_DISABLE_CUDA"] = "1"

def main():
    """
    Usar PPO en lugar de SAC - más estable
    """
    print("🧪 Probando PPO (más estable que SAC)...")
    
    # Ray en modo local para máxima estabilidad
    ray.init(
        local_mode=True,  # Modo completamente local
        ignore_reinit_error=True,
    )
    
    try:
        # PPO es más estable que SAC
        config = (
            PPOConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .training(
                train_batch_size=512,
                lr=5e-4,
            )
            .env_runners(
                num_env_runners=0,  # Sin workers remotos
                num_envs_per_env_runner=1,
                rollout_fragment_length="auto",  # Auto para evitar conflictos
            )
            .resources(num_gpus=0)
            .debugging(seed=42)
        )
        
        print("✅ Configuración PPO creada")
        
        results = tune.run(
            "PPO",
            config=config.to_dict(),
            stop={
                "timesteps_total": 20000,
                "episode_reward_mean": 400,
            },
            storage_path="/tmp/ray_results_ppo",
            name="ppo_cartpole_stable",
            verbose=1,
        )
        
        print("🎉 ¡PPO completado exitosamente!")
        
        best_trial = results.get_best_trial("episode_reward_mean", "max")
        if best_trial:
            final_metrics = best_trial.last_result
            print(f"📈 Reward final: {final_metrics.get('episode_reward_mean', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Error con PPO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
