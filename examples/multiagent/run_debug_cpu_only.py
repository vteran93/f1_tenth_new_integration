import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import gymnasium as gym
import os

# FORZAR CPU - Bloquear completamente CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RAY_DISABLE_CUDA"] = "1" 
os.environ["TORCH_CUDA_MEMORY_FRACTION"] = "0"

def main():
    """
    Versión que FUERZA el uso de CPU para evitar problemas CUDA
    """
    
    # Verificar que CUDA esté deshabilitado
    try:
        import torch
        print(f"🔍 CUDA disponible: {torch.cuda.is_available()}")
        print(f"🔍 Dispositivo PyTorch: {torch.device('cpu')}")
    except:
        pass
    
    # Inicializar Ray forzando CPU
    ray.init(
        ignore_reinit_error=True,
        local_mode=False,  # Usar distribución pero local
        num_cpus=4,  # Limitar CPUs
        num_gpus=0,  # CERO GPUs
        object_store_memory=1_000_000_000,  # 1GB
        configure_logging=False,
        log_to_driver=False,
    )
    
    print("🚀 Ray inicializado en modo CPU únicamente")
    
    try:
        # Configuración SAC forzando CPU
        config = (
            SACConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .env_runners(
                num_env_runners=1,  # Solo 1 worker
                num_envs_per_env_runner=1,
            )
            .training(
                train_batch_size=64,  # Batch pequeño
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 5000,  # Buffer muy pequeño
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha_lr=3e-4,
                target_network_update_freq=1,
                tau=0.005,
                # Configuración adicional para estabilidad
                num_steps_sampled_before_learning_starts=500,  # Menos steps antes de entrenar
            )
            .resources(
                # FORZAR CPU completamente
                num_gpus=0,
                num_cpus_per_env_runner=1,
                num_cpus_for_main_process=1,
                num_gpus_per_env_runner=0,
                num_gpus_per_learner=0,
            )
            .debugging(seed=42)
        )
        
        print("✅ Configuración SAC creada (CPU only)")
        
        # Ejecutar entrenamiento corto
        results = tune.run(
            "SAC",
            config=config.to_dict(),
            stop={
                "timesteps_total": 10000,  # Muy corto
                "training_iteration": 20,   # Máximo 20 iteraciones
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
                num_to_keep=1,
                checkpoint_frequency=10,
            ),
            storage_path="/tmp/ray_results_cpu_only",
            name="sac_cartpole_cpu_only",
            verbose=1,
            max_failures=0,  # No permitir fallos
            # Recursos limitados por trial
            resources_per_trial={
                "cpu": 2,
                "gpu": 0,  # CERO GPU
            },
        )
        
        print("🎉 Entrenamiento completado exitosamente!")
        
        # Mostrar resultados básicos
        best_trial = results.get_best_trial("env_runners/episode_return_mean", "max")
        if best_trial:
            final_metrics = best_trial.last_result
            print(f"📈 Reward final: {final_metrics.get('env_runners/episode_return_mean', 0):.2f}")
            print(f"📊 Timesteps: {final_metrics.get('timesteps_total', 0)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()
        print("✅ Ray cerrado")

if __name__ == "__main__":
    main()
