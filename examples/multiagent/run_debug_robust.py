import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import gymnasium as gym
import os
import tempfile
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Script mejorado para entrenar SAC en CartPole-v1 con mejor manejo de errores
    """
    
    # 1. Limpiar cualquier sesión previa de Ray
    try:
        ray.shutdown()
        logger.info("🔄 Sesión previa de Ray cerrada")
    except:
        pass
    
    # 2. Configuración más conservadora de Ray
    logger.info("🚀 Inicializando Ray con configuración robusta...")
    
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,  # Reduce verbosidad
        configure_logging=False,  # Evita conflictos de logging
        local_mode=False,  # Usa modo distribuido pero local
        # Configuración de recursos más conservadora
        num_cpus=min(8, os.cpu_count()),  # Limita a 8 CPUs máximo
        object_store_memory=2_000_000_000,  # 2GB para object store
        # Configuración de red más estable
        _temp_dir=tempfile.mkdtemp(),  # Directorio temporal único
    )
    
    logger.info(f"✅ Ray inicializado correctamente")
    logger.info(f"📊 Dashboard: {ray.get_dashboard_url()}")
    
    try:
        # 3. Configuración más robusta del agente SAC
        logger.info("🤖 Configurando agente SAC...")
        
        config = (
            SACConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            # Configuración más conservadora de workers
            .env_runners(
                num_env_runners=2,  # Reducido para mayor estabilidad
                num_envs_per_env_runner=1,
                rollout_fragment_length="auto",
                enable_connectors=True,
            )
            .training(
                # Configuración SAC optimizada
                train_batch_size=128,  # Reducido para menor uso de memoria
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 25000,  # Reducido para menor uso de memoria
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
                lr=3e-4,
                target_network_update_freq=1,
                tau=0.005,
                # Configuración adicional para estabilidad
                num_steps_sampled_before_learning_starts=1000,
                training_intensity=None,
            )
            # Configuración de recursos por worker
            .resources(
                num_gpus=0,  # Forzar CPU para evitar problemas GPU
                num_cpus_per_env_runner=1,
                num_cpus_for_main_process=1,
            )
            # Configuración de debugging y checkpointing más robusta
            .debugging(
                seed=42,
                log_level="ERROR",  # Reduce verbosidad
            )
            .checkpointing(
                export_native_model_files=True,
            )
        )
        
        logger.info("✅ Configuración del agente creada")
        
        # 4. Configuración de almacenamiento más robusta
        storage_path = "/tmp/ray_results_robust"
        os.makedirs(storage_path, exist_ok=True)
        
        # 5. Configuración de parada más conservadora
        stop_conditions = {
            "timesteps_total": 25000,  # Reducido para prueba más rápida
            "env_runners/episode_return_mean": 400,
            "training_iteration": 100,  # Límite de iteraciones por seguridad
        }
        
        logger.info("🎯 Iniciando entrenamiento con configuración robusta...")
        logger.info(f"📊 Condiciones de parada: {stop_conditions}")
        
        # 6. Ejecutar entrenamiento con manejo de errores
        results = tune.run(
            "SAC",
            config=config.to_dict(),
            stop=stop_conditions,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
                num_to_keep=2,
                checkpoint_frequency=5,  # Checkpoints más frecuentes
                checkpoint_at_end=True
            ),
            storage_path=storage_path,
            name="sac_cartpole_robust",
            verbose=1,
            # Configuración adicional para robustez
            resume="AUTO+ERRORED",  # Reanudar si hay errores
            max_failures=3,  # Permitir hasta 3 fallos
            # Configuración de recursos para el trial
            resources_per_trial={
                "cpu": 4,  # Límite de CPUs por trial
                "memory": 4_000_000_000,  # 4GB límite de memoria
            },
        )
        
        logger.info("🎉 Entrenamiento completado exitosamente!")
        
        # 7. Mostrar resultados
        best_trial = results.get_best_trial("env_runners/episode_return_mean", "max")
        if best_trial:
            final_metrics = best_trial.last_result
            logger.info("📈 Resultados finales:")
            logger.info(f"  - Timesteps: {final_metrics.get('timesteps_total', 'N/A')}")
            logger.info(f"  - Reward promedio: {final_metrics.get('env_runners/episode_return_mean', 'N/A'):.2f}")
            logger.info(f"  - Iteraciones: {final_metrics.get('training_iteration', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {str(e)}")
        logger.error("💡 Posibles soluciones:")
        logger.error("  1. Reducir num_env_runners a 1")
        logger.error("  2. Usar local_mode=True en ray.init()")
        logger.error("  3. Reducir train_batch_size")
        logger.error("  4. Aumentar la memoria disponible")
        raise
    
    finally:
        # 8. Limpieza garantizada
        logger.info("🧹 Limpiando recursos...")
        try:
            ray.shutdown()
            logger.info("✅ Ray cerrado correctamente")
        except Exception as e:
            logger.warning(f"⚠️ Error al cerrar Ray: {e}")

if __name__ == "__main__":
    main()
