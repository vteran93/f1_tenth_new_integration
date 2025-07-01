#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de métricas de f1tenth_benchmarks.
"""

from callbacks import BenchmarkDataCollectorCallback, BenchmarkMetricsSaver, create_summary_report
import sys
import os
from pathlib import Path
import logging

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_callbacks_import():
    """Probar que los callbacks se pueden importar correctamente."""
    try:
        logger.info("Testing callback imports...")

        # Crear instancias para verificar que funcionan
        data_collector = BenchmarkDataCollectorCallback()
        logger.info("✓ BenchmarkDataCollectorCallback created successfully")

        test_config = {
            'name': 'test_experiment',
            'storage_path': './test_results'
        }
        metrics_saver = BenchmarkMetricsSaver(test_config)
        logger.info("✓ BenchmarkMetricsSaver created successfully")

        logger.info("✓ All callbacks imported and instantiated successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Error importing callbacks: {e}")
        return False


def test_benchmarks_path():
    """Verificar que el path a f1tenth_benchmarks existe."""
    # Test local f1tenth_benchmarks module
    try:
        import f1tenth_benchmarks
        logger.info(f"✓ f1tenth_benchmarks module available locally")

        # Test local module components
        from f1tenth_benchmarks import DataProcessor, BenchmarkAnalyzer
        logger.info(f"✓ f1tenth_benchmarks components importable")

        return True
    except ImportError as e:
        logger.error(f"✗ f1tenth_benchmarks not available: {e}")
        return False
        key_files = [
            "f1tenth_benchmarks/data_tools/plot_trajectory_analysis.py",
            "f1tenth_benchmarks/data_tools/plot_raceline_tracking.py",
            "f1tenth_benchmarks/benchmark_results/"
        ]

        missing_files = []
        for file_path in key_files:
            full_path = benchmarks_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            logger.warning(f"Some key files missing: {missing_files}")
        else:
            logger.info("✓ All key benchmark files found")

        return True
    else:
        logger.error(f"✗ f1tenth_benchmarks not found at: {benchmarks_path}")
        return False


def test_dependencies():
    """Verificar que las dependencias necesarias están disponibles."""
    required_modules = [
        'ray',
        'numpy',
        'pandas',
        'matplotlib',
        'yaml'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module} available")
        except ImportError:
            missing_modules.append(module)
            logger.error(f"✗ {module} not available")

    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return False

    logger.info("✓ All required dependencies available")
    return True


def test_ray_integration():
    """Probar la integración básica con Ray."""
    try:
        import ray
        from ray.tune.callback import Callback

        # Verificar que nuestro callback hereda correctamente
        assert issubclass(BenchmarkMetricsSaver, Callback)
        logger.info("✓ BenchmarkMetricsSaver correctly inherits from ray.tune.Callback")

        # Verificar imports de RLLib
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
        assert issubclass(BenchmarkDataCollectorCallback, DefaultCallbacks)
        logger.info("✓ BenchmarkDataCollectorCallback correctly inherits from DefaultCallbacks")

        return True

    except Exception as e:
        logger.error(f"✗ Ray integration test failed: {e}")
        return False


def test_directory_creation():
    """Probar que se pueden crear directorios de logs."""
    try:
        test_storage = Path("./test_benchmark_logs")
        test_config = {
            'name': 'test_experiment',
            'storage_path': str(test_storage)
        }

        # Simular la creación de estructura de directorios
        log_dir = test_storage / "Logs" / "test_experiment" / "RawData_test_trial"
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_dir.exists():
            logger.info(f"✓ Directory structure created successfully: {log_dir}")

            # Limpiar
            import shutil
            shutil.rmtree(test_storage)
            logger.info("✓ Test directories cleaned up")
            return True
        else:
            logger.error("✗ Failed to create directory structure")
            return False

    except Exception as e:
        logger.error(f"✗ Directory creation test failed: {e}")
        return False


def run_all_tests():
    """Ejecutar todas las pruebas."""
    logger.info("=" * 50)
    logger.info("BENCHMARK INTEGRATION TESTS")
    logger.info("=" * 50)

    tests = [
        ("Callback Imports", test_callbacks_import),
        ("Benchmarks Path", test_benchmarks_path),
        ("Dependencies", test_dependencies),
        ("Ray Integration", test_ray_integration),
        ("Directory Creation", test_directory_creation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test failed: {test_name}")

    logger.info("\n" + "=" * 50)
    logger.info(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Integration is ready.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
