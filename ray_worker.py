import subprocess
import time

HEAD_ADDRESS = "192.168.1.152:6379"
LOCAL_NODE_IP = "10.8.0.3"
HOST_NODE_ID = "node_9a0913008dde70715a115265819284ce58023d434cdf64a428be90b9"
CHECK_INTERVAL = 15  # segundos entre verificaciones


def is_worker_active():
    """Verifica si existen 2 nodos y uno es diferente del host."""
    try:
        result = subprocess.run(["ray", "status"], capture_output=True, text=True)
        output = result.stdout

        # Contar el número de nodos activos
        node_count = output.count("node_")

        # Verificar que hay exactamente 2 nodos
        if node_count != 2:
            print(f"[INFO] Se encontraron {node_count} nodos, se esperan 2")
            return False

        # Verificar que uno de los nodos NO es el host
        lines = output.split('\n')
        nodes_found = []
        for line in lines:
            if "node_" in line:
                # Extraer el ID del nodo de la línea
                node_start = line.find("node_")
                if node_start != -1:
                    # Buscar el final del ID del nodo (hasta el siguiente espacio o carácter especial)
                    node_end = node_start
                    while node_end < len(line) and line[node_end] not in [' ', '\t', ')', ']', ',']:
                        node_end += 1
                    node_id = line[node_start:node_end]
                    nodes_found.append(node_id)

        # Verificar que tenemos exactamente 2 nodos y uno no es el host
        if len(nodes_found) == 2:
            has_host = HOST_NODE_ID in nodes_found
            has_worker = any(node != HOST_NODE_ID for node in nodes_found)
            return has_host and has_worker

        return False
    except Exception as e:
        print(f"[WARN] No se pudo verificar ray status: {e}")
        return False


def start_ray_worker():
    print("🟡 Iniciando ray worker...")
    subprocess.run(["ray", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(
        ["ray", "start", f"--address={HEAD_ADDRESS}"],
        capture_output=True,
        text=True
    )
    if "Ray runtime started" in result.stdout:
        print("✅ ray worker iniciado.")
    else:
        print("❌ Falló al iniciar ray worker:")
        print(result.stdout)
        print(result.stderr)


def main():
    print("🧩 Iniciando watchdog de ray worker...")
    while True:
        if not is_worker_active():
            print(f"[{time.ctime()}] ⚠️ Worker inactivo o desconectado. Reiniciando...")
            start_ray_worker()
            print(f" Worker activo: {is_worker_active()}")
        else:
            print(f"[{time.ctime()}] ✅ Worker activo.")

        # Cuenta regresiva con actualización cada segundo
        for remaining in range(CHECK_INTERVAL, 0, -1):
            print(f"\r⏳ Próxima verificación en {remaining} segundos...", end='', flush=True)
            time.sleep(1)
        print()  # Nueva línea después del contador


if __name__ == "__main__":
    main()
