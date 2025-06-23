import numpy as np
from abc import ABC, abstractmethod
try:
    from planners import PurePursuitPlanner
except ImportError:
    raise ImportError("Cannot import 'planners'. Make sure 'planners.py' exists in the same directory or is in your PYTHONPATH.")

class RewardStrategy(ABC):
    """Clase base abstracta para estrategias de recompensa."""

    def __init__(self, num_agents):
        self.num_agents = num_agents

    @abstractmethod
    def compute_rewards(self, env, obs):
        """Calcula las recompensas para cada agente."""
        pass

    def reset(self, positions):
        """Reinicia el estado interno de la estrategia de recompensa."""
        pass

    def update_actions(self, actions):
        """Actualiza las acciones (usado por TALearningReward y PaperReward)."""
        pass

class ProgressReward(RewardStrategy):
    """Recompensa basada en el progreso (distancia recorrida por el vehículo)."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.progress_scale = config.get("progress_scale", 10.0)
        self.base_reward = config.get("base_reward", 0.1)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.min_speed_penalty = config.get("min_speed_penalty", 5.0)
        self.min_speed_threshold = config.get("min_speed_threshold", 0.5)
        self.last_positions = None

    def reset(self, positions):
        """Reinicia las últimas posiciones conocidas."""
        self.last_positions = positions

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en el progreso y la velocidad."""
        rewards = np.zeros(self.num_agents)
        current_positions = [(env.poses_x[i], env.poses_y[i]) for i in range(self.num_agents)]

        for i in range(self.num_agents):
            if self.last_positions is None:
                rewards[i] = self.base_reward
                continue

            # Calcular la distancia euclidiana recorrida
            last_x, last_y = self.last_positions[i]
            curr_x, curr_y = current_positions[i]
            distance = np.sqrt((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2)

            # Recompensa por progreso
            rewards[i] = self.progress_scale * distance + self.base_reward

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

            # Penalización por velocidad baja
            speed = obs["linear_vels_x"][i]
            if speed < self.min_speed_threshold:
                rewards[i] -= self.min_speed_penalty

        self.last_positions = current_positions
        return rewards.tolist()

class SpeedTrackReward(RewardStrategy):
    """Recompensa basada en la velocidad y la desviación de la pista."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.speed_scale = config.get("speed_scale", 5.0)
        self.track_deviation_penalty = config.get("track_deviation_penalty", 2.0)
        self.collision_penalty = config.get("collision_penalty", 100.0)

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en la velocidad y la desviación."""
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            # Obtener velocidad desde las observaciones
            speed = obs["linear_vels_x"][i]
            # Recompensa proporcional a la velocidad
            rewards[i] = self.speed_scale * speed

            # Penalización por desviación de la pista (usando escaneos LIDAR)
            scan = obs["scans"][i]
            min_distance = np.min(scan)
            if min_distance < 0.5:  # Umbral para considerar desviación
                rewards[i] -= self.track_deviation_penalty * (0.5 - min_distance)

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

        return rewards.tolist()

class CrossTrackHeadReward(RewardStrategy):
    """Recompensa basada en el error de trayectoria (cross-track) y orientación (heading)."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        map_path = config.get("map_path", "../maps/Spielberg/Spielberg_raceline.csv")
        self.planner = PurePursuitPlanner(
            map_path=map_path,
            lookahead_distance=config.get("lookahead_distance", 1.5),
            wheelbase=config.get("wheelbase", 0.33),
            max_steer=config.get("max_steer", 0.5)
        )
        self.cte_scale = config.get("cte_scale", 3.0)
        self.he_scale = config.get("he_scale", 2.0)
        self.max_cte = config.get("max_cte", 2.0)
        self.max_he = config.get("max_he", 0.7854)  # pi/4 radianes
        self.collision_penalty = config.get("collision_penalty", 100.0)

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en errores de trayectoria y orientación."""
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            x, y = env.poses_x[i], env.poses_y[i]
            theta = env.poses_theta[i]

            # Obtener el error de trayectoria (CTE) y orientación (HE)
            cte, he = self.planner.compute_errors(x, y, theta)

            # Limitar los errores para evitar penalizaciones excesivas
            cte = np.clip(cte, -self.max_cte, self.max_cte)
            he = np.clip(he, -self.max_he, self.max_he)

            # Recompensa negativa proporcional a los errores
            rewards[i] = -(self.cte_scale * cte ** 2 + self.he_scale * he ** 2)

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

        return rewards.tolist()

class TALearningReward(RewardStrategy):
    """Recompensa basada en la desviación respecto a un controlador PurePursuit."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        map_path = config.get("map_path", "../maps/Spielberg/Spielberg_raceline.csv")
        self.planner = PurePursuitPlanner(
            map_path=map_path,
            lookahead_distance=config.get("lookahead_distance", 1.5),
            wheelbase=config.get("wheelbase", 0.33),
            max_steer=config.get("max_steer", 0.5)
        )
        self.steer_scale = config.get("steer_scale", 10.0)
        self.speed_scale = config.get("speed_scale", 5.0)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.actions = None

    def update_actions(self, actions):
        """Actualiza las acciones tomadas por los agentes."""
        self.actions = actions

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en la desviación de las acciones del controlador."""
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            if self.actions is None:
                rewards[i] = 0.0
                continue

            x, y = env.poses_x[i], env.poses_y[i]
            theta = env.poses_theta[i]
            agent_action = self.actions[i]  # [steering_angle, speed]

            # Obtener la acción del controlador PurePursuit
            ref_steer, ref_speed = self.planner.compute_action(x, y, theta)

            # Calcular errores entre las acciones del agente y las de referencia
            steer_error = agent_action[0] - ref_steer
            speed_error = agent_action[1] - ref_speed

            # Recompensa negativa proporcional a los errores
            rewards[i] = -(self.steer_scale * steer_error ** 2 + self.speed_scale * speed_error ** 2)

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

        return rewards.tolist()

class ProgressCompetitionReward(RewardStrategy):
    """Recompensa por progreso en pista, adelantamientos y penalización por colisión."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.progress_scale = config.get("progress_scale", 10.0)
        self.overtake_bonus = config.get("overtake_bonus", 2.0)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.last_positions = None
        self.last_ranks = None

    def reset(self, positions):
        self.last_positions = positions
        self.last_ranks = np.arange(self.num_agents)

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en progreso y adelantamientos."""
        rewards = np.zeros(self.num_agents)
        current_positions = [(env.poses_x[i], env.poses_y[i]) for i in range(self.num_agents)]
        # Ranking por progreso en la pista (usando lap_counts como proxy)
        current_ranks = np.argsort([env.lap_counts[i] for i in range(self.num_agents)])

        for i in range(self.num_agents):
            # Progreso
            if self.last_positions is None:
                rewards[i] = 0.0
                continue
            last_x, last_y = self.last_positions[i]
            curr_x, curr_y = current_positions[i]
            distance = np.sqrt((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2)
            rewards[i] += self.progress_scale * distance

            # Bonus por adelantar
            if self.last_ranks is not None and current_ranks[i] < self.last_ranks[i]:
                rewards[i] += self.overtake_bonus

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

        self.last_positions = current_positions
        self.last_ranks = current_ranks
        return rewards.tolist()

class SafetyAwareReward(RewardStrategy):
    """Penaliza colisiones y bonifica mantener distancia segura."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.safe_distance = config.get("safe_distance", 1.5)
        self.safety_bonus = config.get("safety_bonus", 1.0)

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en seguridad."""
        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

            # Bonus por mantener distancia segura
            min_dist = np.inf
            for j in range(self.num_agents):
                if i == j: continue
                dist = np.linalg.norm([env.poses_x[i] - env.poses_x[j], env.poses_y[i] - env.poses_y[j]])
                if dist < min_dist:
                    min_dist = dist
            if min_dist > self.safe_distance:
                rewards[i] += self.safety_bonus

        return rewards.tolist()

class HierarchicalHPRSReward(RewardStrategy):
    """Priorización jerárquica: seguridad > objetivo > comodidad."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.progress_scale = config.get("progress_scale", 10.0)
        self.speed_scale = config.get("speed_scale", 2.0)
        self.jerk_penalty = config.get("jerk_penalty", 0.1)
        self.last_speeds = np.zeros(num_agents)

    def reset(self, positions):
        self.last_speeds = np.zeros(self.num_agents)

    def compute_rewards(self, env, obs):
        """Calcula recompensas jerárquicas."""
        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            # Seguridad (máxima prioridad)
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

            # Objetivo: progreso y velocidad
            speed = obs["linear_vels_x"][i]
            rewards[i] += self.progress_scale * speed

            # Comodidad: penalización por cambios bruscos de velocidad (jerk)
            jerk = abs(speed - self.last_speeds[i])
            rewards[i] -= self.jerk_penalty * jerk
            self.last_speeds[i] = speed

        return rewards.tolist()

class AdversarialReward(RewardStrategy):
    """Recompensa por ventaja frente a oponentes y penalización por bloqueos."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.advantage_bonus = config.get("advantage_bonus", 3.0)
        self.block_penalty = config.get("block_penalty", 2.0)
        self.block_distance = config.get("block_distance", 1.0)

    def compute_rewards(self, env, obs):
        """Calcula recompensas competitivas."""
        rewards = np.zeros(self.num_agents)
        # Ranking por progreso en la pista (usando lap_counts)
        ranks = np.argsort([env.lap_counts[i] for i in range(self.num_agents)])

        for i in range(self.num_agents):
            # Bonus por estar delante de otros
            my_rank = np.where(ranks == i)[0][0]
            rewards[i] += self.advantage_bonus * (self.num_agents - my_rank - 1)

            # Penalización por bloquear (estar cerca y delante)
            for j in range(self.num_agents):
                if i == j: continue
                dist = np.linalg.norm([env.poses_x[i] - env.poses_x[j], env.poses_y[i] - env.poses_y[j]])
                if dist < self.block_distance and env.lap_counts[i] > env.lap_counts[j]:
                    rewards[i] -= self.block_penalty

        return rewards.tolist()

class PaperReward(RewardStrategy):
    """Recompensa basada en arXiv:2103.10098, con progreso, velocidad, desviación y colisión."""

    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        self.collision_penalty = config.get("collision_penalty", 100.0)
        self.speed_scale = config.get("speed_scale", 2.0)
        self.deviation_scale = config.get("deviation_scale", 1.0)
        self.timestep = config.get("timestep", 0.01)
        # Configuración para PurePursuitPlanner
        map_path = config.get("map_path", "../maps/Spielberg/Spielberg_raceline.csv")
        self.planner = PurePursuitPlanner(
            map_path=map_path,
            lookahead_distance=config.get("lookahead_distance", 1.5),
            wheelbase=config.get("wheelbase", 0.33),
            max_steer=config.get("max_steer", 0.5)
        )
        self.actions = None

    def update_actions(self, actions):
        """Actualiza las acciones tomadas por los agentes."""
        self.actions = actions

    def compute_rewards(self, env, obs):
        """Calcula recompensas basadas en el paper."""
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            # Obtener estado del agente
            x, y = env.poses_x[i], env.poses_y[i]
            theta = env.poses_theta[i]
            speed = obs["linear_vels_x"][i]

            # Recompensa por progreso
            cte, he = self.planner.compute_errors(x, y, theta)  # CTE y heading error
            if self.actions is not None:
                steering = self.actions[i][0]  # Ángulo de dirección desde la acción
            else:
                steering = 0.0  # Valor por defecto si no hay acciones
            progress_term = speed * (np.cos(he) - abs(np.sin(he)) - abs(steering)) * self.timestep
            rewards[i] += progress_term

            # Recompensa por velocidad
            rewards[i] += self.speed_scale * speed

            # Penalización por desviación
            rewards[i] -= self.deviation_scale * abs(cte)

            # Penalización por colisión
            if env.collisions[i]:
                rewards[i] -= self.collision_penalty

        return rewards.tolist()
    
class RacePerformanceReward(RewardStrategy):
    """Recompensa compuesta para carreras F1TENTH que valora velocidad, seguridad, adelantamientos y conducción suave."""
    
    def __init__(self, num_agents, config):
        super().__init__(num_agents)
        # Parámetros configurables
        self.progress_scale = config.get("progress_scale", 15.0)
        self.speed_scale = config.get("speed_scale", 5.0)
        self.collision_penalty = config.get("collision_penalty", 200.0)
        self.overtake_bonus = config.get("overtake_bonus", 50.0)
        self.stall_penalty = config.get("stall_penalty", 10.0)
        self.jerk_penalty = config.get("jerk_penalty", 2.0)
        self.min_speed_threshold = config.get("min_speed_threshold", 0.5)  # m/s
        
        # Estado interno
        self.last_positions = None
        self.last_speeds = None
        self.last_steering = None
        self.lap_count = np.zeros(num_agents)
        self.last_lap_progress = np.zeros(num_agents)
        self.agent_positions = np.zeros(num_agents)  # Para tracking de adelantamientos
        
    def reset(self, positions):
        """Reinicia el estado interno del reward."""
        self.last_positions = positions
        self.last_speeds = np.zeros(self.num_agents)
        self.last_steering = np.zeros(self.num_agents)
        self.last_lap_progress = np.zeros(self.num_agents)
        # Orden inicial basado en posición x (simulando posición en carrera)
        x_positions = [p[0] for p in positions]
        self.agent_positions = np.argsort(x_positions)[::-1]  # De mayor a menor x
        
    def compute_rewards(self, env, obs):
        """Calcula la recompensa compuesta."""
        rewards = np.zeros(self.num_agents)
        current_positions = [(env.poses_x[i], env.poses_y[i]) for i in range(self.num_agents)]
        current_speeds = obs["linear_vels_x"]
        current_steering = np.array([a[0] for a in self.actions]) if self.actions else np.zeros(self.num_agents)
        
        # 1. Recompensa por progreso (distancia recorrida)
        progress_rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.last_positions is not None:
                last_x, last_y = self.last_positions[i]
                curr_x, curr_y = current_positions[i]
                distance = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
                progress_rewards[i] = self.progress_scale * distance
        
        # 2. Recompensa por velocidad (incentivar vueltas rápidas)
        speed_rewards = self.speed_scale * current_speeds
        
        # 3. Penalización por colisión
        collision_penalties = np.array([self.collision_penalty if c else 0.0 for c in env.collisions])
        
        # 4. Bonus por adelantamiento (solo multiagente)
        overtake_bonuses = np.zeros(self.num_agents)
        if self.num_agents > 1:
            new_x_positions = [p[0] for p in current_positions]
            new_agent_positions = np.argsort(new_x_positions)[::-1]
            
            for i in range(self.num_agents):
                old_pos = np.where(self.agent_positions == i)[0][0]
                new_pos = np.where(new_agent_positions == i)[0][0]
                if new_pos < old_pos:  # Mejoró su posición
                    overtake_bonuses[i] = self.overtake_bonus * (old_pos - new_pos)
            
            self.agent_positions = new_agent_positions
        
        # 5. Penalización por quedarse parado
        stall_penalties = np.array([self.stall_penalty if s < self.min_speed_threshold else 0.0 
                                   for s in current_speeds])
        
        # 6. Penalización por cambios bruscos de dirección (jerk)
        jerk_penalties = np.zeros(self.num_agents)
        if self.last_steering is not None:
            steering_changes = np.abs(current_steering - self.last_steering)
            jerk_penalties = self.jerk_penalty * steering_changes
        
        # Combinar todos los componentes
        rewards = (progress_rewards + speed_rewards + overtake_bonuses - 
                  collision_penalties - stall_penalties - jerk_penalties)
        
        # Bonus por completar una vuelta (detección simple basada en progreso circular)
        for i in range(self.num_agents):
            if self.last_positions is not None:
                # Detección simple de vuelta completa (podría mejorarse con track progress)
                if current_positions[i][0] - self.last_positions[i][0] < -10.0:  # Cruzó la línea de inicio
                    self.lap_count[i] += 1
                    rewards[i] += 100.0 * self.lap_count[i]  # Bonus incremental por cada vuelta
        
        # Actualizar estado interno
        self.last_positions = current_positions
        self.last_speeds = current_speeds
        self.last_steering = current_steering
        
        return rewards.tolist()