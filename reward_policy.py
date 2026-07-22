from f1tenth_gym.examples.multiagent_sac_experimental import MultiAgentF110
import numpy as np

class GPT4oMini(MultiAgentF110):
    """
    Política de recompensa multiagente para F110Env:
      - Máximo distancia recorrida
      - Bonus por cada vuelta completada
      - Penalización por accidentes (más fuerte si va a alta velocidad)
      - Penalización por alejarse de la raceline
      - Pequeño bonus por explorar nuevos tramos
    """
    def __init__(self,
                 env_config=None,
                 w_dist=1.0,
                 w_lap=100.0,
                 w_coll=-50.0,
                 w_coll_high=-100.0,
                 w_raceline=-1.0,
                 w_expl=0.1,
                 high_speed_thr=2.0):
        super().__init__(env_config)
        self.w_dist = w_dist
        self.w_lap = w_lap
        self.w_coll = w_coll
        self.w_coll_high = w_coll_high
        self.w_raceline = w_raceline
        self.w_expl = w_expl
        self.high_speed_thr = high_speed_thr

    def _get_rewards(self, newly_crashed):
        """
        states, next_states: list de dicts con keys 'distance','lap','speed','raceline_err','pos_id'
        infos: list de dicts con keys 'collision':bool, 'new_region':bool
        devuelve np.array shape=(n_agents,)
        """
        n = len(self.env.poses_x)
        rewards = np.zeros(n, dtype=np.float32)

        for i in range(n):
            s, ns, info = self._last_s[i], self.env.poses_x[i], self.env.infos[i]
            # 1) Distancia recorrida
            delta_dist = ns['distance'] - s['distance']
            r = self.w_dist * delta_dist

            # 2) Vueltas completadas
            lap_diff = ns['lap'] - s['lap']
            r += self.w_lap * lap_diff

            # 3) Colisión
            if newly_crashed[i]:
                if s['speed'] >= self.high_speed_thr:
                    r += self.w_coll_high
                else:
                    r += self.w_coll

            # 4) Proximidad a raceline (error lateral)
            r += self.w_raceline * abs(ns.get('raceline_err', 0.0))

            # 5) Exploración de pista
            if info.get('new_region', False):
                r += self.w_expl

            rewards[i] = r

        return rewards
