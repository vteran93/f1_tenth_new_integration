    def _get_rewards(self, newly_crashed):
        # Calculate rewards
        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            
            if agent in self._crashed_agents and agent not in newly_crashed:
                # Agent was already crashed - no reward calculation needed
                reward = 0.0
            else:
                current_pos = (self.env.poses_x[i], self.env.poses_y[i])
                last_pos = self._last_positions[i]
                
                # Calculate progress (simple euclidean distance moved)
                progress = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
                
                # Reward components:
                # 1. Progress reward (encourage forward movement)
                progress_reward = progress * 10.0  # Scale progress
                # 2. Collision penalty (only applied when crashing this step)
                collision_penalty = 100.0 if agent in newly_crashed else 0.0
                # 3. Small baseline reward for staying alive
                survival_reward = 0.1
                
                reward = progress_reward + survival_reward - collision_penalty
                self._last_positions[i] = current_pos
            
            rewards.append(reward)