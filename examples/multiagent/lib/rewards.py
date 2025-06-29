from .multiagent_env import MultiAgentF110


class ProgressRewardEnv(MultiAgentF110):
    def _get_rewards(self, newly_crashed) -> list:
        """Calculate individual rewards for each agent based on the original F110Env reward function."""
        
        # Initialize last_s tracking if not exists (track progress for each agent)
        if not hasattr(self, '_last_s'):
            self._last_s = [0.0] * self.env.num_agents

        rewards = []
        for i in range(self.env.num_agents):
            agent = self.agents[i]
            
            if agent in self._crashed_agents and agent not in newly_crashed:
                # Agent was already crashed - no reward calculation needed
                reward = 0.0
            else:
                # Calculate track progress using centerline spline (from F110Env)
                current_s, _ = self.env.track.centerline.spline.calc_arclength_inaccurate(
                    self.env.poses_x[i], self.env.poses_y[i]
                )

                # Calculate progress since last step
                prog = current_s - self._last_s[i]
                
                # Handle lap completion (when current_s wraps around to beginning)
                if prog > 0.9 * self.env.track.centerline.spline.s[-1]:
                    prog = (self.env.track.centerline.spline.s[-1] - self._last_s[i]) + current_s
                
                # Start with progress reward (main component from F110Env)
                reward = prog
                
                # Apply collision penalty (from F110Env)
                if agent in newly_crashed:  # Only penalize when agent crashes this step
                    reward -= 1.0
                
                # Update last track position for this agent
                self._last_s[i] = current_s
            
            rewards.append(reward)
        
        return rewards