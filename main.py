import argparse
import logging
from agent import MarioKartAgent
from src.utils import set_logging


class Runner:
    def __init__(self, agent, num_episodes, max_steps):
        self.agent = agent()
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def run(self):
            rewards = [] # Rewards of all episodes
            for episode_num in range(1, self.num_episodes):
                state = self.agent.reset()
                episode_reward = 0

                logging.info("phase 1") # NOOP until green light
                for _ in range(100):
                    (obs, rew, end, info) = self.agent.step(0) 
                    self.agent.conditional_render()

                logging.info("phase 2") # Train actor and critic networks
                for t in range(1,self.max_steps):
                    action, action_prob = self.agent.select_action(state)
                    next_state, observed_reward, terminated, info = self.agent.step(action)
                    self.agent.train(state=state, next_state=next_state,
                            action_prob=action_prob, observed_reward=observed_reward)

                    episode_reward += observed_reward
                    state = next_state
                    if terminated:
                        logging.info(f'Episode {episode_num} finished with reward: {episode_reward}')
                        rewards.append(episode_reward)
                        break

            input("press <enter> to exit....")
            self.running = False
            self.env.close()

def main(args):
    set_logging(args.log_file, args.log_level, not args.stop_log_stdout)
    agent = MarioKartAgent(args.graphic_output)
    runner = Runner(agent=agent, num_episodes=400, max_steps=10000)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RL Agent for Mario Kart for N64 Emulator")
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--stop-log-stdout", action="store_false", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--graphic-output", action="store_true", default=False,
                        help="toggles weather the graphical output of Mario Kart should be rendered")

    args = parser.parse_args()
    main(args)