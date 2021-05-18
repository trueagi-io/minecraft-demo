from log import setup_logger
import torch
import logging
import MalmoPython
import json
import logging
import sys
import time


def train_agent(agent, Trainer):
    num_repeats = 400
    eps = 0.26
    eps_start = eps
    eps_end = 0.09
    eps_decay = 0.99
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.0005,
                                    weight_decay=0.01)
    mc = None
    for i in range(0, num_repeats):
        mc = Trainer.init_mission(i, mc)

        logging.debug("\nMission %d of %d:" % (i + 1, num_repeats))
        mc.safeStart()

        # -- run the agent in the world -- #
        trainer = Trainer(agent, mc, optimizer, eps)
        comulative_reward, steps, solved = trainer.run_episode() 
        logging.info('episode %i: solved %i: comulative reward: %f', i, solved, comulative_reward)
        logging.debug("eps: %f", eps)
        eps = max(eps * eps_decay, eps_end)

        # -- clean up -- #
        time.sleep(0.5)  # (let the Mod reset)

        if i % 14 == 0:
            torch.save(agent.state_dict(), 'agent.pth')


def train_cliff():
    from cliff import load_agent_cliff, Trainer
    agent = load_agent_cliff()
    train_agent(agent, Trainer)


if __name__ == '__main__':
    setup_logger('train.log')
    train_cliff()
