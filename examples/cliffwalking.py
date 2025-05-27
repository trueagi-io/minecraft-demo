from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb
import numpy as np
import time
import random
import pygame
import sys

class QLearning:
    def __init__(self, mission, qTable = None, epsilon = 1., 
                 alpha = 0.1, gamma = 1.0):
        self.mission = mission
        if qTable is not None:
            self.QTable = qTable
        else:
            self.QTable = np.zeros((6, 14, 4)) #width, length and action space
        self.trajectory = np.zeros_like(self.QTable)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.actions = ["west", "east", "north", "south"]
        self.offset = {"XPos" : 4, "ZPos" : 1}
        self.statKeys = ["XPos", "ZPos"]
        self.prev_s = None
        self.prev_a = None
        self.next_a = None
        self.iter = 0
        
    def start(self):
        self.mc = MCConnector(self.mission)
        started = self.mc.safeStart()
        
        if not started:
            return False
        time.sleep(5)
        
        world_state = self.mc.agent_hosts[0].getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.mc.agent_hosts[0].getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print()
        return True
    
    def training(self):
        self.train = True
        
    def evaluate(self):
        self.train = False
        
    def act(self, action):
        self.mc.discreteMove(self.actions[action])
            
    def stop(self):
        self.reset()
        self.mc.stop()
        
    def getRewards(self):
        try:
            self.mc.getRewards()
        except:
            self.mc.agent_hosts[0].getFinalReward()
        
    def reset(self):
        self.prev_a = None
        self.prev_s = None
        
    def choose_action(self, state):
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = np.argmax(self.QTable[*state])
        return a
        
    def updateQTable(self, state, next_state, action, reward):
        old_Q = self.QTable[*state, action]
        self.QTable[*state, action] = old_Q + self.alpha * (reward + self.gamma * np.max(self.QTable[*next_state]) - old_Q)
    
    def update_epsilon(self, factor = 0.9):
        self.epsilon *= factor
    
    def step(self):
        self.mc.observeProc()
        obs = self.mc.getFullStat("XPos")
        while obs is None:
            self.mc.observeProc()     
            obs = self.mc.getFullStat("XPos")
            time.sleep(0.1)
        current_s = [int(self.mc.getFullStat(key)) - self.offset[key] + 1 for key in self.statKeys]
        action = self.choose_action(current_s)
        self.trajectory[*current_s, action] += 1
        self.act(action)
        self.mc.observeProc()  
        new_obs = self.mc.getFullStat("XPos")
        while new_obs is None:
            self.mc.observeProc()     
            new_obs = self.mc.getFullStat("XPos")
            time.sleep(0.1)
        next_s = [int(self.mc.getFullStat(key)) - self.offset[key] + 1 for key in self.statKeys]
        time.sleep(0.5)
        rewards = self.mc.getRewards()[0].reward.reward_values
        reward = rewards[0]
        print(reward)
        self.updateQTable(current_s, next_s, action, reward)
        if self.iter > 0 and self.iter % 10 == 0:
            self.update_epsilon()
        self.iter += 1

class TableDisplayer:
    def __init__(self, blockWidth, blockHeight, blockSize = 20):
        self.blockWidth = blockWidth
        self.blockHeight = blockHeight
        self.blockSize = blockSize
        self.width = blockWidth * blockSize
        self.height = blockHeight * blockSize
        self.gridColor = (150, 150, 150)
        self.screen = pygame.display.set_mode((self.width, self.height))  
        self.screen.fill((0,0,0))

    def drawGrid(self):
        for x in range(0, self.width, self.blockSize):
            pygame.draw.line(self.screen, self.gridColor, (x, 0), (x, self.height), width=1)
        for y in range(0, self.height, self.blockSize):
            pygame.draw.line(self.screen, self.gridColor, (0, y), (self.width, y), width=1)

    def getColor(self, value, min_val=-150, max_val=100):
        if min_val is None:
            min_val = np.min(value)
        if max_val is None:
            max_val = np.max(value)
        
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        return (red, green, 0)

    def drawQTable(self, QTable : np.ndarray, episode = None):
        self.drawGrid()
        self.drawValues(QTable)
        
        if episode is not None:
            font = pygame.font.SysFont('Arial', 16)
            episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
            self.screen.blit(episode_text, (10, 10))
    
    def drawValues(self, QTable : np.ndarray):
        actions = {"west" : (0.4, 0), "east" : (-0.4, 0), "north" : (0, -0.4), "south" : (0, 0.4)}
        center = (0.5, 0.5)
        for cell_x in range(QTable.shape[0]):
            pos_x = (cell_x + center[0]) * self.blockSize
            for cell_y in range(QTable.shape[1]):
                pos_y = (cell_y + center[1]) * self.blockSize
                for action_idx, action in enumerate(actions):
                    diff = actions[action]
                    x = pos_x + diff[0] * self.blockSize
                    y = pos_y + diff[1] * self.blockSize
                    color = self.getColor(QTable[cell_x, cell_y, action_idx])
                    pygame.draw.circle(self.screen, color,(x,y), radius=self.blockSize / 6)

def main():
    np.random.seed(3)
    pygame.init()
    path = "D:/Downloads/Telegram Desktop/cliff_walking_1.xml"
    mission = mb.MissionXML(xml_path=path)
    model = QLearning(mission)
    # drawer = TableDisplayer(model.QTable.shape[0], model.QTable.shape[1], blockSize=40)
    # clock = pygame.time.Clock()
    
    model.training()
    episode_num = 100
    
    running = True
    current_episode = 0
    
    while running and current_episode < episode_num:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        print(f"Starting episode {current_episode + 1}...")
        started = model.start()
        if not started:
            print("Episode did not start, retrying...")
            continue
            
        episode_running = True
        while episode_running and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_running = False
            
            if not running:
                break
            
            # drawer.drawQTable(model.QTable)
            # pygame.display.flip()
            
            # Check mission status
            if not model.mc.is_mission_running():
                print(f"Episode {current_episode + 1} ended")
                model.stop()
                episode_running = False
                current_episode += 1 
                model.reset() 
                time.sleep(1) 
                break
            
            # Perform Q-learning step
            model.step()
            # clock.tick(10)  
    
    pygame.quit()
    sys.exit()
    
if __name__ == "__main__":
    main()