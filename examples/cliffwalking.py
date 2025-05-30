from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb
import numpy as np
import time
import random
import pygame
import sys
from scipy.special import softmax
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
        return True
    
    def training(self):
        self.train = True
        
    def evaluate(self):
        self.train = False
        
    def act(self, action):
        if isinstance(action, str):
            self.mc.discreteMove(action)
        else:
            self.mc.discreteMove(self.actions[action])
            
    def stop(self):
        self.reset()
        self.mc.stop()
        
    def reset(self):
        self.prev_a = None
        self.prev_s = None
    
    def choose_softmax(self, state):
        values = self.QTable[*state]
        probs = softmax(values)
        action = np.random.choice(len(self.actions), p=probs)
        return action
        
    def choose_action(self, state, softmax):
        if softmax:
            return self.choose_softmax(state)
        
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = np.argmax(self.QTable[*state])
        return a
    
    def getObs(self):
        self.mc.observeProc()
        x = self.mc.getFullStat("XPos")
        z = self.mc.getFullStat("ZPos")
        while x is None or z is None:
            self.mc.observeProc()
            x = self.mc.getFullStat("XPos")
            z = self.mc.getFullStat("ZPos")
        return [int(x), int(z)]
        
    def updateQTable(self, state, next_state, action, reward):
        old_Q = self.QTable[*state, action]
        self.QTable[*state, action] = old_Q + self.alpha * (reward + self.gamma * np.max(self.QTable[*next_state]) - old_Q)
    
    def update_epsilon(self, factor = 0.9):
        self.epsilon *= factor
    
    def step(self):
        current_s = self.getObs()
        action = self.choose_action(current_s, True)
        self.trajectory[*current_s, action] += 1
        self.act(action)
        next_s = self.getObs()
        time.sleep(0.5)
        try:
            rewards = self.mc.getRewards()[0].reward.reward_values
            reward = rewards[0]
        except:
            return next_s, None
        if reward is None:
            return next_s, None
        self.updateQTable(current_s, next_s, action, reward)
        if self.iter > 0 and self.iter % 10 == 0:
            self.update_epsilon()
        self.iter += 1
        return next_s, reward

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

    def getRandomColor(self):
        return (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))

    def drawGrid(self):
        for x in range(0, self.width, self.blockSize):
            pygame.draw.line(self.screen, self.gridColor, (x, 0), (x, self.height), width=1)
        for y in range(0, self.height, self.blockSize):
            pygame.draw.line(self.screen, self.gridColor, (0, y), (self.width, y), width=1)

    def getColor(self, value, min_val=-100, max_val=100):
        if min_val is None:
            min_val = np.min(value)
        if max_val is None:
            max_val = np.max(value)
        
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        
        normalized = np.clip(normalized, 0, 1)
        
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        return (red, green, 0)

    def markCell(self, state):
        color = self.getRandomColor()
        center = (0.5, 0.5)
        x = (state[0] + center[0]) * self.blockSize
        y = (state[1] + center[1]) * self.blockSize
        pygame.draw.circle(self.screen,color,(x,y), radius=self.blockSize / 7)
        pygame.display.update()


    def drawQTable(self, QTable : np.ndarray, display_vals : bool, trajectory : np.ndarray = None, episode : int = None):
        self.drawGrid()
        self.drawValues(QTable, display_vals, trajectory=trajectory)
        
        if episode is not None:
            font = pygame.font.SysFont('Arial', 16)
            episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
            self.screen.blit(episode_text, (10, 10))
    
    def adjustByTrajectory(self, color, trajVal):
        multiplier = min(1, trajVal / 1)
        return (int(color[0] * multiplier), int(color[1] * multiplier), 0)
    
    def drawValues(self, QTable : np.ndarray, display_vals : bool, trajectory : np.ndarray):
        actions = {"west" : (-0.3, 0), "east" : (0.3, 0), "north" : (0, -0.3), "south" : (0, 0.3)}
        center = (0.5, 0.5)
        for cell_x in range(QTable.shape[0]):
            pos_x = (cell_x + center[0]) * self.blockSize
            for cell_y in range(QTable.shape[1]):
                pos_y = (cell_y + center[1]) * self.blockSize
                for action_idx, action in enumerate(actions):
                    diff = actions[action]
                    x = pos_x + diff[0] * self.blockSize
                    y = pos_y + diff[1] * self.blockSize
                    val = QTable[cell_x, cell_y, action_idx]
                    if display_vals:
                        text = str(np.round(val, 1))
                        font = pygame.font.SysFont('freesansbold.ttf', 20)
                        text_surface = font.render(text, True, (255, 255, 255))
                        self.screen.blit(text_surface, (x, y))
                    else:
                        color = self.getColor(val)
                        if trajectory is not None:
                            color = self.adjustByTrajectory(color, trajectory[cell_x, cell_y, action_idx])
                        pygame.draw.circle(self.screen, color,(x,y), radius=self.blockSize / 6)
        pygame.display.update()

def main():
    np.random.seed(3)
    pygame.init()
    path = "examples/missions/cliff_walking.xml"
    mission = mb.MissionXML(xml_path=path)
    model = QLearning(mission)
    drawer = TableDisplayer(model.QTable.shape[0], model.QTable.shape[1], blockSize=60)
    clock = pygame.time.Clock()
    
    model.training()
    episode_num = 100
    
    running = True
    current_episode = 0
    
    while running and current_episode < episode_num:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
            
            drawer.drawQTable(model.QTable, trajectory=model.trajectory, display_vals=False)
            pygame.display.flip()
            
            # Check mission status
            if not model.mc.is_mission_running():
                model.stop()
                episode_running = False
                current_episode += 1 
                model.reset() 
                time.sleep(1) 
                break
            
            # Perform Q-learning step
            _, reward = model.step()
            # sys.stdout.write(f"\rЭпизод: {current_episode + 1} | Последняя награда: {reward}")
            # sys.stdout.flush()
            clock.tick(10)  
    
    pygame.quit()
    sys.exit()
    
if __name__ == "__main__":
    main()