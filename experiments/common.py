import logging
import torch
import math
import numpy
import numpy.linalg
from time import sleep

logger = logging.getLogger()

passableBlocks = ['air', 'water', 'lava', 'double_plant', 'tallgrass',
                  'reeds', 'red_flower', 'yellow_flower', 'flowing_lava',
                  'cobblestone', 'stone', 'sandstone', 'lapis_block']


visible_blocks = [
            'water',
            'dirt',
            'tallgrass',
            'sand',
            'log',
            'leaves',
            'grass',
            'lava',
            'double_plant',
            'reeds',
            'red_flower',
            'yellow_flower',
            'cobblestone',
            'stone',
            'sandstone',
            'flowing_lava',
            'obsidian',
            'lapis_block',
            'gravel',
            # to be continued..
            ]


visible_block_num = dict({None: 0})
for item in visible_blocks:
    visible_block_num[item] = len(visible_block_num)
visible_block_num['leaves2'] = visible_block_num['leaves']
visible_block_num['log2'] = visible_block_num['log']

block_id_cliff_walking = {'air': 0,
            'water': 1,
            'lava': 2,
            'flowing_lava': 2,
            'tallgrass': 3,
            'double_plant': 3,
            'reeds': 3,
            'grass': 3,
            'red_flower': 3,
            'yellow_flower': 3,
            'cobblestone': 4,
            'log': 4,
            'coal_ore': 4,
            'stone': 4,
            'dirt': 4,
            'gravel': 4,
            'clay': 4,
            'sandstone': 4,
            'lapis_block': 4,
            'sand': 4,
            'pumkin': 4,
            'pumpkin': 4,
            'lapis_ore': 4,
            'fire': 4, # fire is kind-of not passable
            'gold_ore': 4,
            'iron_ore': 4,
            'leaves': 4,
            }

max_id_blocks_walking = max(block_id_cliff_walking.values())


def grid_to_vec_walking(block_list):
    codes = numpy.zeros((len(block_list), max_id_blocks_walking + 1))
    for i,item in enumerate(block_list):
        codes[i][block_id_cliff_walking[item]] = 1
    return codes


# A simplistic search behavior
# Note that we don't use video input and rely on a small Grid and Ray,
# so our agent can miss objects visible by human
def search4blocks(mc, blocks, run=True):
    print('search for blocks')
    for t in range(3000):
        sleep(0.02)
        mc.observeProc()
        if mc.getLineOfSight('type') in blocks:
            stopMove(mc)
            return [mc.getLineOfSight('type'), mc.getLineOfSight('x'), mc.getLineOfSight('y'), mc.getLineOfSight('z')]
        if run:
            grid = mc.getNearGrid()
            if grid is not None:
                for i in range(len(grid)):
                    if grid[i] in blocks:
                        stopMove(mc)
                        import pdb;pdb.set_trace()
                        return [grid[i]] + mc.gridIndexToAbsPos(i)
            gridSlice = mc.gridInYaw()
            if gridSlice == None:
                continue
            ground = gridSlice[(len(gridSlice) - 1) // 2 - 1]
            solid = all([not (b in passableBlocks) for b in ground])
            wayLv0 = gridSlice[(len(gridSlice) - 1) // 2]
            wayLv1 = gridSlice[(len(gridSlice) - 1) // 2 + 1]
            passWay = all([b in passableBlocks for b in wayLv0]) and all([b in passableBlocks for b in wayLv1])
        turnVel = 0.125 * math.sin(t * 0.005)
        if run and (not (passWay and solid)):
            turnVel -= 1
        pitchVel = -0.015 * math.cos(t * 0.005)
        if run:
            mc.sendCommand("move 1")
        mc.sendCommand("turn " + str(turnVel))
        mc.sendCommand("pitch " + str(pitchVel))
    stopMove(mc)
    return None


def normAngle(angle):
    while (angle < -math.pi): angle += 2 * math.pi
    while (angle > math.pi): angle -= 2 * math.pi
    return angle


# Look at a specified location
def lookAt(mc, pos):
    print('look at')
    for t in range(3000):
        sleep(0.02)
        mc.observeProc()
        aPos = mc.getAgentPos()
        if aPos is None:
            continue
        [pitch, yaw] = mc.dirToPos(pos)
        pitch = normAngle(pitch - aPos[3]*math.pi/180.)
        yaw = normAngle(yaw - aPos[4]*math.pi/180.)
        if abs(pitch)<0.02 and abs(yaw)<0.02: break
        mc.sendCommand("turn " + str(yaw*0.4))
        mc.sendCommand("pitch " + str(pitch*0.4))
    mc.sendCommand("turn 0")
    mc.sendCommand("pitch 0")
    return math.sqrt((aPos[0] - pos[0]) * (aPos[0] - pos[0]) + (aPos[2] - pos[2]) * (aPos[2] - pos[2]))


def stopMove(mc):
    mc.sendCommand("move 0")
    mc.sendCommand("turn 0")
    mc.sendCommand("pitch 0")
    mc.sendCommand("jump 0")
    mc.sendCommand("strafe 0")


def direction_to_target(mc, pos):
    aPos = mc.getAgentPos()
    [pitch, yaw] = mc.dirToPos(pos)
    pitch = normAngle(pitch - aPos[3]*math.pi/180.)
    yaw = normAngle(yaw - aPos[4]*math.pi/180.)
    dist = math.sqrt((aPos[0] - pos[0]) * (aPos[0] - pos[0]) + (aPos[2] - pos[2]) * (aPos[2] - pos[2]))
    return pitch, yaw, dist


def stop_motion(mc):
    mc.sendCommand('move 0')
    mc.sendCommand('strafe 0')
    mc.sendCommand('pitch 0')
    mc.sendCommand('turn 0')
    mc.sendCommand('jump 0')


def learn(agent, optimizer):
    losses = []
    for i in range(40):
        optimizer.zero_grad()
        loss = agent.compute_loss()
        if loss is not None:
            # Optimize the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 2)
            optimizer.step()
            losses.append(loss.cpu().detach())
    if losses:
        logging.debug('optimizing')
        logging.debug('loss %f', numpy.mean(losses))
    return numpy.mean(losses)


class Trainer:
    def __init__(self, train=True):
        self.train = train
        if not self.train:
            logging.info('evaluation mode')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('using device {0}'.format(self.device))

    def collect_state(self):
        raise NotImplementedError()

    def run_episode(self):
        """ Deep Q-Learning episode
        """
        raise NotImplementedError()

    def learn(self, *args, **kwargs):
        if self.train:
            return learn(*args, **kwargs)
        return 0

    def act(self, actions):
        raise NotImplementedError()

    @classmethod
    def init_mission(i, mc):
        raise NotImplementedError()


def vectors_angle(vec1, vec2):
    angle = numpy.arccos(vec1 @ vec2 / (numpy.linalg.norm(vec1, 2) *
                                         numpy.linalg.norm(vec2, 2)))
    return angle



class  BaseLoader:
    def load_checkpoint(self, state_dict: dict, strict=True) -> bool:
        """
        Handles size mismatch

        adapted from
        https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
        """
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True
        self.load_state_dict(state_dict, strict=strict)
        return is_changed

