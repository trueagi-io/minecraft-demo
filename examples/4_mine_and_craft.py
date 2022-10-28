import logging
import math
from time import sleep
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
import tagilmo.utils.mission_builder as mb

from tagilmo.utils.mathutils import normAngle, degree2rad

import numpy as np
import minecraft_data


# This script shows a relatively complex behavior of gathering resources
# for an iron pickaxe. It includes searchng for logs, mining stones, etc.
# There are many heurisics and many things that can go wrong,
# so the final result will not be achieved. And this is OK.
#
# The example is intented to show the capabilities and limitations of
# imperative scripts and hand-coding. Failer can be due to the lack
# of robustness and flexibility of:
# * local skills (e.g. the agent can start mining a shaft in the direction of lava)
# * high-level plan (the agent can run out of sticks, when the plan assumes it has enough)
# The longer the plan, the more things can go wrong.
# It is instructive to examine failure cases.

# apparantely logs naming is different in different versions of Minecraft,
# so we need to get right names for our current version

log_names = []
versions = minecraft_data.common().protocolVersions
mcd = minecraft_data('1.18.1')  # here we must put current minecraft version
for item in mcd.items_list:
    iname = item['name']
    if 'log' in iname:
        log_names.append(iname)

# ============== some hand-coded skills ==============

# This function is executed to run to a visible object,
# but it doesn't check if the path is safe
def runStraight(rob, dist, keepHeight=False):
    logging.info("\tinside runStraight")
    start = rob.waitNotNoneObserve('getAgentPos')
    rob.sendCommand("move 1")
    bJump = False
    for t in range(2 + int(dist * 5)):
        sleep(0.1)
        rob.observeProcCached()
        pos = rob.getCachedObserve('getAgentPos')
        if pos is None:
            continue
        if dist * dist < (pos[0] - start[0]) * (pos[0] - start[0]) + (pos[2] - start[2]) * (pos[2] - start[2]):
            break
        if pos[1] < start[1] - 0.5 and keepHeight:
            rob.sendCommand("jump 1")
            rob.sendCommand("attack 1")
            bJump = True
        elif bJump:
            rob.sendCommand("jump 0")
            rob.sendCommand("attack 0")
            bJump = False
        los = rob.getCachedObserve('getLineOfSights')
        if los is not None and los['distance'] < 0.5 and \
               not los['distance'] in RobustObserver.passableBlocks and \
               not los['type'] in RobustObserver.passableBlocks and\
               not bJump:
            break
    rob.sendCommand("move 0")


# Just look at a specified direction
def lookDir(rob, pitch, yaw):
    logging.info("\tinside lookDir")
    for t in range(3000):
        sleep(0.02)  # wait for action
        aPos = rob.waitNotNoneObserve('getAgentPos')
        dPitch = normAngle(pitch - degree2rad(aPos[3]))
        dYaw = normAngle(yaw - degree2rad(aPos[4]))
        if abs(dPitch) < 0.02 and abs(dYaw) < 0.02: break
        rob.sendCommand("turn " + str(dYaw * 0.4))
        rob.sendCommand("pitch " + str(dPitch * 0.4))
    rob.sendCommand("turn 0")
    rob.sendCommand("pitch 0")


# Look at a specified location
def lookAt(rob, pos):
    logging.info("\tinside lookAt")
    for t in range(3000):
        sleep(0.02)
        aPos = rob.waitNotNoneObserve('getAgentPos')
        [pitch, yaw] = rob.dirToAgentPos(pos)
        pitch = normAngle(pitch - degree2rad(aPos[3]))
        yaw = normAngle(yaw - degree2rad(aPos[4]))
        if abs(pitch) < 0.02 and abs(yaw) < 0.02: break
        rob.sendCommand("turn " + str(yaw * 0.4))
        rob.sendCommand("pitch " + str(pitch * 0.4))
    rob.sendCommand("turn 0")
    rob.sendCommand("pitch 0")
    return math.sqrt((aPos[0] - pos[0]) * (aPos[0] - pos[0]) + (aPos[2] - pos[2]) * (aPos[2] - pos[2]))


def strafeCenterX(rob):
    logging.info("\tinside strafeCenterX")
    rob.sendCommand('strafe 0.1')
    for t in range(200):
        sleep(0.02)
        aPos = rob.waitNotNoneObserve('getAgentPos')
        if int(abs(aPos[0]) * 10 + 0.5) % 10 == 5:
            break
    rob.stopMove()


# A simplistic search behavior
# Note that we don't use video input and rely on a small Grid and Ray,
# so our agent can miss objects visible by human
def search4blocks(rob, blocks):
    logging.info("\tinside search4blocks")
    for t in range(3000):
        sleep(0.02)  # for action execution - not observations
        grid = rob.waitNotNoneObserve('getNearGrid')
        output = [[grid[i],i] for i in range(len(grid)) for j in blocks if j in grid[i]]
        if len(output) > 0:
            rob.stopMove()
            poses = []
            for out in output:
                poses.append(rob.mc.gridIndexToPos(out[1]))
            poses = np.asarray(poses)
            sum = np.sum(poses, axis=1)
            ind = np.argmin(np.abs(sum))
            return [output[ind][0]] + rob.gridIndexToAbsPos(output[ind][1])
        los = rob.getCachedObserve('getLineOfSights')
        if (los["hitType"] != "MISS"):
            if los is not None and los['type'] in blocks:
                rob.stopMove()
                return [los['type'], los['x'], los['y'], los['z']]
        path = rob.analyzeGridInYaw()
        turnVel = 0.25 * math.sin(t * 0.05)
        if not (path['passWay'] and path['solid']):
            turnVel -= 1
        pitchVel = -0.015 * math.cos(t * 0.03)
        rob.sendCommand("move 1")
        rob.sendCommand("turn " + str(turnVel))
        rob.sendCommand("pitch " + str(pitchVel))
    rob.stopMove()
    return None


# Just attacking while the current block is not destroyed
# assuming nothing else happens
def mineAtSight(rob):
    logging.info("\tinside mineAtSight")
    sleep(0.1)
    rob.observeProcCached()
    los = rob.getCachedObserve('getLineOfSights')
    if los is None or los['type'] is None or not los['inRange']:
        return False
    dist = los['distance']
    obj = los['type']
    rob.sendCommand('attack 1')
    for t in range(100):
        los = rob.getCachedObserve('getLineOfSights')
        if los['hitType'] == 'MISS':
            continue
        if los is None or los['type'] is None or \
           abs(dist - los['distance']) > 0.01 or obj != los['type']:
            rob.sendCommand('attack 0')
            return True
        sleep(0.1)
        rob.observeProcCached()
    rob.sendCommand('attack 0')
    return False


# A skill to choose a tool for mining (in the context of the current example)
def chooseTool(rob):
    # logging.info("\tinside chooseTool")
    los = rob.getCachedObserve('getLineOfSights')
    if los is None:
        return
    wooden_pickaxe = rob.filterInventoryItem('wooden_pickaxe')
    if wooden_pickaxe and wooden_pickaxe[0]['index'] != 0:
        rob.sendCommand('swapInventoryItems 0 ' + str(wooden_pickaxe[0]['index']))
    stone_pickaxe = rob.filterInventoryItem('stone_pickaxe')
    if stone_pickaxe and stone_pickaxe[0]['index'] != 1:
        rob.sendCommand('swapInventoryItems 1 ' + str(stone_pickaxe[0]['index']))
    if los['type'] in ['dirt', 'grass']:
        rob.sendCommand('hotbar.9 1')
        rob.sendCommand('hotbar.9 0')
    elif los['type'] in ['iron_ore']:
        rob.sendCommand('hotbar.2 1')
        rob.sendCommand('hotbar.2 0')
    else: # 'stone', etc.
        if wooden_pickaxe:
            rob.sendCommand('hotbar.1 1')
            rob.sendCommand('hotbar.1 0')
        else:
            rob.sendCommand('hotbar.2 1')
            rob.sendCommand('hotbar.2 0')
    
# Mine not just one block, but everything in range
def mineWhileInRange(rob):
    logging.info("\tinside mineWhileInRange")
    rob.sendCommand('attack 1')
    rob.observeProcCached()
    while rob.getCachedObserve('getLineOfSights') is None or rob.getCachedObserve('getLineOfSights', 'inRange'):
        sleep(0.02)
        rob.observeProcCached()
        chooseTool(rob)
    rob.sendCommand('attack 0')

# A higher-level skill for getting sticks
def getSticks(rob):
    logging.info("\tinside getSticks")
    # repeat 3 times, because the initial target can be wrong due to tallgrass
    # or imprecise direction to a distant tree
    target_name = log_names
    for i in range(3):
        target = search4blocks(rob, target_name)
        dist = lookAt(rob, target[1:4])
        runStraight(rob, dist, True)
        target_name = [target[0]]

    target = rob.nearestFromGrid(target_name)
    while target is not None:
        lookAt(rob, target)
        if not mineAtSight(rob):
            break
        target = rob.nearestFromEntities(target_name)
        if target is not None:
            runStraight(rob, lookAt(rob, target), True)
        target = rob.nearestFromGrid(target_name)

    while True: # [] != None as well
        filtered_inv = rob.softFilterInventoryItem('log')
        if filtered_inv == []:
            break
        for f_inv in filtered_inv:
            name_prefix = f_inv['type'].split("_")[0]
            rob.craft(name_prefix+'_planks')

    rob.craft('stick')


# A very simple skill for leaving a flat shaft mined in a certain direction
def leaveShaft(rob, angle):
    logging.info("\tinside leaveShaft")
    lookDir(rob, 0, angle)
    rob.sendCommand('move 1')
    rob.sendCommand('jump 1')
    while rob.waitNotNoneObserve('getAgentPos')[1] < 30.:
        sleep(0.1)
    sleep(1.)
    rob.stopMove()


# Making a shaft in a certain direction
def mineStone(rob):
    logging.info("\tinside mineStone")
    lookDir(rob, math.pi/4, 0.0)
    strafeCenterX(rob)
    while True:
        mineWhileInRange(rob)
        runStraight(rob, 1)
        stones = rob.filterInventoryItem('cobblestone')
        if stones != None and stones != [] and stones[0]['quantity'] >= 3: break


# The skill that will most likely fail: it's not that easy to find iron ore and coal
# without even looking around
def mineIron(rob):
    logging.info("\tinside mineIron")
    strafeCenterX(rob)
    while rob.waitNotNoneObserve('getAgentPos')[1] > 22.:
        mineWhileInRange(rob)
        runStraight(rob, 1)
    rob.sendCommand('move 1')
    rob.sendCommand('attack 1')
    while True:
        sleep(0.1)
        rob.observeProcCached()
        chooseTool(rob)
        iron_ore = rob.filterInventoryItem('iron_ore')
        coal = rob.filterInventoryItem('coal')
        if iron_ore != None and iron_ore != [] and iron_ore[0]['quantity'] >= 3 and \
           coal != None and coal != [] and coal[0]['quantity'] >= 3:
               rob.craft('iron_ingot')
               rob.craft('iron_ingot')
               rob.craft('iron_ingot')
               rob.craft('iron_pickaxe')
               break
        pickaxe = rob.filterInventoryItem('stone_pickaxe')
        if pickaxe == []:
            break


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the mission")
    miss = mb.MissionXML(
        agentSections=[mb.AgentSection(name='Agent',
                                       agentstart=mb.AgentStart([1, 67, 1, 1]))])
    world = mb.defaultworld(
        seed='5',
        forceReset="false",
        forceReuse="true")
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    mc = MalmoConnector(miss)
    mc.safeStart()

    rob = RobustObserver(mc)
    # fixing bug with falling through while reconnecting
    logging.info("Initializing the starting position")
    sleep(2)
    rob.sendCommand("jump 1")
    sleep(0.1)
    rob.sendCommand("jump 0")
    lookDir(rob, 0, 0)

    logging.info("The first search for sticks")
    getSticks(rob)

    logging.info("Trying to craft a crafting table")
    rob.craft('crafting_table')

    logging.info("Trying to craft a wooden pickaxe")
    rob.craft('wooden_pickaxe')
    sleep(0.1)
    pickaxe = rob.filterInventoryItem('wooden_pickaxe')
    if pickaxe == []:
        print("Failed")
        exit()

    # put pickaxe into inventory_0 == hotbar.1 slot
    rob.sendCommand('swapInventoryItems 0 ' + str(pickaxe[0]['index']))

    logging.info("Mining stones")
    mineStone(rob)

    logging.info("Crafting stone_pickaxe")
    rob.craft('stone_pickaxe')
    stone_pickaxe = rob.filterInventoryItem('stone_pickaxe')
    # put pickaxe into inventory_1 == hotbar.2 slot
    rob.sendCommand('swapInventoryItems 0 ' + str(stone_pickaxe[0]['index']))

    #climbing up
    logging.info("Leaving the shaft")
    leaveShaft(rob, math.pi)
    rob.sendCommand('move 1')
    rob.sendCommand('attack 1')
    sleep(3)
    rob.stopMove()

    logging.info("Getting sticks once again")
    getSticks(rob)

    logging.info("Mining for iron")
    lookDir(rob, math.pi/4, 0.0)
    mineIron(rob)
    logging.info("Leaving the shaft")
    leaveShaft(rob, math.pi)

    if not rob.filterInventoryItem('iron_pickaxe'):
        logging.info("One more attempt of iron mining")
        rob.craft('stone_pickaxe')
        lookDir(rob, math.pi/4, math.pi)
        mineIron(rob)
        leaveShaft(rob, 0.0)

