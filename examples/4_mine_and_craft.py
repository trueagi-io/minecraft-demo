import math
from time import sleep
from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb

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

passableBlocks = ['air', 'water', 'lava', 'double_plant', 'tallgrass', 'reeds', 'red_flower', 'yellow_flower']


# ============== some helper functions ==============

def normAngle(angle):
    while (angle < -math.pi): angle += 2 * math.pi
    while (angle > math.pi): angle -= 2 * math.pi
    return angle

def stopMove(mc):
    mc.sendCommand("move 0")
    mc.sendCommand("turn 0")
    mc.sendCommand("pitch 0")
    mc.sendCommand("jump 0")
    mc.sendCommand("strafe 0")

def nearestFromGrid(mc, obj):
    mc.observeProc()
    while mc.getNearGrid() is None:
        sleep(0.02)
        mc.observeProc()
    grid = mc.getNearGrid()
    d2 = 10000
    target = None
    for i in range(len(grid)):
        if grid[i] != obj: continue
        [x, y, z] = mc.gridIndexToPos(i)
        d2c = x * x + y * y + z * z
        if d2c < d2:
            d2 = d2c
            target = mc.gridIndexToAbsPos(i)
    return target

def nearestFromEntities(mc, obj):
    mc.observeProc()
    while mc.getNearEntities() is None:
        sleep(0.02)
        mc.observeProc()
    ent = mc.getNearEntities()
    d2 = 10000
    pos = mc.getAgentPos()
    target = None
    for e in ent:
        if e['name'] != obj: continue
        [x, y, z] = [e['x'], e['y'], e['z']]
        if abs(y - pos[1]) > 1: continue
        d2c = (x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1]) + (z - pos[2]) * (z - pos[2])
        if d2c < d2:
            d2 = d2c
            target = [x, y, z]
    return target

def getInvSafe(mc, item):
    while True:
        sleep(0.3)
        mc.observeProc()
        if mc.isInventoryAvailable(): return mc.filterInventoryItem(item)


# ============== some hand-coded skills ==============

# This function is executed to run to a visible object,
# but it doesn't check if the path is safe
def runStraight(mc, dist):
    mc.observeProc()
    while mc.getAgentPos() is None:
        sleep(0.1)
        mc.observeProc()
    start = mc.getAgentPos()
    mc.sendCommand("move 1")
    for t in range(3000):
        sleep(0.1)
        mc.observeProc()
        pos = mc.getAgentPos()
        if pos is None: continue
        if dist * dist < (pos[0] - start[0]) * (pos[0] - start[0]) + (pos[2] - start[2]) * (pos[2] - start[2]):
            break
        if mc.isLineOfSightAvailable() and mc.getLineOfSight('distance') < 0.5 and \
           not mc.getLineOfSight('distance') in passableBlocks and \
           not mc.getLineOfSight('type') in passableBlocks:
            break
    mc.sendCommand("move 0")

# Just look at a specified direction
def lookDir(mc, pitch, yaw):
    for t in range(3000):
        sleep(0.02)
        mc.observeProc()
        aPos = mc.getAgentPos()
        if aPos is None:
            continue
        dPitch = normAngle(pitch - aPos[3]*math.pi/180.)
        dYaw = normAngle(yaw - aPos[4]*math.pi/180.)
        if abs(dPitch)<0.02 and abs(dYaw)<0.02: break
        mc.sendCommand("turn " + str(dYaw*0.4))
        mc.sendCommand("pitch " + str(dPitch*0.4))
    mc.sendCommand("turn 0")
    mc.sendCommand("pitch 0")

# Look at a specified location
def lookAt(mc, pos):
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

def strafeCenterX(mc):
    mc.sendCommand('strafe 0.1')
    for t in range(200):
        sleep(0.02)
        mc.observeProc()
        aPos = mc.getAgentPos()
        if aPos is None:
            continue
        if int(abs(aPos[0])*10+0.5)%10==5:
            break
    stopMove(mc)
    

# A simplistic search behavior
# Note that we don't use video input and rely on a small Grid and Ray,
# so our agent can miss objects visible by human
def search4blocks(mc, blocks):
    for t in range(3000):
        sleep(0.02)
        mc.observeProc()
        if mc.getLineOfSight('type') in blocks:
            stopMove(mc)
            return [mc.getLineOfSight('type'), mc.getLineOfSight('x'), mc.getLineOfSight('y'), mc.getLineOfSight('z')]
        grid = mc.getNearGrid()
        if grid is not None:
            for i in range(len(grid)):
                if grid[i] in blocks:
                    stopMove(mc)
                    return [grid[i]] + mc.gridIndexToAbsPos(i)
        gridSlice = mc.gridInYaw()
        if gridSlice == None:
            continue
        ground = gridSlice[(len(gridSlice) - 1) // 2 - 1]
        solid = all([not (b in passableBlocks) for b in ground])
        wayLv0 = gridSlice[(len(gridSlice) - 1) // 2]
        wayLv1 = gridSlice[(len(gridSlice) - 1) // 2 + 1]
        passWay = all([b in passableBlocks for b in wayLv0]) and all([b in passableBlocks for b in wayLv1])
        turnVel = 0.25 * math.sin(t * 0.05)
        if not (passWay and solid):
            turnVel -= 1
        pitchVel = -0.015 * math.cos(t * 0.03)
        mc.sendCommand("move 1")
        mc.sendCommand("turn " + str(turnVel))
        mc.sendCommand("pitch " + str(pitchVel))
    stopMove(mc)
    return None

# Just attacking while the current block is not destroyed
# assuming nothing else happens
def mineAtSight(mc):
    sleep(0.1)
    mc.observeProc()
    if mc.getLineOfSight('type') is None or not mc.getLineOfSight('inRange'): return False
    dist = mc.getLineOfSight('distance')
    obj = mc.getLineOfSight('type')
    mc.sendCommand('attack 1')
    for t in range(100):
        if mc.getLineOfSight('type') is None or abs(dist - mc.getLineOfSight('distance')) > 0.01 or obj != mc.getLineOfSight('type'):
            mc.sendCommand('attack 0')
            return True
        sleep(0.1)
        mc.observeProc()
    mc.sendCommand('attack 0')
    return False

# A skill to choose a tool for mining (in the context of the current example)
def chooseTool(mc):
    if not mc.isLineOfSightAvailable(): return
    wooden_pickaxe = mc.filterInventoryItem('wooden_pickaxe')
    if wooden_pickaxe and wooden_pickaxe[0]['index'] != 0:
        mc.sendCommand('swapInventoryItems 0 ' + str(wooden_pickaxe[0]['index']))
    stone_pickaxe = mc.filterInventoryItem('stone_pickaxe')
    if stone_pickaxe and stone_pickaxe[0]['index'] != 1:
        mc.sendCommand('swapInventoryItems 1 ' + str(stone_pickaxe[0]['index']))
    if mc.getLineOfSight('type') in ['dirt', 'grass']:
        mc.sendCommand('hotbar.9 1')
        mc.sendCommand('hotbar.9 0')
    elif mc.getLineOfSight('type') in ['iron_ore']:
        mc.sendCommand('hotbar.2 1')
        mc.sendCommand('hotbar.2 0')
    else: # 'stone', etc.
        if wooden_pickaxe:
            mc.sendCommand('hotbar.1 1')
            mc.sendCommand('hotbar.1 0')
        else:
            mc.sendCommand('hotbar.2 1')
            mc.sendCommand('hotbar.2 0')
    
# Mine not just one block, but everything in range
def mineWhileInRange(mc):
    mc.sendCommand('attack 1')
    mc.observeProc()
    while not mc.isLineOfSightAvailable() or mc.getLineOfSight('inRange'):
        sleep(0.02)
        mc.observeProc()
        chooseTool(mc)
    mc.sendCommand('attack 0')

# A higher-level skill for getting sticks
def getSticks(mc):
    # repeat 3 times, because the initial target can be wrong due to tallgrass
    # or imprecise direction to a distant tree
    for i in range(3):
        target = search4blocks(mc, ['log', 'leaves'])
        dist = lookAt(mc, target[1:4])
        runStraight(mc, dist)

    target = nearestFromGrid(mc, 'log')
    while not target is None:
        lookAt(mc, target)
        if not mineAtSight(mc):
            break
        target = nearestFromEntities(mc, 'log')
        if not target is None:
            runStraight(mc, lookAt(mc, target))
        target = nearestFromGrid(mc, 'log')

    while mc.filterInventoryItem('log') != []: # [] != None as well
        mc.sendCommand('craft planks')
        sleep(0.02)
        mc.observeProc()

    mc.sendCommand('craft stick')
    sleep(0.02)

# A very simple skill for leaving a flat shaft mined in a certain direction
def leaveShaft(mc):
    lookDir(mc, 0, math.pi)
    mc.sendCommand('move 1')
    mc.sendCommand('jump 1')
    while mc.getAgentPos() is None or mc.getAgentPos()[1] < 30.:
        sleep(0.1)
        mc.observeProc()
    stopMove(mc)

# Making a shaft in a certain direction
def mineStone(mc):
    lookDir(mc, math.pi/4, 0.0)
    strafeCenterX(mc)
    while True:
        mineWhileInRange(mc)
        runStraight(mc, 1)
        stones = mc.filterInventoryItem('cobblestone')
        if stones != None and stones != [] and stones[0]['quantity'] >= 3: break

# The skill that will most likely fail: it's not that easy to find iron ore and coal
# without even looking around
def mineIron(mc):
    strafeCenterX(mc)
    while mc.getAgentPos() is None or mc.getAgentPos()[1] > 22.:
        mineWhileInRange(mc)
        runStraight(mc, 1)
    mc.sendCommand('move 1')
    mc.sendCommand('attack 1')
    while True:
        sleep(0.1)
        mc.observeProc()
        chooseTool(mc)
        iron_ore = mc.filterInventoryItem('iron_ore')
        coal = mc.filterInventoryItem('coal')
        if iron_ore != None and iron_ore != [] and iron_ore[0]['quantity'] >= 3 and \
           coal != None and coal != [] and coal[0]['quantity'] >= 3:
               mc.sendCommand('craft iron_ingot')
               mc.sendCommand('craft iron_ingot')
               mc.sendCommand('craft iron_ingot')
               mc.sendCommand('craft iron_pickaxe')
               break
        pickaxe = mc.filterInventoryItem('stone_pickaxe')
        if mc.isInventoryAvailable() and pickaxe == []:
            break


miss = mb.MissionXML()
miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake", forceReset="true"))
miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
mc = MalmoConnector(miss)
mc.safeStart()
# fixing bug with falling through while reconnecting
sleep(2)
mc.sendCommand("jump 1")
sleep(0.1)
mc.sendCommand("jump 0")

lookDir(mc, 0, 0)

getSticks(mc)

mc.sendCommand('craft wooden_pickaxe')

pickaxe = getInvSafe(mc, 'wooden_pickaxe')
if pickaxe == []:
    print("Failed")
    exit()

# put pickaxe into inventory_0 == hotbar.1 slot
mc.sendCommand('swapInventoryItems 0 ' + str(pickaxe[0]['index']))

mineStone(mc)

mc.sendCommand('craft stone_pickaxe')
pickaxe = getInvSafe(mc, 'stone_pickaxe')
# put pickaxe into inventory_1 == hotbar.2 slot
mc.sendCommand('swapInventoryItems 1 ' + str(pickaxe[0]['index']))

#climbing up
leaveShaft(mc)

mc.sendCommand('move 1')
mc.sendCommand('attack 1')
sleep(3)
stopMove(mc)
getSticks(mc)

lookDir(mc, math.pi/4, 0.0)
mineIron(mc)
leaveShaft(mc)

if not getInvSafe(mc, 'iron_pickaxe'):
    lookDir(mc, math.pi/4, math.pi)
    mineIron(mc)
    leaveShaft(mc)
