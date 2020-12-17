from time import sleep
from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb


miss = mb.MissionXML()
# https://www.chunkbase.com/apps/superflat-generator
miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold(distance=15),biome_1(distance=15),village(size=2 distance=10),decoration,dungeon,lake,mineshaft(chance=0.04),lava_lake"))
#miss.addAgent(1)

mc = MalmoConnector(miss)
mc.safeStart()


fullStatKeys = ['XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw']
stats_old = [0]*len(fullStatKeys)
seenObjects = []
nearEntities = []
gridObjects = []

for i in range(600):
    mc.observeProc()

    stats_new = [mc.getFullStat(key) for key in fullStatKeys]
    if stats_new != stats_old and stats_new[0] != None:
        print(' '.join(['%s: %.2f' % (fullStatKeys[n], stats_new[n]) for n in range(len(stats_new))]))
        stats_old = stats_new

    crossObj = mc.getLineOfSight('type')
    if crossObj is not None:
        crossObj += ' ' + mc.getLineOfSight('hitType')
        if not crossObj in seenObjects:
            seenObjects += [crossObj]
            print('******** Novel object in line-of-sight : ', crossObj)
    
    nearEnt = mc.getNearEntities()
    if nearEnt != None:
        for e in nearEnt:
            if e['name'] != 'Agent-0':
                if not e['name'] in nearEntities:
                    nearEntities += [e['name']]
                    print('++++++++ Novel nearby entity: ', e['name'])
                elif abs(e['x'] - mc.getFullStat('XPos')) + abs(e['y'] - mc.getFullStat('YPos')) < 1:
                    print('!!!!!!!! Very close entity ', e['name'])

    grid = mc.getNearGrid()
    for o in (grid if grid is not None else []):
        if not o in gridObjects:
            gridObjects += [o]
            print('-------- Novel grid object: ', o)

    sleep(0.5)

# run the script and control the agent manually to see updates
