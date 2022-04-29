import logging
import json
import time
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver
from experiments import common
from experiments.common import stop_motion, grid_to_vec_walking, direction_to_target



def init_mission(mc, start_x=None, start_y=None):
    want_depth = False
    video_producer = mb.VideoProducer(width=320 * 4,
                                      height=240 * 4, want_depth=want_depth)

    obs = mb.Observations()
    obs.gridNear = [[-1, 1], [-2, 1], [-1, 1]]


    agent_handlers = mb.AgentHandlers(observations=obs, video_producer=video_producer)

    print('starting at ({0}, {1})'.format(start_x, start_y))

    #miss = mb.MissionXML(namespace="ProjectMalmo.microsoft.com",
    miss = mb.MissionXML(
                         agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers,
                                      #    depth
             agentstart=mb.AgentStart([start_x, 74.0, start_y, 1]))])
    flat_json = {"biome":"minecraft:plains",
                 "layers":[{"block":"minecraft:diamond_block","height":1}],
                 "structures":{"structures": {"village":{}}}}

    flat_param = "3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"
    flat_json = json.dumps(flat_json).replace('"', "%ESC")
    world = mb.defaultworld(
        seed='5',
        forceReuse="true",
        forceReset="false")
    flat_world = mb.flatworld(flat_json,
                    seed='43',
                    )
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    if mc is None:
        mc = MalmoConnector(miss)
        obs = RobustObserver(mc)
    else:
        mc.setMissionXML(miss)
    return mc, obs


def test_basic_motion():
    pass 


def main():
    import MalmoPython
#    MalmoPython.setLoggingComponent(MalmoPython.LoggingComponent.LOG_TCP, True)
#    MalmoPython.setLogging('log.txt', MalmoPython.LoggingSeverityLevel.LOG_FINE)

    start = 316.5, 5375.5
    start = (-108.0, -187.0)
    mc, obs = init_mission(None, start_x=start[0], start_y=start[1]) 

    mc.safeStart()
    time.sleep(4)
    print('sending command')
    mc.sendCommand('move 1')
        
    #print('send chat')
    #obs.sendCommand("chat /difficulty peaceful")
    time.sleep(1)
    print('sending command')
    mc.sendCommand('move 1')
        

    time.sleep(1)
    print('sending command move 0')
    mc.sendCommand('move 0')
    import pdb;pdb.set_trace()
    while True:
       mc.observeProc()
       time.sleep(2)
       print('waiting')
       img_frame = mc.frames[0] 
       if img_frame is None:
           print(img_frame)
       ray = mc.getLineOfSights()
       print(ray)


        
if __name__ == '__main__':
   main()
