import xml.etree.ElementTree as ET
import tagilmo.utils.mission_builder as mb
from .xml_util import xml_to_dict, remove_namespaces

def _createAbout(aboutDict):
    return mb.About(summary_string=aboutDict['Summary']['text'])


def _createModSettings(msDict):
    return mb.ModSettings(ms_per_tick=msDict['MsPerTick']['text'])


def _createServerInitialConditions(initConditionsDict):
    initial_conditions = mb.ServerInitialConditions(time_start_string=initConditionsDict['Time']['StartTime']['text'],
                                                    time_pass_string=initConditionsDict['Time']['AllowPassageOfTime']['text'],
                                                    weather_string=initConditionsDict['Weather']['text'],
                                                    spawning_string=initConditionsDict['AllowSpawning']['text'])
    return initial_conditions


def _createServerHandlers(handlersDict):
    serverHandlers = mb.ServerHandlers()
    
    if 'FlatWorldGenerator' in handlersDict:
        world_generator = mb.flatworld(handlersDict['FlatWorldGenerator']['generatorString'])
    serverHandlers.worldgenerator = world_generator
    decorators = []
    drawing_elements = handlersDict['DrawingDecorator']
    for drawing_element in drawing_elements:
        els = drawing_elements[drawing_element]
        if isinstance(els, dict):
            els = [els]
        for el in els:
            match drawing_element:
                case 'DrawCuboid':
                    decorators.append(mb.DrawCuboid(x1=el['x1'], y1=el['y1'], z1=el['z1'],
                                                        x2=el['x2'], y2=el['y2'], z2=el['z2'],
                                                        blockType=el['type']))
                    
                case 'DrawBlock':
                    decorators.append(mb.DrawBlock(x=el['x'], y=el['y'], z=el['z'],
                                                        blockType=el['type']))
                    
                case 'DrawLine':
                    decorators.append(mb.DrawLine(x1=el['x1'], y1=el['y1'], z1=el['z1'],
                                                        x2=el['x2'], y2=el['y2'], z2=el['z2'],
                                                        blockType=el['type']))

                case 'DrawItem':
                    decorators.append(mb.DrawItem(x=el['x'], y=el['y'], z=el['z'], itemType=el['type']))    
                
    serverHandlers.drawingdecorator = mb.DrawingDecorator(decorators)
    if 'ServerQuitFromTimeUp' in handlersDict:
        serverHandlers.timeLimitsMs = handlersDict['ServerQuitFromTimeUp']['timeLimitMs']
        
    return serverHandlers


def _createServerSection(serverSectionDict):
    init_conditions = _createServerInitialConditions(serverSectionDict['ServerInitialConditions'])
    handlers = _createServerHandlers(serverSectionDict['ServerHandlers'])
    return mb.ServerSection(handlers=handlers, initial_conditions=init_conditions)


def _createCommands(commandsDict = None):
    if not commandsDict:
        return mb.Commands()
    

def _createRewardForTouchingBlockType(rewardForTouchingBlockTypeDict):
    rewards = []
    for block_element in rewardForTouchingBlockTypeDict:
        rwrds = rewardForTouchingBlockTypeDict[block_element]
        if isinstance(rwrds, dict):
            rwrds = [rwrds]
        for rwrd in rwrds:
            rewards.append(mb.Block(reward=rwrd['reward'], blockType=rwrd['type'], behaviour=rwrd['behaviour']))
    return mb.RewardForTouchingBlockType(rewards)


def _createAgentQuitFromTouchingBlockType(agentQuitFromTouchingBlockType):
    quit_blocks = []
    for block_element in agentQuitFromTouchingBlockType:
        qt_blcks = agentQuitFromTouchingBlockType[block_element]
        if isinstance(qt_blcks, dict):
            qt_blcks = [qt_blcks]
        for blk in qt_blcks:
            quit_blocks.append(mb.Block(blockType=blk['type']))
    return mb.AgentQuitFromTouchingBlockType(quit_blocks)


def _createAgentHandlers(agentHandlersDict = None):
    if not agentHandlersDict:
        return mb.AgentHandlers()
    
    obs = mb.Observations(bFullStats=('ObservationFromFullStats' in agentHandlersDict))
    video_producer = mb.VideoProducer()
    if 'VideoProducer' in agentHandlersDict:
        producer = agentHandlersDict['VideoProducer']
        video_producer = mb.VideoProducer(height=producer['Height']['text'], width=producer['Width']['text'], want_depth=producer['want_depth'])
    else:
        video_producer = mb.VideoProducer()
    commands = _createCommands()
    
    rewards_for_touching = None
    if 'RewardForTouchingBlockType' in agentHandlersDict:
        rewards_for_touching = _createRewardForTouchingBlockType(agentHandlersDict['RewardForTouchingBlockType'])
        
    reward_for_sending_command = None
    if 'RewardForSendingCommand' in agentHandlersDict:
        reward_for_command = agentHandlersDict['RewardForSendingCommand']
        reward_for_sending_command = mb.RewardForSendingCommand(reward=reward_for_command['reward'])
        
    agent_quit = None
    if 'AgentQuitFromTouchingBlockType' in agentHandlersDict:
        agent_quit = _createAgentQuitFromTouchingBlockType(agentHandlersDict['AgentQuitFromTouchingBlockType'])
        
    return mb.AgentHandlers(commands=commands, observations=obs, video_producer=video_producer, rewardForTouchingBlockType=rewards_for_touching,
                            rewardForSendingCommand=reward_for_sending_command, agentQuitFromTouchingBlockType=agent_quit)
    

def _createAgentSection(agentSectionDict = None):
    if not agentSectionDict:
        return [mb.AgentSection()]
    
    sections = []
    if isinstance(agentSectionDict, dict):
        agents = [agentSectionDict]
    else:
        agents = agentSectionDict
    for agent in agents:
        agent_section = mb.AgentSection()
        if 'mode' in agent:
            agent_section.mode = agent['mode']
            
        if 'Name' in agent:
            agent_section.name = agent['Name']['text']
            
        if 'AgentStart' in agent:
            xyzp = [float(v) for _,v in agent['AgentStart']['Placement'].items()][:4] #yaw is not in constructor
            agent_start = mb.AgentStart(place_xyzp=xyzp)
            agent_section.agentstart = agent_start
            
        if 'AgentHandlers' in agent:
            agent_section.agenthandlers = _createAgentHandlers(agent['AgentHandlers'])
            
        sections.append(agent_section)
        
    return sections


def _createMissionXML(missionDict):
    mission = mb.MissionXML()
    if 'About' in missionDict:
        mission.about = _createAbout(missionDict['About'])
    if 'ModSettings' in missionDict:
        mission.modSettings = _createModSettings(missionDict['ModSettings'])
    if 'ServerSection' in missionDict:
        mission.serverSection = _createServerSection(missionDict['ServerSection'])
    if 'AgentSection' in missionDict:
        mission.agentSections = _createAgentSection(missionDict['AgentSection'])
        
    return mission


def load_mission(path):
    tree = ET.parse(path)
    root = tree.getroot()
    remove_namespaces(root)
    parsed_dict = xml_to_dict(root)
    miss = _createMissionXML(parsed_dict)
    return miss