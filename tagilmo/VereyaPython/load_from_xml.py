import xml.etree.ElementTree as ET
import tagilmo.utils.mission_builder as mb
from .xml_util import xml_to_dict, remove_namespaces

def _createAbout(aboutRoot = None):
    about = mb.About()
    if aboutRoot is None or len(list(aboutRoot)) == 0:
        return about
    summary = aboutRoot.find("Summary")
    if summary is not None:
        about.summary = summary.text
    return about


def _createModSettings(msRoot = None):
    ms = mb.ModSettings()
    if msRoot is None or len(list(msRoot)) == 0:
        return ms
    
    msTick = msRoot.find("MsPerTick")
    if msTick is not None:
        ms.ms_per_tick = msTick.text
    return ms


def _createTime(timeRoot = None):
    if timeRoot is None or len(list(timeRoot)) == 0:
        return None, None
    
    time_start = None
    if timeRoot.find("StartTime") is not None:
        time_start = timeRoot.find("StartTime").text

    time_pass = None
    if timeRoot.find("AllowPassageOfTime") is not None:
        time_pass = timeRoot.find("AllowPassageOfTime").text
        
    return time_start, time_pass
    
    
def _createServerInitialConditions(initConditionsRoot = None):
    if initConditionsRoot is None or len(list(initConditionsRoot)) == 0:
        return mb.ServerInitialConditions()
    
    time_start, time_pass = _createTime(initConditionsRoot.find("Time"))
    
    weather = None
    if initConditionsRoot.find("Weather") is not None:
        weather = initConditionsRoot.find("Weather").text
        
    allow_spawning = "true"
    if initConditionsRoot.find("AllowSpawning") is not None:
        allow_spawning = initConditionsRoot.find("AllowSpawning").text
    
    initial_conditions = mb.ServerInitialConditions(time_start_string=time_start,
                                                    time_pass_string=time_pass,
                                                    weather_string=weather,
                                                    spawning_string=allow_spawning)
    return initial_conditions

def _createDrawingDecorator(drawingDecoratorRoot = None):
    if drawingDecoratorRoot is None or len(list(drawingDecoratorRoot)) == 0:
        return mb.DrawingDecorator()
    decorators = []
    drawing_elements = list(drawingDecoratorRoot)
    for el in drawing_elements:
        match el.tag:
            case "DrawCuboid":
                decorators.append(mb.DrawCuboid(el.attrib["x1"], el.attrib["y1"], el.attrib["z1"], 
                                                el.attrib["x2"], el.attrib["y2"], el.attrib["z2"],
                                                el.attrib["type"]))
            case "DrawBlock":
                decorators.append(mb.DrawBlock(el.attrib["x"], el.attrib["y"], el.attrib["z"],
                                               el.attrib["type"]))
            case "DrawLine":
                decorators.append(mb.DrawLine(el.attrib["x1"], el.attrib["y1"], el.attrib["z1"], 
                                                el.attrib["x2"], el.attrib["y2"], el.attrib["z2"],
                                                el.attrib["type"]))
            case "DrawItem":
                decorators.append(mb.DrawItem(el.attrib["x"], el.attrib["y"], el.attrib["z"],
                                               el.attrib["type"]))
            case _:
                continue
    return mb.DrawingDecorator(decorators)
                

def _createServerHandlers(handlersRoot = None):
    serverHandlers = mb.ServerHandlers()
    if handlersRoot is None or len(list(handlersRoot)) == 0:
        return serverHandlers
    
    world_generator = mb.defaultworld()
    flat_gen = handlersRoot.find("FlatWorldGenerator")
    if flat_gen is not None:
        world_generator = mb.flatworld(flat_gen.attrib["generatorString"])
    serverHandlers.worldgenerator = world_generator
    
    serverHandlers.drawingdecorator = _createDrawingDecorator(handlersRoot.find("DrawingDecorator"))
    
    time_limit = None
    if handlersRoot.find("ServerQuitFromTimep") is not None:
        time_limit = handlersRoot.find("ServerQuitFromTimep").attrib["timeLimitMs"]
    serverHandlers.timeLimitsMs = time_limit
        
    return serverHandlers


def _createServerSection(serverSectionRoot = None):
    init_conditions = _createServerInitialConditions(serverSectionRoot.find("ServerInitialConditions"))
    handlers = _createServerHandlers(serverSectionRoot.find('ServerHandlers'))
    return mb.ServerSection(handlers=handlers, initial_conditions=init_conditions)


def _createCommands(commandsRoot= None):
    if commandsRoot is None:
        return mb.Commands()
    # TODO: parsing non-empty root
    return mb.Commands()
    

def _createRewardForTouchingBlockType(rewardForTouchingBlockTypeRoot = None):
    if rewardForTouchingBlockTypeRoot is None or len(list(rewardForTouchingBlockTypeRoot)) == 0:
        return mb.RewardForTouchingBlockType()
    rewards = []
    reward_blocks = rewardForTouchingBlockTypeRoot.findall("Block")
    for block in reward_blocks:
            rewards.append(mb.Block(reward=block.attrib['reward'], 
                                    blockType=block.attrib['type'], 
                                    behaviour=block.attrib['behaviour']))
    return mb.RewardForTouchingBlockType(rewards)


def _createAgentQuitFromTouchingBlockType(agentQuitFromTouchingBlockType = None):
    if agentQuitFromTouchingBlockType is None or len(list(agentQuitFromTouchingBlockType)) == 0:
        return mb.AgentQuitFromTouchingBlockType()
    quit_blocks = []
    blocks = agentQuitFromTouchingBlockType.findall("Block")
    for block in blocks:
        quit_blocks.append(mb.Block(blockType=block.attrib['type']))
    return mb.AgentQuitFromTouchingBlockType(quit_blocks)


def _createAgentHandlers(agentHandlersRoot = None):
    if agentHandlersRoot is None:
        return mb.AgentHandlers()
    obsFullStats = None
    if agentHandlersRoot.find("ObservationFromFullStats") is not None:
        obsFullStats = True
    obs = mb.Observations(bFullStats=obsFullStats)
    video_producer = mb.VideoProducer()
    if agentHandlersRoot.find("VideoProducer") is not None:
        producer = agentHandlersRoot.find("VideoProducer")
        video_producer = mb.VideoProducer(height=int(producer.find("Height").text), 
                                          width=int(producer.find("Width").text), 
                                          want_depth=producer.attrib["want_depth"] == "true")
    commands = _createCommands()
    
    rewards_for_touching = None
    if agentHandlersRoot.find("RewardForTouchingBlockType") is not None:
        rewards_for_touching = _createRewardForTouchingBlockType(agentHandlersRoot.find("RewardForTouchingBlockType"))
        
    reward_for_sending_command = None
    if agentHandlersRoot.find("RewardForSendingCommand") is not None:
        reward = agentHandlersRoot.find("RewardForSendingCommand").attrib['reward']
        reward_for_sending_command = mb.RewardForSendingCommand(reward=reward)
        
    agent_quit = None
    if agentHandlersRoot.find('AgentQuitFromTouchingBlockType') is not None:
        agent_quit = _createAgentQuitFromTouchingBlockType(agentHandlersRoot.find('AgentQuitFromTouchingBlockType'))
        
    return mb.AgentHandlers(commands=commands, observations=obs, video_producer=video_producer, 
                            rewardForTouchingBlockType=rewards_for_touching,
                            rewardForSendingCommand=reward_for_sending_command, agentQuitFromTouchingBlockType=agent_quit)
    

def _createAgentSection(agentSectionRoot = None):
    
    if agentSectionRoot is None or len(agentSectionRoot) == 0:
        return [mb.AgentSection()]
    
    sections = []
    for agent in agentSectionRoot:
        agent_section = mb.AgentSection()
        if agent.find("mode") is not None:
            agent_section.mode = agent.attrib["mode"]
            
        if agent.find("Name") is not None:
            agent_section.name = agent.find("Name").text
            
        if agent.find("AgentStart") is not None:
            xyzp = None
            if agent.find("AgentStart").find("Placement") is not None:
                placement = agent.find("AgentStart").find("Placement")
                xyzp = [float(v) for _,v in placement.items()][:4] #yaw is not in constructor
            agent_start = mb.AgentStart(place_xyzp=xyzp)
            agent_section.agentstart = agent_start
            
        agent_section.agenthandlers = _createAgentHandlers(agent.find("AgentHandlers"))
            
        sections.append(agent_section)
        
    return sections


def _createMissionXML(missionRoot):
    mission = mb.MissionXML()
    if missionRoot is None or len(list(missionRoot)) == 0:
        return mission
    
    mission.about = _createAbout(missionRoot.find("About"))
    mission.modSettings = _createModSettings(missionRoot.find("ModSettings"))
    mission.serverSection = _createServerSection(missionRoot.find("ServerSection"))
    mission.agentSections = _createAgentSection(missionRoot.findall("AgentSection"))
        
    return mission


def load_mission(path):
    tree = ET.parse(path)
    root = tree.getroot()
    remove_namespaces(root)
    miss = _createMissionXML(root)
    return miss