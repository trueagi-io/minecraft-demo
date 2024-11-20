class About:

    def __init__(self, summary_string=None):
        self.summary = summary_string

    def xml(self):
        _xml = '<About>\n'
        if self.summary:
            _xml += "<Summary>"+self.summary+"</Summary>"
        else:
            _xml += "<Summary/>";
        _xml += '\n</About>\n'
        return _xml


class ModSettings:

    def __init__(self, ms_per_tick=None):
        self.ms_per_tick = ms_per_tick

    def xml(self):
        _xml = '<ModSettings>\n'
        if self.ms_per_tick:
            _xml += "<MsPerTick>"+self.ms_per_tick+"</MsPerTick>"
        else:
            _xml += "<MsPerTick/>"
        _xml += '\n<ModSettings>\n'
        return _xml


class ServerInitialConditions:
    
    def __init__(self, day_always=False, time_start_string=None, time_pass_string=None,
                 weather_string=None, spawning_string="true", allowedmobs_string=None):
        self.day_always = day_always
        self.time_start = time_start_string
        self.time_pass = time_pass_string
        self.weather = weather_string
        self.spawning = spawning_string
        self.allowedmobs = allowedmobs_string

    def xml(self):
        _xml = '<ServerInitialConditions>\n'
        if self.day_always:
            _xml += '''    <Time>
        <StartTime>6000</StartTime>
        <AllowPassageOfTime>false</AllowPassageOfTime>
    </Time>
    <Weather>clear</Weather>\n'''
        # ignore time_start_string, time_pass, and weather_string
        else:
            if self.time_start or self.time_pass:
                _xml += '<Time>\n'
                if self.time_start:
                    _xml += '<StartTime>'+self.time_start+'</StartTime>\n';
                if self.time_pass:
                    _xml += '<AllowPassageOfTime>'+self.time_pass+'</AllowPassageOfTime>\n';
                _xml += '</Time>\n'
            if self.weather:
                # "clear", "rain", "thunder"?
                _xml += '<Weather>'+self.weather+'</Weather>\n'
        if self.allowedmobs: self.spawning = "true"
        if self.spawning:
            # "false" or "true"
            _xml += '<AllowSpawning>'+self.spawning+'</AllowSpawning>\n'
            if self.allowedmobs:
                _xml += '<AllowedMobs>'+self.allowedmobs+'</AllowedMobs>\n' #e.g. "Pig Sheep"
        _xml += '</ServerInitialConditions>\n'
        return _xml


def flatworld(generatorString, forceReset="false", seed=''):
    return '<FlatWorldGenerator generatorString="' + generatorString + '" forceReset="' + forceReset + '" seed="' + seed + '"/>'

def defaultworld(seed=None, forceReset=False, forceReuse=False):
    if isinstance(forceReset, bool):
        forceReset = 'true' if forceReset else 'false'
    world_str = '<DefaultWorldGenerator '
    if seed:
        world_str += 'seed="' + str(seed) + '" '
    if forceReset:
        world_str += 'forceReset="' + forceReset + '" '
    if forceReuse:
        world_str += 'forceReuse="' + forceReuse + '" '
    world_str += '/>'
    return world_str

def fileworld(uri2save, forceReset="false"):
    str = '<FileWorldGenerator '
    str += 'src="' + uri2save + '" '
    if forceReset:
        str += 'forceReset="' + forceReset + '" '
    str += '/>'
    return str


class ServerHandlers:
    
    def __init__(self, worldgenerator_xml=defaultworld(), alldecorators_xml=None,
                 bQuitAnyAgent=False, timeLimitsMs_string=None, drawingdecorator = None):
        self.worldgenerator = worldgenerator_xml
        self.alldecorators = alldecorators_xml
        self.bQuitAnyAgent = bQuitAnyAgent
        self.timeLimitsMs = timeLimitsMs_string
        self.drawingdecorator = drawingdecorator
        
    def xml(self):
        _xml = '<ServerHandlers>\n' + self.worldgenerator + '\n'
        if self.drawingdecorator:
           _xml += '<DrawingDecorator>\n' + self.drawingdecorator.xml() + '</DrawingDecorator>\n'
        #<BuildBattleDecorator> --
        #<MazeDecorator> --
        if self.alldecorators:
            _xml += self.alldecorators + '\n'
        if self.bQuitAnyAgent:
            _xml += '<ServerQuitWhenAnyAgentFinishes />\n'
        if self.timeLimitsMs:
            _xml += '<ServerQuitFromTimeUp timeLimitMs="' + self.timeLimitsMs +\
                '" description="Time limit" />\n'
        _xml += '</ServerHandlers>\n'
        return _xml


class ServerSection:

    def __init__(self, handlers=ServerHandlers(), initial_conditions=ServerInitialConditions()):
        self.handlers = handlers
        self.initial_conditions = initial_conditions

    def xml(self):
        return '<ServerSection>\n'+self.initial_conditions.xml()+self.handlers.xml()+'</ServerSection>\n'


class DrawingDecorator:
    def __init__(self, decorators = []):
        """
            Draw all given Draw objects

            Parameters:
                decorators (List[Union[DrawBlock, DrawCuboid, DrawItem, DrawLine]]) : a list of objects to be drawn.\nEach object can be one of the following types:
                    - DrawBlock: represents a block.
                    - DrawCuboid: represents a cuboid.
                    - DrawItem: represents an item.
                    - DrawLine: represents a line between two points.
        """
        self.decorators = decorators

    def xml(self):
        _xml = ""
        for decorator in self.decorators:
            _xml += decorator.xml()
        return _xml


class DrawBlock:
    def __init__(self, x, y, z, blockType):
        """
            Draw a block in world.

            Parameters:
                x (int | str): x coordinate.
                y (int | str): y coordinate.
                z (int | str): z coordinate.
                blockType (str): block that will be used.
        """
        self.x = x
        self.y = y
        self.z = z
        self.blockType = blockType

    def xml(self):
        return f'<DrawBlock x="{self.x}" y="{self.y}" z="{self.z}" type="{self.blockType}"/>\n'
    

class DrawCuboid:
    def __init__(self, x1, y1, z1, x2, y2, z2, blockType):
        """
            Draw a cuboid in world.

            Parameters:
                x1 (int | str): x coordinate of the first corner.
                y1 (int | str): y coordinate of the first corner.
                z1 (int | str): z coordinate of the first corner.
                x2 (int | str): x coordinate of the second corner.
                y2 (int | str): y coordinate of the second corner.
                z2 (int | str): z coordinate of the second corner.
                blockType (str): block that will be used.
        """
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.blockType = blockType

    def xml(self):
        return f'<DrawCuboid x1="{self.x1}" y1="{self.y1}" z1="{self.z1}" x2="{self.x2}" y2="{self.y2}" z2="{self.z2}" type="{self.blockType}"/>\n'


class DrawLine:
    def __init__(self, x1, y1, z1, x2, y2, z2, blockType):
        """
            Draw a line of blocks in world.

            Parameters:
                x1 (int | str): x coordinate of the first point.
                y1 (int | str): y coordinate of the first point.
                z1 (int | str): z coordinate of the first point.
                x2 (int | str): x coordinate of the second point.
                y2 (int | str): y coordinate of the second point.
                z2 (int | str): z coordinate of the second point.
                blockType (str): block that will be used.
        """
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.blockType = blockType

    def xml(self):
        return f'<DrawLine x1="{self.x1}" y1="{self.y1}" z1="{self.z1}" x2="{self.x2}" y2="{self.y2}" z2="{self.z2}" type="{self.blockType}"/>\n'


class DrawItem:
    def __init__(self, x, y, z, itemType):
        """
            Draw an item in world.

            Parameters:
                x (int | str): x coordinate.
                y (int | str): y coordinate.
                z (int | str): z coordinate.
                itemType (str): item that will be used.
        """
        self.x = x
        self.y = y
        self.z = z
        self.itemType = itemType

    def xml(self):
        return f'<DrawItem x="{self.x}" y="{self.y}" z="{self.z}" type="{self.itemType}"/>\n'


class DrawSphere:
    def __init__(self, x, y, z, radius, blockType):
        """
            Draw a block in world.

            Parameters:
                x (int | str): x coordinate.
                y (int | str): y coordinate.
                z (int | str): z coordinate.
                radius (int | str): radius.
                blockType (str): block that will be used.
        """
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.blockType = blockType

    def xml(self):
        return f'<DrawBlock x="{self.x}" y="{self.y}" z="{self.z}" radius="{self.radius}" type="{self.blockType}"/>\n'
    

class Commands:

    def __init__(self, bAll=True, bContinuous=None, bDiscrete=None, bInventory=None,
                 bSimpleCraft=None, bChat=None, bPlaceBlock=None, bControlMob=None):
        self.bAll = bAll
        self.bContinuous = bContinuous
        self.bDiscrete = bDiscrete
        self.bInventory = bInventory
        self.bSimpleCraft = bSimpleCraft
        self.bChat = bChat
        self.bPlaceBlock = bPlaceBlock
        self.bControlMob = bControlMob

    def xml(self):
        _xml = ""
        if self.bAll or self.bContinuous:
            _xml += "<ContinuousMovementCommands turnSpeedDegs=\"420\"/>\n"
        if self.bAll or self.bDiscrete:
            _xml += "<DiscreteMovementCommands />\n"
        if self.bAll or self.bInventory:
            _xml += "<InventoryCommands />\n"
        if self.bAll or self.bSimpleCraft:
            _xml += "<SimpleCraftCommands />\n"
        if self.bAll or self.bChat:
            _xml += "<ChatCommands />\n"
        if self.bAll or self.bPlaceBlock:
            _xml += "<BlockPlaceCommands />\n"
        if self.bAll or self.bControlMob:
            _xml += "<CommandForWheeledRobotNavigationMob/>"
        #<AbsoluteMovementCommands /> --
        #<MissionQuitCommands /> --
        #<HumanLevelCommands/> --
        #<TurnBasedCommands/> --
        return _xml


class Observations:

    def __init__(self, bAll=True, bRay=None, bFullStats=None,
            bInvent=None, bNearby=None, bGrid=None, bFindBlock=None, bChat=None, bRecipes=False, bItems=False,
            bHuman=None, bBlocksDrops=False, bSolidness=False):
        self.bAll = bAll
        self.bRay = bRay
        self.bFullStats = bFullStats
        self.bInvent = bInvent
        self.bNearby = bNearby
        self.bGrid = bGrid
        self.bFindBlock = bFindBlock
        self.gridNear = [[-5, 5], [-2, 2], [-5, 5]]
        self.gridBig = [[-25, 25], [-25, 25], [-25, 25]]
        self.bChat = bChat
        self.bRecipes = bRecipes
        self.bItems = bItems
        self.bHuman = bHuman
        self.bBlocksDrops = bBlocksDrops
        self.bSolidness = bSolidness

    def xml(self):
        _xml = ""
        if (self.bAll or self.bRay) and not (self.bRay == False):
            _xml += "<ObservationFromRay />\n"
        if (self.bAll or self.bFullStats) and not (self.bFullStats == False):
            _xml += "<ObservationFromFullStats />\n"
        if (self.bAll or self.bInvent) and not (self.bInvent == False):
            _xml += "<ObservationFromFullInventory  flat='false'/>\n"
        if (self.bAll or self.bRecipes):
            _xml += "<ObservationFromRecipes/>"
        if (self.bAll or self.bItems):
            _xml += "<ObservationFromItems/>"
        if (self.bAll or self.bHuman) and not (self.bHuman == False):
            _xml += "<ObservationFromHuman/>"
        if (self.bAll or self.bBlocksDrops):
            _xml += "<ObservationFromBlocksDrops/>"
        if (self.bAll or self.bSolidness):
            _xml += "<ObservationFromSolidness/>"
        # <ObservationFromHotBar /> --
        if (self.bAll or self.bNearby) and not (self.bNearby == False):
            # we don't need higher update_frequency; it seems we can get new observations in 0.1 with frequency=1
            # we don't need <Range name="r_close" xrange="2" yrange="2" zrange="2" update_frequency="1" /> separately,
            # because we can extract this information by ourselves and we don't need low frequency for distant entities
            # entities include mobs and items
            _xml += '''
<ObservationFromNearbyEntities>
    <Range name="ents_near" xrange="15" yrange="10" zrange="15" update_frequency="1" />
</ObservationFromNearbyEntities>'''
        if (self.bAll or self.bGrid) and not (self.bGrid == False):
            # Grid doesn't take the agent's orientation into account, so it should be symmetric
            # It rapidly becomes huge, so we have to limit our observations by a small grid
            # TODO? We may add [-2, -5, -2]x[2, -3, 2] and [-2, 3, -2]x[2, 5, 2] grids
            _xml += '''
<ObservationFromGrid>
    <Grid name="grid_near" absoluteCoords="false">
        <min x="'''+str(self.gridNear[0][0])+'" y="'+str(self.gridNear[1][0])+'" z="'+str(self.gridNear[2][0])+'''"/>
        <max x="'''+str(self.gridNear[0][1])+'" y="'+str(self.gridNear[1][1])+'" z="'+str(self.gridNear[2][1])+'''"/>
    </Grid>
</ObservationFromGrid>'''
        if (self.bAll or self.bFindBlock) and not (self.bFindBlock == False):
            _xml += '''
<ObservationFromFindBlock>
    <Grid name="grid_big" absoluteCoords="false">
        <min x="''' + str(self.gridBig[0][0]) + '" y="' + str(self.gridBig[1][0]) + '" z="' + str(
        self.gridBig[2][0]) + '''"/>
        <max x="''' + str(self.gridBig[0][1]) + '" y="' + str(self.gridBig[1][1]) + '" z="' + str(
        self.gridBig[2][1]) + '''"/>
    </Grid>
</ObservationFromFindBlock>'''
        if (self.bAll or self.bChat) and not (self.bChat == False):
            _xml += "<ObservationFromChat />\n"

        #<ObservationFromRecentCommands/>
        #<ObservationFromDiscreteCell/>
        #<ObservationFromSubgoalPositionList>
        #<ObservationFromDistance><Marker name="Start" x="0.5" y="227" z="0.5"/></ObservationFromDistance>
        #<ObservationFromTurnScheduler/> --
        return _xml


class VideoProducer:
    def __init__(self, height=0, width=0, want_depth=False):
        self.height = height
        self.width = width
        self.want_depth = want_depth

    def xml(self):
        return '<VideoProducer want_depth="{0}"> \
            <Width>{width}</Width> \
            <Height>{height}</Height> \
            </VideoProducer>'.format('true' if self.want_depth else 'false',
                width=self.width, height=self.height)


class ColourMapProducer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def xml(self):
        return '<ColourMapProducer> \
                   <Width>{width}</Width> \
                   <Height>{height}</Height> \
                </ColourMapProducer>'.format(width=self.width, height=self.height)


class RewardBlock:
    def __init__(self, reward, blockType, behavior):
        self.reward = reward
        self.blockType = blockType
        self.behavior = behavior

    def xml(self):
        return f'<Block reward="{self.reward}" type="{self.blockType}" behavior="{self.behavior}"/>\n'

class RewardForTouchingBlockType:
    def __init__(self, rewardBlocks = []):
        self.rewardBlocks = rewardBlocks

    def xml(self):
        _xml = "<RewardForTouchingBlockType>\n"
        for rewardBlock in self.rewardBlocks:
            _xml += rewardBlock.xml()
        _xml += "/RewardForTouchingBlockType>\n"
        return _xml


class RewardForSendingCommand:
    def __init__(self, reward):
        self.reward = reward
        
    def xml(self):
        return f'<RewardForSendingCommand reward="{self.reward}"/>\n'


class AgentHandlers:

    def __init__(self, commands=Commands(), observations=Observations(),
                 all_str='', video_producer=None, colourmap_producer=None, rewardForTouchingBlockType = None, rewardForSendingCommand=None):
        self.commands = commands
        self.observations = observations
        self.all_str = all_str
        self.video_producer = video_producer
        self.colourmap_producer = colourmap_producer
        self.rewardForTouchingBlockType = rewardForTouchingBlockType
        self.rewardForSendingCommand = rewardForSendingCommand

    def xml(self):
        _xml = '<AgentHandlers>\n'
        _xml += self.commands.xml()
        _xml += '' if self.rewardForTouchingBlockType is None else self.rewardForTouchingBlockType.xml()
        _xml += '' if self.rewardForSendingCommand is None else self.rewardForSendingCommand.xml()
        _xml += self.observations.xml()
        _xml += self.all_str
        _xml += '' if self.video_producer is None else self.video_producer.xml()
        _xml += '' if self.colourmap_producer is None else self.colourmap_producer.xml()
        _xml += '</AgentHandlers>\n'
        # <VideoProducer want_depth=... viewpoint=...> --
        # <DepthProducer> --
        # <ColourMapProducer> --
        # ...
        return _xml

    def hasVideo(self):
        if self.video_producer is None:
            return False
        return True

    def hasSegmentation(self):
        if self.colourmap_producer is None:
            return False
        return True


class AgentStart:

    def __init__(self, place_xyzp=None, inventory_list=None):
        # place_xyzp format: [0.5, 1.0, 0.5, 0]
        self.place = place_xyzp
        self.inventory = inventory_list

    def xml(self):
        if self.place or self.inventory:
            _xml = '<AgentStart>\n';
            if self.place:
                _xml += '<Placement x="' + str(self.place[0]) + '" y="' + str(self.place[1]) +\
                    '" z="' + str(self.place[2]) + '" pitch="' + str(self.place[3]) + '\"/>\n'
            if self.inventory:
                _xml += '<Inventory>\n'
                for item in self.inventory:
                    _xml += '<InventoryItem type="'
                    if type(item) == list:
                        _xml += item[0] + '"'
                        if len(item) > 1: _xml += ' quantity="' + str(item[1]) + '"'
                        if len(item) > 2: _xml += ' slot="' + str(item[2]) + '"'
                    else: _xml += item + '"'
                    _xml += '/>\n'
                _xml += '</Inventory>\n'
            _xml += '</AgentStart>\n'
        else: _xml = '<AgentStart/>'
        return _xml


class AgentSection:

    def __init__(self, mode="Survival", name="Agent-0", agentstart=AgentStart(), agenthandlers=AgentHandlers()):
        self.mode = mode
        self.name = name
        self.agentstart = agentstart
        self.agenthandlers = agenthandlers

    def xml(self):
        _xml = '<AgentSection mode="' + self.mode + '">\n'
        _xml += '<Name>' + self.name + '</Name>\n'
        _xml += self.agentstart.xml()
        _xml += self.agenthandlers.xml()
        _xml += '</AgentSection>\n'
        return _xml

    def hasVideo(self):
        if self.agenthandlers.hasVideo():
            return True
        return False

    def hasSegmentation(self):
        if self.agenthandlers.hasSegmentation():
            return True
        return False


class MissionXML:

    def __init__(self, about=About(), modSettings = ModSettings(), serverSection=ServerSection(), agentSections=[AgentSection()], namespace=None):
        self.namespace = namespace
        self.about = about
        self.modSettings = modSettings
        self.serverSection = serverSection
        self.agentSections = agentSections
    
    def hasVideo(self):
        for section in self.agentSections:
            if section.hasVideo():
                return True
        return False

    def hasSegmentation(self):
        for section in self.agentSections:
           if section.hasSegmentation():
               return True
        return False

    def setSummary(self, summary_string):
        self.about.summary = summary_string
    
    def setWorld(self, worldgenerator_xml):
        self.serverSection.handlers.worldgenerator = worldgenerator_xml
    
    def setTimeLimit(self, timeLimitMs):
        self.serverSection.handlers.timeLimitMs = str(timeLimitMs)
        
    def addAgent(self, nCount=1, agentSections=None):
        if agentSections:
            self.agentSections += agentSections
        else:
            for i in range(nCount):
                ag = AgentSection(name="Agent-"+str(len(self.agentSections)))
                self.agentSections += [ag]

    def setObservations(self, observations, nAgent=None):
        if nAgent is None:
            for ag in self.agentSections:
                ag.agenthandlers.observations = observations
        else:
            self.agentSections[nAgent].agenthandlers.observations = observations
    
    def getAgentNames(self):
        return [ag.name for ag in self.agentSections]

    def xml(self):
        namespace = self.namespace
        if namespace is None:
            namespace = 'ProjectMalmo.singularitynet.io'
        _xml = '''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://{0}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
'''.format(namespace)
        _xml += self.about.xml()
        # _xml += self.modSettings.xml()
        _xml += self.serverSection.xml()
        for agentSection in self.agentSections:
            _xml += agentSection.xml()
        _xml += '</Mission>'
        return _xml
