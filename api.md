
# tagilmo

Tagilmo provides high level api for iteractions with minecraft from python code

for installation of minecraft mod see https://github.com/trueagi-io/Vereya
install tagilmo with:
```
pip install git+https://github.com/trueagi-io/minecraft-demo.git
```

Two main modules in tagilmo are

tagilmo.utils.mission_builder - used to set up a world in minecraft.
tagilmo.utils.malmo_wrapper - used to control agent in the game


mission_builder difines various classes to build xml description of the world.

basic setup:
```
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserver


# use default observations
obs = mb.Observations()
agent_handlers = mb.AgentHandlers(observations=obs)
miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers)])
             
world = mb.defaultworld(seed='5')

miss.setWorld(world)
miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
# disable passage of time
miss.serverSection.initial_conditions.time_pass = 'false'
miss.serverSection.initial_conditions.time_start = "1000"
mc = MalmoConnector(miss)
rob = RobustObserver(mc)

mc.safeStart()
```

Default observations include 
<ObservationFromRay/> - object visible at the center of the screen
<ObservationFromFullStats/> - health, position, world time  
<ObservationFromFullInventory/> - all items in inventory  
<ObservationFromRecipes/> - all the recipies, turned off after the game start  
<ObservationFromNearbyEntities/> - position of nearby mobes and floating objects  
<ObservationFromGrid/> - grid of 5x4x5 blocks around the player  




