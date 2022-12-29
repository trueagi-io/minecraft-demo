from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb


miss = mb.MissionXML()
# just an empty flat world for testing
## FIXME: currently using flatworld could lead to an error when launching, so for now we're using defaultworld here
# miss.setWorld(mb.flatworld(""))
world = mb.defaultworld(
        seed='5',
        forceReset="false",
        forceReuse="true")
miss.setWorld(world)

# addAgent currently breaks connection
# miss.addAgent()
# no observations in this example
miss.setObservations(mb.Observations(bAll=False))
# we typically don't need this for real Minecraft agents, but this example just checks the connection to Minecraft
miss.setTimeLimit(5000)

mc = MalmoConnector(miss)
# Note that we need two instances of Minecraft running
mc.safeStart()
