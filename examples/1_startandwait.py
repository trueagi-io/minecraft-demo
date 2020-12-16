from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb


miss = mb.MissionXML()
# just an empty flat world for testing
miss.setWorld(mb.flatworld(""))
miss.addAgent()
# no observations in this example
miss.setObservations(mb.Observations(bAll=False))
# we typically don't need this for real Minecraft agents, but this example just checks the connection to Minecraft
miss.setTimeLimit(5000)

mc = MalmoConnector(miss)
# Note that we need two instances of Minecraft running
mc.safeStart()
