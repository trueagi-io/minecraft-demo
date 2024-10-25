from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb

block = mb.DrawBlock("-1", "-60", "4", "cobblestone")
it = mb.DrawItem(0, -60, 0, "diamond")
cuboid = mb.DrawCuboid("1", -60, "0", 3, -60, 3, "sandstone")
line = mb.DrawLine(5, -60, 3, 9, -55, 8, "diamond_ore")
draw = mb.DrawingDecorator([block, it, cuboid, line])

miss = mb.MissionXML(serverSection=mb.ServerSection(handlers=mb.ServerHandlers(drawingdecorator=draw)))
miss.setWorld(mb.flatworld("",
                           seed= '5',
              forceReset = "true"))
miss.setObservations(mb.Observations())
# we typically don't need this for real Minecraft agents, but this example just checks the connection to Minecraft
miss.setTimeLimit(5000)

mc = MCConnector(miss)
# Note that we need two instances of Minecraft running
mc.safeStart()

mc.stop()