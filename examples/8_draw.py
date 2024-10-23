from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb

draw_str = """<DrawCuboid x1="0" y1="-60" z1="10" x2="2" y2="-59" z2="13" type="stone"/>
                <DrawBlock x="8" y="-60" z="10" type="cobblestone"/>
                <DrawItem x="10" y="-60" z="10" type="diamond"/>
                <DrawLine x1="10" y1="-60" z1="4" x2="14" y2="-60" z2="11" type="sandstone"/>"""
                #TODO: implement drawSphere  <DrawSphere x="4" y="-55" z="12" radius="3" type="lapis_block"/>
miss = mb.MissionXML(serverSection=mb.ServerSection(handlers=mb.ServerHandlers(drawingdecorator_xml=draw_str)))
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