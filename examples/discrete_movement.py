from tagilmo.utils.vereya_wrapper import MCConnector
import tagilmo.utils.mission_builder as mb
from tagilmo.VereyaPython.load_from_xml import load_mission
from time import sleep

block = mb.DrawBlock("-1", "-60", "4", "cobblestone")
it = mb.DrawItem(0, -60, 0, "diamond")
cuboid = mb.DrawCuboid("1", -60, "0", 3, -60, 3, "sandstone")
line = mb.DrawLine(5, -60, 3, 9, -55, 8, "diamond_ore")
draw = mb.DrawingDecorator([block, it, cuboid, line])
path = "D:/Downloads/Telegram Desktop/cliff_walking_1.xml"
miss = load_mission(path)
mc = MCConnector(miss)
mc.safeStart()