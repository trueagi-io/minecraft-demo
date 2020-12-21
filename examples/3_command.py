import math
from time import sleep
from tagilmo.utils.malmo_wrapper import MalmoConnector
import tagilmo.utils.mission_builder as mb


class TargetRunner:
    
    def choose_target(self):
        mc.observeProc()
        crossObj = mc.getLineOfSight('type')
        xs = mc.getAgentPos()
        if crossObj is None:
            print("Warning: no object selected. Just running a little")
            self.target = [xs[0] - 10 * math.sin(math.pi*xs[4]/180.), xs[1], xs[2] + 10 * math.cos(math.pi*xs[4]/180.)]
        else:
            self.target = [mc.getLineOfSight('x'), mc.getLineOfSight('y'), mc.getLineOfSight('z')]

    def run_to_target(self):
        if self.target is None: return
        mc.observeProc()
        xs = mc.getAgentPos()
        while abs(xs[0]-self.target[0]) + abs(xs[2]-self.target[2]) > 3:
            sleep(0.02)
            mc.observeProc()
            if mc.getAgentPos() is None: continue
            xs = mc.getAgentPos()
            print([self.target[i] - xs[i] for i in range(3)])
            direction = -math.atan2(self.target[0]-xs[0], self.target[2]-xs[2])*180./math.pi
            diff = xs[4] - direction
            while diff < -180: diff += 360
            while diff > 180: diff -= 360
            if abs(diff) > 10:
                if diff < 0:
                    mc.sendCommand("turn 0.2")
                else:
                    mc.sendCommand("turn -0.2")
            else:
                mc.sendCommand("turn 0")
            mc.sendCommand("move 1")
        sleep(0.1)
        print("Done")
        mc.sendCommand("move 0")
        mc.sendCommand("turn 0")

miss = mb.MissionXML()
miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"))
miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
mc = MalmoConnector(miss)
mc.safeStart()

tr = TargetRunner()

# run the script in the interactive mode, control the agent manually and send commands via Python shell:
# Execute tr.choose_target() to select the target first. Then, manually go to a different place. Release the control to AI.
# Execute tr.run_to_target().
