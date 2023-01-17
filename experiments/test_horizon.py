import cv2
import common
import math
from common import *
import numpy
import time
from examples.neural import get_image
from behaviours import TurnTo, PITCH, YAW
import tagilmo.utils.mission_builder as mb
from tagilmo.utils.mission_builder import AgentStart
from tagilmo.utils.malmo_wrapper import MalmoConnector, RobustObserverWithCallbacks
from tagilmo.utils import segment_mapping
from tagilmo.utils.mathutils import degree2rad
import tagilmo.utils.mission_builder as mb
from tagilmo.VereyaPython import setupLogger


SCALE = 2 
WIDTH = 320 * SCALE
HEIGHT = 240 * SCALE


def observe_by_line(rob):
    visible = rob.getCachedObserve('getLineOfSights')
    if visible is not None:
        result = [visible,
              visible['x'],
              visible['y'],
              visible['z']]
    return result


def runSkill(rob, b):
    status = 'running' 
    while status == 'running':
        rob.updateAllObservations()
        status, actions = b()         
        for act in actions:
            rob.sendCommand(act)
        time.sleep(0.2)


def collect_data(rob):
    # up is negative, down positive
    pitch_t = [15, -5, 5]
    # right is positive, left is negative
    yaw_t = [-30, 0, 30] 
    rob.updateAllObservations()
    pos = rob.waitNotNoneObserve('getAgentPos')
    current_pitch = pos[PITCH]
    current_yaw = pos[YAW]
    data = []
    for p in pitch_t:
        for y in yaw_t:
            b = TurnTo(rob, current_pitch + p, current_yaw + y) 
            runSkill(rob, b)
            # turned to desired direction, collect point
            point = observe_by_line(rob)
            # collect frame
            frame = rob.getCachedObserve('getImageFrame')
            data.append((point, frame))
            print(point)
            print(numpy.asarray(frame.modelViewMatrix))

    b = TurnTo(rob, current_pitch, current_yaw) 
    runSkill(rob, b)
    point = observe_by_line(rob)
    data.append((point, frame))
    return data


def start_mission():
    miss = mb.MissionXML()
    colourmap_producer = mb.ColourMapProducer(width=WIDTH, height=HEIGHT)
    video_producer = mb.VideoProducer(width=WIDTH, height=HEIGHT, want_depth=False)
    colourmap_producer = None

    obs = mb.Observations()
    agent_handlers = mb.AgentHandlers(observations=obs)

    agent_handlers = mb.AgentHandlers(observations=obs,
        colourmap_producer=colourmap_producer,
        video_producer=video_producer)

    miss = mb.MissionXML(agentSections=[mb.AgentSection(name='Cristina',
             agenthandlers=agent_handlers)])
             #agenthandlers=agent_handlers, agentstart=AgentStart([0, 31, 0, 0]))])

    # good point seed='2', x=-90, y=71, z=375
    # good point seed='2', x=6, y=71, z=350
    world = mb.defaultworld(
        seed='28',
        forceReset="true")

    
    world1 = mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake",
            seed='42',
            forceReset="true")

    # good seed with mountains 31
    miss.setWorld(world)
    miss.serverSection.initial_conditions.allowedmobs = "Pig Sheep Cow Chicken Ozelot Rabbit Villager"
    # uncomment to disable passage of time:
    miss.serverSection.initial_conditions.time_pass = 'false'
    miss.serverSection.initial_conditions.time_start = "1000"

    mc = MalmoConnector(miss)
    obs1 = RobustObserverWithCallbacks(mc)
    return mc, obs1


def to_opengl(vec):
    """
    x and z axis are flipped
    """
    res = vec.copy()
    res[0] *= -1
    res[2] *= -1
    return res


def vec2screen(pt, pitch, yaw, perspective):
    """
    project vector from camera coordinates to pixel coordinates

    pt: list[Int]
      3d point in camera reference frame with malmo coordinates
    pitch: float
      camera rotation around x axis(radians)
    yaw: float
      camera rotation around y axis(radians)
    perspective: numpy.array
      opengl perspective projection matrix 4x4
    """
    pt_c = numpy.asarray(pt) 
    # change axis from mincraft to right-hand side rule expected by the rotation matrix
    # z -> x, x -> y, y -> z
    pt_c = pt_c[[2, 0, 1]] 
    # apply camera rotation
    R = rotation_matrix(0, pitch, yaw)
    pt_c_r = R @ pt_c
    # to minecraft
    # z -> y, x -> z, y -> x 
    pt_c_r = pt_c_r[[1, 2, 0]]
    # apply perspective matrix
    # derivation https://www.songho.ca/opengl/gl_transform.html
    pt_c_gl = numpy.append(to_opengl(pt_c_r), [1])
    pt_clip = perspective @ pt_c_gl
    y = pt_clip[1] / pt_clip[-1]
    x = pt_clip[0] / pt_clip[-1]
    z = pt_clip[2] / pt_clip[-1]
    h_half = HEIGHT // 2
    w_half = WIDTH // 2
    y_w = h_half * y + (-1 + h_half)
    x_w = w_half * x + (-1 + w_half)
    # don't need z right now
    # f and n are near and far plane of perspective frustum
    # they can be computed from perspective matrix - see derivation
    # z_w = (f - n) / 2 * z + (f + n) / 2

    point = [WIDTH - round(x_w), HEIGHT - round(y_w)]
    return point


def show_horizon(rob, height=64, horizon_dist=200):
    """
    draw horizon line on images from minecraft

    Parameters
    ----------
    rob: RobustObserver
    height: int 
      ground level in mincraft coordinates,
      64 is good default value for most default worlds
    horizon_dist: int
      aprox. render distance in blocks
      this increase this parameter together with render distance
    """
    rob.updateAllObservations()
    time.sleep(0.15)
    frame = rob.getCachedObserve('getImageFrame')
    image = get_image(frame, 1, SCALE) 
    # transpose since opengl uses column,row and numpy uses row,column matrix representation
    perspective_matrix = common.perspective_gl[85].T
    camera_coords = [frame.xPos, frame.yPos + 1.62, frame.zPos]
    pitch, yaw = frame.pitch, frame.yaw
    pitch, yaw = degree2rad(pitch), degree2rad(yaw)
    # point on horizon line
    pt4 = numpy.zeros(3, dtype=numpy.float32)
    # set z to a point at render distance
    # minecraft renders only a small distance by default
    pt4[2] = horizon_dist
    # set y to horizon level
    pt4[1] = height - camera_coords[1]
    pt4 = vec2screen(pt4, -pitch, 0, perspective_matrix)
    image = get_image(frame, 1, SCALE) 
    # 0th row is top of the image, so going in negative direction
    # corresponds to going up
    if pt4[1] < 0: # all points below the horizon
        print('below')
        pt4[1] = 0
    if HEIGHT <= pt4[1]: # all points above the horizon
        print('above')
        pt4[1] = HEIGHT - 1
    image = cv2.line(image, (0, pt4[1]), (WIDTH, pt4[1]), (0, 255, 0), 2)
    cv2.imshow('img', image)
    cv2.waitKey(300) 


def main():
    mc, rob = start_mission()
    mc.safeStart()

    setupLogger()
    while True:
        show_horizon(rob, 64)

if __name__ == '__main__':
    main()
