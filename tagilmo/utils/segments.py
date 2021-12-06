

s = """
162,232,70 leaves/oak
162,0,93 log/oak
185,0,93 log/oak1
0,209,70 leaves/birch
70,0,93 log/birch
93,0,93 log/birch1
209,0,93 log/spruce
232,0,93 log/spruce1
185,232,70 leaves/spruce

232,185,70 leaves2/dark_oak
23,0,93 log2/dark_oak
46,0,93 log2/dark_oak1
209,232,23 red_mushroom_block
0,116,23 brown_mushroom_block
185,46,46 mushroom/stem

0,162,23 coal_ore

46,46,0 stone/stone
185,116,23 stone/granite
232,232,0 stone/diorite
93,209,23 stone/cobblestone
23,116,0 stone/andesite

93,162,23 dirt
"""

segment_mapping = dict()

for line in s.split('\n'):
    if line.strip():
        code, name = line.split(' ')
        code = [int(x) for x in code.split(',')]
        segment_mapping[tuple(code)] = name.strip()
