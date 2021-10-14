

s = """
97, grass
50, dirt
68, fire
66, white_flower/azure_bluet
22, tallgrass
69, yellow_flower/dandelion
67, red_flower/oxeye_daisy
122, log/oak
121, log/birch
115, leaves/birch
120, leaves/oak
35, double_plant/sunflower
14, stone/andesite
98, gravel
102, water
27, sunflower
96, gold_ore
6, stone/stone
117, sand
114, lava"""

segment_mapping = dict()

for line in s.split('\n'):
    if line.strip():
        code, name = line.split(',')
        segment_mapping[int(code.strip())] = name.strip()


