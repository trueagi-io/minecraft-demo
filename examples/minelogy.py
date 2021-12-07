mines = [({'blocks': [{'type': 'log'}],
           'tools': ['stone_axe', 'wooden_axe', None]},
          {'type': 'log'}
         ),
         ({'blocks': [{'type': 'log2'}],
           'tools': ['stone_axe', 'wooden_axe', None]},
          {'type': 'log2'}
         ),
         ({'blocks': [{'type': 'grass'}, {'type': 'dirt'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'dirt'}
         ),
         ({'blocks': [{'type': 'stone', 'variant': 'stone'}],
           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
          {'type': 'cobblestone'}
         ),
         ({'blocks': [{'type': 'coal_ore'}],
           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
          {'type': 'coal'}
         ),
         ({'blocks': [{'type': 'iron_ore'}],
           'tools': ['stone_pickaxe']},
          {'type': 'iron_ore'}
         ),
         ({'blocks': [{'type': 'pumpkin'}],
           'tools': [None]},
          {'type': 'pumpkin'}
         ),
         ({'blocks': [{'type': 'leaves', 'variant': 'oak'}],
           'tools': [None]},
          {'type': 'apple'}
         ),
         ({'blocks': [{'type': 'leaves'}],
           'tools': [None]},
          {'type': 'sapling'}
         ),
         ({'blocks': [{'type': 'tallgrass'}],
           'tools': [None]},
          {'type': 'wheat_seeds'}
         )
        ]

crafts = [([{'type': 'log', 'quantity': 1}],
           {'type': 'planks', 'quantity': 4}),
          ([{'type': 'log2', 'quantity': 1}],
           {'type': 'planks', 'quantity': 4}),
          ([{'type': 'planks', 'quantity': 2}],
            {'type': 'stick', 'quantity': 4}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 3}],
           {'type': 'wooden_axe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 3}],
           {'type': 'wooden_pickaxe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 1}],
           {'type': 'wooden_shovel'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
           {'type': 'stone_axe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
           {'type': 'stone_pickaxe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 1}],
           {'type': 'stone_shovel'}),
          # we don't actually need coal with simplified furnance
          ([{'type': 'iron_ore', 'quantity': 1}], #{'type': 'coal', 'quantity': 1}],
           {'type': 'iron_ingot'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
           {'type': 'iron_axe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
           {'type': 'iron_pickaxe'}),
          ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 1}],
           {'type': 'iron_shovel'}),
          ([{'type': 'stick', 'quantity': 1}, {'type': 'coal', 'quantity': 1}],
           {'type': 'torch', 'quantity': 4}),
          ([{'type': 'planks', 'quantity': 2}],
            {'type': 'wooden_pressure_plate'}),
          ([{'type': 'planks', 'quantity': 3}],
            {'type': 'wooden_slab', 'quantity': 6}),
          ([{'type': 'planks', 'quantity': 1}],
            {'type': 'wooden_button'}),
          ([{'type': 'planks', 'quantity': 6}],
            {'type': 'wooden_door', 'quantity': 3}),
          ([{'type': 'planks', 'quantity': 6}],
            {'type': 'trapdoor', 'quantity': 2}),
          ([{'type': 'cobblestone', 'quantity': 3}],
            {'type': 'stone_slab', 'quantity': 6}),
          ([{'type': 'cobblestone', 'quantity': 3}],
            {'type': 'cobblestone_wall', 'quantity': 6}),
          ([{'type': 'stick', 'quantity': 1}, {'type': 'cobblestone', 'quantity': 1}],
           {'type': 'lever'}),
          ([{'type': 'pumpkin', 'quantity': 1}],
            {'type': 'pumpkin_seeds'})
         ]

def get_otype(obj):
    t = None
    if 'type' in obj:
        t = obj['type']
    elif 'name' in obj:
        t = obj['name']
    return t

def get_ovariant(obj):
    v = None
    if 'variant' in obj:
        v = obj['variant']
    elif 'variation' in obj:
        v = obj['variation']
    return v

def get_otlist(objs):
    return list(map(get_otype, objs))

def matchEntity(source, target):
    if source is None:
        return False
    # if target is None: return True
    if get_otype(source) != get_otype(target):
        return False
    target_v = get_ovariant(target)
    if target_v is not None:
        if target_v != get_ovariant(source) and target_v[0] != '$':
            return False
    return True
