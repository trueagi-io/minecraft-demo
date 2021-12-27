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
         ({'blocks': [{'type': 'sand'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'sand'}
         ),
         ({'blocks': [{'type': 'sandstone'}],
           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
          {'type': 'sandstone'}
         ),
         ({'blocks': [{'type': 'stone'}],
           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
          {'type': 'stone'}
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
         #({'blocks': [{'type': 'deadbush'}],
         #  'tools': [None]},
         # {'type': 'stick'}
         #)
        ]

crafts = [([{'type': 'log', 'quantity': 1}],
           {'type': 'planks', 'quantity': 4}),
          ([{'type': 'log', 'variant': 'spruce', 'quantity': 1}],
           {'type': 'planks', 'variant': 'spruce', 'quantity': 4}),
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
          ([{'type': 'planks', 'variant': 'spruce', 'quantity': 6}],
            {'type': 'spruce_door', 'quantity': 3}),
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
            {'type': 'pumpkin_seeds'}),
          ([{'type': 'sand', 'quantity': 1}],
            {'type': 'glass'})
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

def find_mine_by_block(block):
    for mine in mines:
        for b in mine[0]['blocks']:
            if matchEntity(b, block):
                return mine
    return None

def find_mines_by_result(entity):
    return list(filter(lambda mine: matchEntity(mine[1], entity), mines))

def find_crafts_by_result(entity):
    return list(filter(lambda craft: matchEntity(craft[1], entity), crafts))

def select_minetool(invent, mine_entry):
    if mine_entry is None:
        return None
    inv = [{'type': 'air', 'index': n, 'quantity': 64} for n in range(36)]
    for item in invent:
        inv[item['index']] = item
    result = None
    for tool in mine_entry[0]['tools']:
        for item in inv:
            if tool is None and (result is None or result['quantity'] < item['quantity']):
                result = item
            elif tool == item['type']:
                return item
    return result

def findInInventory(invent, target):
    for item in invent:
        if not matchEntity(item, target):
            continue
        if 'quantity' in target:
            if item['quantity'] < target['quantity']:
                continue
        return item
    return None

def isInInventory(invent, target):
    return findInInventory(invent, target) is not None

def lackCraftItems(invent, craft_entry):
    missing = []
    for item in craft_entry[0]:
        if not isInInventory(invent, item):
            missing += [item]
    return missing

def assoc_blocks(blocks):
    assoc = {'log': ['log2', 'leaves', 'leaves2'],
             'log2': ['log', 'leaves2', 'leaves'],
             'coal_ore': ['stone'],
             'iron_ore': ['stone']}
    blocks2 = []
    for b in blocks:
        if b in assoc:
            blocks2 += assoc[b]
    return blocks2


