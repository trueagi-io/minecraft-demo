import minecraft_data
log_names = []
planks_names = []
leaves_names = []
leaves_names_t = []
door_names_t = []
trapdoor_names_t = []
log_names_t = []
planks_names_t = []
drs = []
# versions = minecraft_data.common().protocolVersions
mcd = minecraft_data('1.18.1')  # here we must put current minecraft version
count = 0
for item in mcd.items_list:
    iname = item['name']
    if 'log' in iname:
        log_names.append(iname)
        log_names_t.append({'type': iname})
    if 'planks' in iname:
        planks_names.append(iname)
        planks_names_t.append({'type': iname})
    if 'leaves' in iname:
        leaves_names.append(iname)
        leaves_names_t.append({'type': iname})
    if '_door' in iname:
        door_names_t.append({'type': iname})
    if 'trapdoor' in iname:
        trapdoor_names_t.append({'type': iname})

def make_log_plank_triple(log_names, plank_names):
    res = []
    for log_name in log_names:
        log_variant = log_name.split("log")[0]
        if log_variant == '':
            return None
        for plank_name in plank_names:
            plank_variant = plank_name.split("plank")[0]
            if plank_variant == log_variant:
                res.append((log_name, plank_name, log_variant[:-1]))
    return res

def make_door_plank_quadriple(door_names, plank_names):
    res = []
    for door_name in door_names:
        door_variant = door_name.split("door")[0]
        if door_variant == '':
            return None
        for plank_name in plank_names:
            plank_variant = plank_name.split("plank")[0]
            if plank_variant == door_variant:
                res.append((door_name, door_variant+"trapdoor", plank_name, door_variant[:-1]))
    return res

# log_plank_triple_list = make_log_plank_triple(log_names, planks_names)
# door_plank_quadriple_list = make_door_plank_quadriple(door_names, planks_names)
# def make_mines_list(log_names):
#     mines = []
#     new_version = False
#     for log_name in log_names:
#         variant = None
#         if "_" in log_name:
#             variant = log_name.split("_")[0]
#             new_version = True
#         if variant is not None:
#             mines.append(({'blocks': [{'type': log_name, 'variant': variant}],
#                            'tools': ['stone_axe', 'wooden_axe', None]},
#                           {'type': log_name, 'variant': variant}
#                           ))
#         mines.append(({'blocks': [{'type': log_name}],
#                        'tools': ['stone_axe', 'wooden_axe', None]},
#                       {'type': log_name}
#                       ))
#     if new_version == False:
#         mines.extend([
#             ({'blocks': [{'type': 'log', 'variant': 'oak'}],
#               'tools': ['stone_axe', 'wooden_axe', None]},
#              {'type': 'log', 'variant': 'oak'}
#              ),
#             ({'blocks': [{'type': 'log', 'variant': 'birch'}],
#               'tools': ['stone_axe', 'wooden_axe', None]},
#              {'type': 'log', 'variant': 'birch'}
#              ),
#             ({'blocks': [{'type': 'log', 'variant': 'spruce'}],
#               'tools': ['stone_axe', 'wooden_axe', None]},
#              {'type': 'log', 'variant': 'spruce'}
#              )])
#     mines.extend([
#         ({'blocks': [{'type': 'grass'}, {'type': 'dirt'}],
#           'tools': ['stone_shovel', 'wooden_shovel', None]},
#          {'type': 'dirt'}
#          ),
#         ({'blocks': [{'type': 'sand'}],
#           'tools': ['stone_shovel', 'wooden_shovel', None]},
#          {'type': 'sand'}
#          ),
#         ({'blocks': [{'type': 'sand'}],
#           'tools': ['stone_shovel', 'wooden_shovel', None]},
#          {'type': 'sand'}
#          ),
#         ({'blocks': [{'type': 'clay'}],
#           'tools': ['stone_shovel', 'wooden_shovel', None]},
#          {'type': 'clay_ball'}
#          ),
#         ({'blocks': [{'type': 'gravel'}],
#           'tools': [None]},
#          {'type': 'gravel'}
#          ),
#         ({'blocks': [{'type': 'sandstone'}],
#           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
#          {'type': 'sandstone'}
#          ),
#         ({'blocks': [{'type': 'stone'}],
#           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
#          {'type': 'stone'}
#          ),
#         ({'blocks': [{'type': 'stone', 'variant': 'stone'}],
#           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
#          {'type': 'cobblestone'}
#          ),
#         ({'blocks': [{'type': 'coal_ore'}],
#           'tools': ['stone_pickaxe', 'wooden_pickaxe']},
#          {'type': 'coal'}
#          ),
#         ({'blocks': [{'type': 'iron_ore'}],
#           'tools': ['stone_pickaxe']},
#          {'type': 'iron_ore'}
#          ),
#         ({'blocks': [{'type': 'pumpkin'}],
#           'tools': [None]},
#          {'type': 'pumpkin'}
#          ),
#         ({'blocks': [{'type': 'leaves', 'variant': 'oak'}],
#           'tools': [None]},
#          {'type': 'apple'}
#          ),
#         ({'blocks': [{'type': 'leaves'}],
#           'tools': [None]},
#          {'type': 'sapling'}
#          ),
#         ({'blocks': [{'type': 'tallgrass'}],
#           'tools': [None]},
#          {'type': 'wheat_seeds'}
#          )])
#     return mines
#
# def make_crafts_list(log_plank_triple_list, door_plank_quadriple_list):
#     crafts = []
#     for lp_triple in log_plank_triple_list:
#         log_n = lp_triple[0]
#         plank_n = lp_triple[1]
#         variant_n = lp_triple[2]
#         crafts.extend([([{'type': log_n, 'quantity': 1}],
#                         {'type': plank_n, 'quantity': 4}),
#                        ([{'type': log_n, 'variant': variant_n, 'quantity': 1}],
#                         {'type': plank_n, 'variant': variant_n, 'quantity': 4}),
#                        ([{'type': plank_n, 'quantity': 2}],
#                         {'type': 'stick', 'quantity': 4}),
#                        ([{'type': 'stick', 'quantity': 2}, {'type': plank_n, 'quantity': 3}],
#                         {'type': 'wooden_axe'}),
#                        ([{'type': 'stick', 'quantity': 2}, {'type': plank_n, 'quantity': 3}],
#                         {'type': 'wooden_pickaxe'}),
#                        ([{'type': 'stick', 'quantity': 2}, {'type': plank_n, 'quantity': 1}],
#                         {'type': 'wooden_shovel'}),
#                        ([{'type': plank_n, 'quantity': 2}],
#                         {'type': 'wooden_pressure_plate'}),
#                        ([{'type': plank_n, 'quantity': 3}],
#                         {'type': 'wooden_slab', 'quantity': 6}),
#                        ([{'type': plank_n, 'quantity': 1}],
#                         {'type': 'wooden_button'})
#                        ])
#     for dp_quadriple in door_plank_quadriple_list:
#         door_n = dp_quadriple[0]
#         trapdoor_n = dp_quadriple[1]
#         plank_n = dp_quadriple[2]
#         variant_n = dp_quadriple[3]
#         crafts.extend([
#             ([{'type': plank_n, 'variant': variant_n, 'quantity': 6}],
#              {'type': door_n, 'quantity': 3}),
#             ([{'type': plank_n, 'variant': variant_n, 'quantity': 6}],
#              {'type': trapdoor_n, 'quantity': 2})
#         ])
#     crafts.extend([
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
#          {'type': 'stone_axe'}),
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
#          {'type': 'stone_pickaxe'}),
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 1}],
#          {'type': 'stone_shovel'}),
#         # we don't actually need coal with simplified furnance
#         ([{'type': 'iron_ore', 'quantity': 1}], #{'type': 'coal', 'quantity': 1}],
#          {'type': 'iron_ingot'}),
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
#          {'type': 'iron_axe'}),
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
#          {'type': 'iron_pickaxe'}),
#         ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 1}],
#          {'type': 'iron_shovel'}),
#         ([{'type': 'stick', 'quantity': 1}, {'type': 'coal', 'quantity': 1}],
#          {'type': 'torch', 'quantity': 4}),
#         ([{'type': 'cobblestone', 'quantity': 3}],
#          {'type': 'stone_slab', 'quantity': 6}),
#         ([{'type': 'cobblestone', 'quantity': 3}],
#          {'type': 'cobblestone_wall', 'quantity': 6}),
#         ([{'type': 'stick', 'quantity': 1}, {'type': 'cobblestone', 'quantity': 1}],
#          {'type': 'lever'}),
#         ([{'type': 'pumpkin', 'quantity': 1}],
#          {'type': 'pumpkin_seeds'}),
#         ([{'type': 'sand', 'quantity': 1}],
#          {'type': 'glass'})
#     ])
#     return crafts
#
# crafts = make_crafts_list(log_plank_triple_list, door_plank_quadriple_list)
# mines = make_mines_list(log_names)

mines = [({'blocks': [{'type': 'log'}],
           'tools': ['stone_axe', 'wooden_axe', None]},
          {'type': 'log'}
         ),
         # ({'blocks': [{'type': 'log', 'variant': 'oak'}],
         #   'tools': ['stone_axe', 'wooden_axe', None]},
         #  {'type': 'log', 'variant': 'oak'}
         # ),
         # ({'blocks': [{'type': 'log', 'variant': 'birch'}],
         #   'tools': ['stone_axe', 'wooden_axe', None]},
         #  {'type': 'log', 'variant': 'birch'}
         # ),
         # ({'blocks': [{'type': 'log', 'variant': 'spruce'}],
         #   'tools': ['stone_axe', 'wooden_axe', None]},
         #  {'type': 'log', 'variant': 'spruce'}
         # ),
         # ({'blocks': [{'type': 'log2'}],
         #   'tools': ['stone_axe', 'wooden_axe', None]},
         #  {'type': 'log2'}
         # ),
         ({'blocks': [{'type': 'grass'}, {'type': 'dirt'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'dirt'}
         ),
         ({'blocks': [{'type': 'sand'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'sand'}
         ),
         ({'blocks': [{'type': 'sand'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'sand'}
         ),
         ({'blocks': [{'type': 'clay'}],
           'tools': ['stone_shovel', 'wooden_shovel', None]},
          {'type': 'clay_ball'}
         ),
         ({'blocks': [{'type': 'gravel'}],
           'tools': [None]},
          {'type': 'gravel'}
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
         #({'blocks': [{'type': 'cobblestone'}],
         #  'tools': ['stone_pickaxe', 'wooden_pickaxe']},
         # {'type': 'cobblestone'}
         #),
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
          # ([{'type': 'log', 'variant': 'oak', 'quantity': 1}],
          #  {'type': 'planks', 'variant': 'oak', 'quantity': 4}),
          # ([{'type': 'log', 'variant': 'spruce', 'quantity': 1}],
          #  {'type': 'planks', 'variant': 'spruce', 'quantity': 4}),
          # ([{'type': 'log', 'variant': 'birch', 'quantity': 1}],
          #  {'type': 'planks', 'variant': 'birch', 'quantity': 4}),
          # ([{'type': 'log2', 'quantity': 1}],
          #  {'type': 'planks', 'quantity': 4}),
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
          ([{'type': 'planks', 'variant': 'birch', 'quantity': 6}],
            {'type': 'birch_door', 'quantity': 3}),
          ([{'type': 'planks', 'variant': 'oak', 'quantity': 6}],
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

def get_oatargets(obj):
    t = get_otype(obj)
    if t == 'log':
        return log_names_t + leaves_names_t
    if t == 'stone':
        return [{'type': 'dirt'}, {'type': 'grass'}]
    elif t == 'coal_ore' or t == 'iron_ore':
        return [{'type': 'stone'}]
    return []

def get_ovariant(obj):
    v = None
    if 'variant' in obj:
        v = obj['variant']
    elif 'variation' in obj:
        v = obj['variation']
    return v

def get_target_variants(target):
    if target['type'] == "planks":
        res = []
        for p_name in planks_names:
            res.append({'type': p_name, 'quantity': target['quantity']})
        return res
    elif target['type'] == "door":
        return door_names_t
    elif target['type'] == "trapdoor":
        return trapdoor_names_t
    elif target['type'] == "log":
        return log_names_t
    return target

def get_new_type(obj):
    splitted_obj = obj['type'].split("_")
    if len(splitted_obj) != 1:
        res = ''
        for splitted in splitted_obj[:-1]:
            res += splitted + "_"
        obj_type = res[:-1]
        return obj_type
    return None

def get_craft_variants(to_craft, target):
    if to_craft['type'] == "planks":
        target_type = get_new_type(target)
        res = []
        exact_res = []
        for p_name in planks_names:
            temp = {'type': p_name, 'quantity': to_craft['quantity']}
            craft_type = get_new_type(temp)
            res.append(temp)
            if craft_type == target_type:
                exact_res.append(temp)
                return exact_res
        return res
    elif to_craft['type'] == "door":
        return door_names_t
    elif to_craft['type'] == "trapdoor":
        return trapdoor_names_t
    elif to_craft['type'] == "log":
        target_type = get_new_type(target)
        for log_name in log_names_t:
            log_type = get_new_type(log_name)
            if target_type == log_type:
                return log_name
        return log_names_t
    return to_craft

def get_otlist(objs):
    return list(map(get_otype, objs))

def matchEntity(source, target):
    if source is None:
        return False
    # if target is None: return True
    source_type = get_otype(source)
    target_type = get_otype(target)
    if (source_type != target_type) and (source_type != target_type.split("_")[-1]) and (target_type != source_type.split("_")[-1]):
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
    assoc = {'log': log_names.extend(leaves_names),
             # 'log2': ['log', 'leaves2', 'leaves'],
             'coal_ore': ['stone'],
             'iron_ore': ['stone']}
    blocks2 = []
    for b in blocks:
        if b in assoc:
            blocks2 += assoc[b]
    return blocks2

def checkCraftType(to_craft, to_mine):
    to_mine_type = get_otype(to_mine)
    to_craft_type = get_otype(to_craft)
    if (to_mine_type in leaves_names) or (to_mine_type in log_names):
        if matchEntity(to_craft, planks_names_t[0]):
            to_craft_type = get_new_type(to_mine) + "_" + to_craft_type
            if ('stripped' in to_mine_type):
                to_craft_type = to_craft_type.split("stripped_")[1]
            return {"type" : to_craft_type, "quantity": to_craft['quantity']}
    return None