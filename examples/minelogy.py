from difflib import SequenceMatcher
import time



class Minelogy():
    def __init__(self, item_list, items_to_craft, mcrecipes, items_to_mine, blockdrops):
        self.log_names = []
        self.planks_names = []
        self.leaves_names = []
        self.leaves_names_t = []
        self.door_names_t = []
        self.trapdoor_names_t = []
        self.log_names_t = []
        self.planks_names_t = []
        self.fuel_priority = {}
        self.mines = []
        # self.mines = [({'blocks': [{'type': 'log'}],
        #            'tools': ['stone_axe', 'wooden_axe', None]},
        #           {'type': 'log'}
        #           ),
        #          ({'blocks': [{'type': 'grass'}, {'type': 'dirt'}],
        #            'tools': ['stone_shovel', 'wooden_shovel', None]},
        #           {'type': 'dirt'}
        #           ),
        #          ({'blocks': [{'type': 'sand'}],
        #            'tools': ['stone_shovel', 'wooden_shovel', None]},
        #           {'type': 'sand'}
        #           ),
        #          ({'blocks': [{'type': 'diorite'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'diorite'}
        #           ),
        #           ({'blocks': [{'type': 'granite'}],
        #             'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #            {'type': 'granite'}
        #            ),
        #          ({'blocks': [{'type': 'clay'}],
        #            'tools': ['stone_shovel', 'wooden_shovel', None]},
        #           {'type': 'clay_ball'}
        #           ),
        #          ({'blocks': [{'type': 'gravel'}],
        #            'tools': [None]},
        #           {'type': 'gravel'}
        #           ),
        #          ({'blocks': [{'type': 'sandstone'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'sandstone'}
        #           ),
        #          ({'blocks': [{'type': 'stone'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'stone'}
        #           ),
        #          ({'blocks': [{'type': 'deepslate'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'deepslate'}
        #           ),
        #          ({'blocks': [{'type': 'cobblestone'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'cobblestone'}
        #           ),
        #          ({'blocks': [{'type': 'coal_ore'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']},
        #           {'type': 'coal'}
        #           ),
        #          ({'blocks': [{'type': 'diamond_ore', 'depthmin': 5}],
        #            'tools': ['iron_pickaxe']},
        #           {'type': 'diamond'}
        #           ),
        #          ({'blocks': [{'type': 'copper_ore'}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe']},
        #           {'type': 'raw_copper'}
        #           ),
        #          ({'blocks': [{'type': 'iron_ore', 'depthmin': 25}],
        #            'tools': ['iron_pickaxe', 'stone_pickaxe']},
        #           {'type': 'raw_iron'}
        #           ),
        #          ({'blocks': [{'type': 'pumpkin'}],
        #            'tools': [None]},
        #           {'type': 'pumpkin'}
        #           ),
        #          ({'blocks': [{'type': 'leaves', 'variant': 'oak'}],
        #            'tools': [None]},
        #           {'type': 'apple'}
        #           ),
        #          ({'blocks': [{'type': 'leaves'}],
        #            'tools': [None]},
        #           {'type': 'sapling'}
        #           ),
        #          ({'blocks': [{'type': 'tallgrass'}],
        #            'tools': [None]},
        #           {'type': 'wheat_seeds'}
        #           ),
        #          ({'blocks': [{'type': 'deadbush'}],
        #            'tools': [None]},
        #           {'type': 'stick'}
        #           ),
        #          ]
        self.crafts = []
        # self.crafts = [([{'type': 'log', 'quantity': 1}],
        #            {'type': 'planks', 'quantity': 4}),
        #           ([{'type': 'planks', 'quantity': 2}],
        #            {'type': 'stick', 'quantity': 4}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 3}],
        #            {'type': 'wooden_axe'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 3}],
        #            {'type': 'wooden_pickaxe'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'planks', 'quantity': 1}],
        #            {'type': 'wooden_shovel'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
        #            {'type': 'stone_axe'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 3}],
        #            {'type': 'stone_pickaxe'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'cobblestone', 'quantity': 1}],
        #            {'type': 'stone_shovel'}),
        #           ([{'type': 'raw_iron', 'quantity': 1}, {'type': 'furnace', 'quantity': 1},
        #             {'type': 'fuel', 'quantity': 1}],
        #            {'type': 'iron_ingot'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
        #            {'type': 'iron_axe'}),
        #           ([{'type': 'cobblestone', 'quantity': 8}],
        #            {'type': 'furnace'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 3}],
        #            {'type': 'iron_pickaxe'}),
        #           ([{'type': 'stick', 'quantity': 2}, {'type': 'iron_ingot', 'quantity': 1}],
        #            {'type': 'iron_shovel'}),
        #           ([{'type': 'stick', 'quantity': 1}, {'type': 'coal', 'quantity': 1}],
        #            {'type': 'torch', 'quantity': 4}),
        #           ([{'type': 'planks', 'quantity': 2}],
        #            {'type': 'pressure_plate'}),
        #           ([{'type': 'planks', 'quantity': 3}],
        #            {'type': 'slab', 'quantity': 6}),
        #           ([{'type': 'planks', 'quantity': 1}],
        #            {'type': 'button'}),
        #           ([{'type': 'planks', 'variant': 'spruce', 'quantity': 6}],
        #            {'type': 'spruce_door', 'quantity': 3}),
        #           ([{'type': 'planks', 'variant': 'birch', 'quantity': 6}],
        #            {'type': 'birch_door', 'quantity': 3}),
        #           ([{'type': 'planks', 'variant': 'oak', 'quantity': 6}],
        #            {'type': 'wooden_door', 'quantity': 3}),
        #           ([{'type': 'planks', 'quantity': 6}],
        #            {'type': 'trapdoor', 'quantity': 2}),
        #           ([{'type': 'cobblestone', 'quantity': 3}],
        #            {'type': 'stone_slab', 'quantity': 6}),
        #           ([{'type': 'cobblestone', 'quantity': 3}],
        #            {'type': 'cobblestone_wall', 'quantity': 6}),
        #           ([{'type': 'stick', 'quantity': 1}, {'type': 'cobblestone', 'quantity': 1}],
        #            {'type': 'lever'}),
        #           ([{'type': 'pumpkin', 'quantity': 1}],
        #            {'type': 'pumpkin_seeds'}),
        #           ([{'type': 'sand', 'quantity': 1}, {'type': 'furnace', 'quantity': 1},
        #             {'type': 'fuel', 'quantity': 1}],
        #            {'type': 'glass'}),
        #           ([{"type": "stick", "quantity": 4}, {"type": "planks", "quantity": 2}],
        #            {"type": "fence_gate", "quantity": 1}),
        #           ([{"type": "planks", "quantity": 6}, {"type": "stick", "quantity": 1}],
        #            {"type": "sign", "quantity": 3}),
        #           ([{"type": "planks", "quantity": 5}],
        #            {"type": "boat", "quantity": 1})
        #           ]
        self.initialize_minelogy(item_list)
        self.set_recipes_for_items(items_to_craft, mcrecipes)
        self.set_mines(items_to_mine, blockdrops)

    def set_mines(self, items_to_mine, blockdrops):
        possible_tool_qualities = ['wooden', 'stone', 'iron', 'golden', 'diamond', 'netherite']
        for blockdrop in blockdrops:
            item_name = blockdrop['item_name']
            if item_name not in items_to_mine and item_name not in self.fuel_priority:
                continue
            block_name = blockdrop['block_name']
            tool = blockdrop['tool']
            tool_list = []
            tool_split = tool.split("_")
            if "AnyTool" in tool:
                tool = tool.replace("AnyTool", "pickaxe")
            if tool == "" or tool == "None":
                tool_list.append(None)
            elif tool_split[0] in possible_tool_qualities:
                tool_prefxs = possible_tool_qualities[possible_tool_qualities.index(tool_split[0]):]
                tool_list = [tool_prefx + "_" + tool_split[1] for tool_prefx in reversed(tool_prefxs)]
            elif "silkt" in tool:
                continue #currently we are not using silktouch enchantment for agent
            elif tool == "shears":
                tool_list = ['shears']
            elif tool == "pickaxe":
                tool_list = [tool_prefx + "_" + tool for tool_prefx in reversed(possible_tool_qualities)]
            self.mines.append(({'blocks': [{'type': block_name}],
                                'tools': tool_list},
                               {'type': item_name}))

    def add_craft_tool(self, craft_tool):
        ingredient = None
        if craft_tool == 'crafting':
            return ingredient
        elif craft_tool == 'smelting':
            ingredient = [{'type': 'furnace', 'quantity': 1},
                    {'type': 'fuel', 'quantity': 1}]
        elif craft_tool == 'stonecutting':
            ingredient = [{'type': 'stonecutter', 'quantity': 1}]
        elif craft_tool == 'blasting':
            ingredient = [{'type': 'blast_furnace', 'quantity': 1}]
        elif craft_tool == 'smoking':
            ingredient = [{'type': 'smoker', 'quantity': 1}, #Probably Smoker can be replaced by furnace?
                          {'type': 'fuel', 'quantity': 1}]
        elif craft_tool == 'campfire_cooking':
            ingredient = [{'type': 'campfire', 'quantity': 1}]
        else:
            print("unknown craft_tool detected")
        return ingredient

    def set_recipes(self, mcrecipes):
        self.crafts = []
        for mcrecipe in mcrecipes:
            craft_name = mcrecipe['name'].split('.')[-1]
            self.__set_one_recipe(mcrecipe, craft_name)

    def set_recipes_for_items(self, items_to_add, mcrecipes, clear_recipes=False, strict_matching=True):  # here we're sending list of items we want to add recipes for
        if clear_recipes:
            self.crafts = []
        for mcrecipe in mcrecipes:
            craft_name = mcrecipe['name'].split('.')[-1]
            if (craft_name in items_to_add) or (not strict_matching and craft_name.split("_")[-1] in items_to_add):
                self.__set_one_recipe(mcrecipe, craft_name)

    def __match_materials(self, materials):
        materials = [material['type'].split(".")[-1] for material in materials]
        name1 = materials[0]
        materials_list=[name1]
        if "log" in name1:
            return "log"
        for i in range(1, len(materials)):
            name2 = materials[i]
            if "log" in name2:
                return "log"
            materials_list.append(name2)
            match = SequenceMatcher(None, name1, name2).find_longest_match()
            if match.size > 1:
                name1 = name2[match.b:match.b + match.size]
        if name1 == "stone":
            materials_list.append(name1) #  temporary stub
        exec("self.{}_types=materials_list".format(name1.replace("_", "")))
        return name1.replace("_", "")

    # this private method is just to avoid code duplication in set_recipes and set_recipes_for_items
    def __set_one_recipe(self, mcrecipe, craft_name):
        if "planks" in craft_name:
            craft_name = "planks"
        craft_quantity = mcrecipe['count']
        craft_tool = mcrecipe['recipe_type']
        craft_group = mcrecipe['group']  # Need to understand how this could be used
        ingredients = {}
        craft_ingredients = []
        for material in mcrecipe['ingredients']:
            if len(material) == 0:
                continue
            if len(material) > 1:
                material_name = self.__match_materials(material)
            else:
                material_name = material[0]['type'].split(".")[-1]
            if material_name not in ingredients:
                ingredients[material_name] = 1
            else:
                ingredients[material_name] += 1
        for ingredient in ingredients:
            craft_ingredients.append({'type': ingredient, 'quantity': ingredients[ingredient]})
        if len(craft_ingredients) == 0:
            return
        additional_tool = self.add_craft_tool(craft_tool)
        if additional_tool is not None:
            craft_ingredients.extend(additional_tool)
        craft_entity = (craft_ingredients,
                        {'type': craft_name, 'quantity': craft_quantity})
        if (craft_entity not in self.crafts) and (len(craft_entity) > 0):
            self.crafts.append(craft_entity)


    def initialize_minelogy(self, item_list):
        for iname in item_list:
            if 'log' in iname:
                self.log_names.append(iname)
                self.log_names_t.append({'type': iname})
                self.fuel_priority[iname] = 2
            elif 'planks' in iname:
                self.planks_names.append(iname)
                self.planks_names_t.append({'type': iname})
                self.fuel_priority[iname] = 1
            elif 'leaves' in iname:
                self.leaves_names.append(iname)
                self.leaves_names_t.append({'type': iname})
            elif '_door' in iname:
                self.door_names_t.append({'type': iname})
            elif 'trapdoor' in iname:
                self.trapdoor_names_t.append({'type': iname})
            elif 'sapling' in iname:
                self.fuel_priority[iname] = 0

        self.fuel_priority['stick'] = 3
        self.fuel_priority['coal'] = 4

    def get_otype(self, obj):
        t = None
        if 'type' in obj:
            t = obj['type']
        elif 'name' in obj:
            t = obj['name']
        return t

    def get_oatargets(self, obj):
        t = obj
        if t == 'log':
            return self.log_names_t + self.leaves_names_t
        if t == 'stone':
            return [{'type': 'dirt'}, {'type': 'grass'}, {'type': 'grass_block'}]
        elif t == 'coal_ore' or t == 'iron_ore':
            return [{'type': 'stone'}]
        return []

    def get_ovariant(self, obj):
        v = None
        if 'variant' in obj:
            v = obj['variant']
        elif 'variation' in obj:
            v = obj['variation']
        return v

    def mimic_target(self, target, variants):
        res = []
        for var in variants:
            var_target = target.copy()
            if isinstance(var, str):
                var_target['type'] = var
            else:
                var_target['type'] = var['type']
            res.append(var_target)
        return res

    def get_target_variants(self, target):
        if target['type'] == "planks":
            res = []
            for p_name in self.planks_names:
                res.append({'type': p_name, 'quantity': target['quantity']})
            return res
        elif target['type'] == "door":
            return self.door_names_t
        elif target['type'] == "trapdoor":
            return self.trapdoor_names_t
        elif target['type'] == "log":
            return self.mimic_target(target, self.log_names_t)
        elif hasattr(self, target['type']+"_types"):
            return self.mimic_target(target, getattr(self, "{}_types".format(target['type'])))
        return target

    def get_log_names(self):
        return self.log_names

    def get_new_type(self, obj):
        splitted_obj = obj['type'].split("_")
        if len(splitted_obj) != 1:
            res = ''
            for splitted in splitted_obj[:-1]:
                res += splitted + "_"
            obj_type = res[:-1]
            return obj_type
        return None

    def get_craft_variants(self, to_craft, target):
        if "planks" in to_craft['type']:
            target_type = self.get_new_type(target)
            res = []
            exact_res = []
            for p_name in self.planks_names:
                temp = {'type': p_name, 'quantity': to_craft['quantity']}
                craft_type = self.get_new_type(temp)
                res.append(temp)
                if craft_type == target_type:
                    exact_res.append(temp)
                    return exact_res
            return res
        elif "door" in to_craft['type']:
            return self.door_names_t
        elif "trapdoor" in to_craft['type']:
            return self.trapdoor_names_t
        elif "log" in to_craft['type']:
            target_type = self.get_new_type(target)
            for log_name in self.log_names_t:
                log_type = self.get_new_type(log_name)
                if target_type == log_type:
                    return log_name
            return self.log_names_t
        return to_craft

    def get_otlist(self, objs):
        return list(map(self.get_otype, objs))

    def __matchEntity(self, source_type, target_type):
        if target_type == 'fuel' and source_type in self.fuel_priority:
            return True
        if (source_type != target_type) and (source_type != target_type.split("_")[-1]) and (target_type != source_type.split("_")[-1]):
            return False
        # target_v = self.get_ovariant(target)
        # if target_v is not None:
        #     if target_v != self.get_ovariant(source) and target_v[0] != '$':
        #         return False
        return True

    def matchEntity(self, source, target):
        if source is None:
            return False
        source_type = self.get_otype(source) if isinstance(source, dict) else source
        target_type = self.get_otype(target) if isinstance(target, dict) else target
        if hasattr(self, source_type+"_types"):
            source_types = getattr(self, "{}_types".format(source_type))
            for s_type in source_types:
                if self.__matchEntity(s_type, target_type):
                    return True
        elif hasattr(self, target_type+"_types"):
            target_types = getattr(self, "{}_types".format(target_type))
            for t_type in target_types:
                if self.__matchEntity(source_type, t_type):
                    return True
        return self.__matchEntity(source_type, target_type)

    def find_mine_by_block(self, block):
        for mine in self.mines:
            for b in mine[0]['blocks']:
                if self.matchEntity(b, block):
                    return mine
        return None

    def find_mines_by_result(self, entity):
        return list(filter(lambda mine: self.matchEntity(mine[1], entity), self.mines))

    def find_crafts_by_result(self, entity):
        return list(filter(lambda craft: self.matchEntity(craft[1], entity), self.crafts))

    def find_fuel(self, invent):
        return list(filter(lambda item: item is not None, list(self.findInInventory(invent, fl) for fl in self.fuel_priority)))

    def select_minetool(self, invent, mine_entry, result=None):
        if mine_entry is None:
            return None
        for tool in reversed(mine_entry[0]['tools']):
            if tool == None and isinstance(result, str):
                return None
            for item in invent:
                if tool is None and (result is None or isinstance(result, str) or result['quantity'] < item['quantity']):
                    result = item
                elif tool == item['type']:
                    return item
        return result

    def findInInventory(self, invent, target):
        for item in invent:
            if not self.matchEntity(item, target):
                continue
            if 'quantity' in target:
                if item['quantity'] < target['quantity']:
                    continue
            return item
        return None

    def isInInventory(self, invent, target):
        return self.findInInventory(invent, target) is not None

    def lackCraftItems(self, invent, craft_entry):
        missing = []
        for item in craft_entry[0]:
            if not self.isInInventory(invent, item):
                missing += [item]
        return missing

    def assoc_blocks(self, blocks):
        assoc = {'log': self.log_names+self.leaves_names,
                 'coal_ore': ['stone'],
                 'iron_ore': ['stone'],
                 'diamond_ore': ['stone', 'deepslate']}
        blocks2 = []
        for b in blocks:
            if b in assoc:
                blocks2 += assoc[b]
        return blocks2

    def checkCraftType(self, to_craft, to_mine):
        craft_is_str = False
        if isinstance(to_craft, str):
            craft_is_str = True
            to_craft_dict = {"type" : to_craft}
        else:
            to_craft_dict = to_craft

        if isinstance(to_mine, str):
            to_mine = {"type" : to_mine}
        if (to_craft_dict['type'] == 'planks' or to_craft_dict['type'] == 'fence_gate' or to_craft_dict['type'] == 'sign'
            or to_craft_dict['type'] == "pressure_plate" or to_craft_dict['type'] == "button" or to_craft_dict['type'] == "boat"
            or to_craft_dict['type'] == "slab"):
            to_mine_type = self.get_otype(to_mine)
            to_craft_type = self.get_otype(to_craft_dict)
            if (to_mine_type in self.leaves_names) or (to_mine_type in self.log_names):
                if self.matchEntity(to_craft_dict, self.planks_names_t[0]):
                    to_craft_type = self.get_new_type(to_mine) + "_" + to_craft_type
                    if ('stripped' in to_mine_type):
                        to_craft_type = to_craft_type.split("stripped_")[1]
                    if craft_is_str:
                        return to_craft_type
                    else:
                        return {"type" : to_craft_type, "quantity": to_craft_dict['quantity']}
        return to_craft

    def addFuel(self, to_craft, invent):
        if (to_craft == 'iron_ingot' or to_craft == 'glass'):
            fuels = self.find_fuel(invent)
            least_priority = len(self.fuel_priority) + 5
            least_prior_fuel = ''
            for fuel in fuels:
                cur_priority = self.fuel_priority[fuel['type']]
                if cur_priority < least_priority:
                    least_priority = cur_priority
                    least_prior_fuel = fuel['type']
            return f'{to_craft} {least_prior_fuel}'
        return to_craft
