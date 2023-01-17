import minecraft_data
# to install minecraft_data package:
# pip install minecraft_data

# Java edition minecraft-data
mcd = minecraft_data("1.18.2")


class MCDataWrapper:

    def __init__(self, mcd):
        self.mcd = mcd
        self.what_to_filter = {'result': 0, 'ingredients': 1}

    def _idToName(self, id_num):
        return None if id_num is None else mcd.find_item_or_block(id_num)['name']

    def _getBlockIdByName(self, name):
        return mcd.find_item_or_block(name)['id']

    def _processMetedata(self, id_num):
        return None

    def _processGridItems(self, grid_item):
        if isinstance(grid_item, dict):
            return grid_item['id']
        elif grid_item is None:
            return -1
        else:
            return grid_item

    def _inclusionCheck(self, item, recipe):
        if 'inShape' in recipe:
            grid = recipe['inShape']
            first_or_default = next((j for row in grid for j in row if self._processGridItems(j) == item), None)
            return False if first_or_default is None else True
        else:
            grid = recipe['ingredients']
            first_or_default = next((j for j in grid if self._processGridItems(j) == item), None)
            return False if first_or_default is None else True

    def _processIngredient(self, item):
        if 'metadata' in item:
            return self._idToName(item['id'])+'_'
        else:
            return self._idToName(item['id'])

    def _processRow(self, row):
        result = [self._processIngredient(grid_item) if isinstance(grid_item, dict) else self._idToName(grid_item) for grid_item in row]
        return result

    def _recipe_processing(self, recipe):
        output_recipe = {}
        if 'inShape' in recipe:
            grid = recipe['inShape']
            new_grid = [self._processRow(row) for row in grid]
            output_recipe['inShape'] = new_grid
        else:
            grid = recipe['ingredients']
            new_grid = [self._processIngredient(j) if isinstance(j, dict) else self._idToName(j) for j in grid]
            output_recipe['ingredients'] = new_grid
        result = recipe['result'].copy()
        result['name'] = self._idToName(result['id'])
        output_recipe['result'] = result
        return output_recipe

    def _convertIdsToNamesForRecipeList(self, recipe_list):
        return [self._recipe_processing(recipe) for recipe in recipe_list]

    def _applyFilter(self, item, metadata):
        output = False
        if isinstance(item, dict):
            if 'variant' in item:
                if item['variant'] == metadata:
                    output = True
        return output

    # TODO more efficient filtering
    def _metadataIngregientFilter(self, recipe, metadata):
        if 'inShape' in recipe:
            grid = recipe['inShape']
            first_or_default = next((j for row in grid for j in row if self._applyFilter(j, metadata)), None)
            return False if first_or_default is None else True
        else:
            grid = recipe['ingredients']
            first_or_default = next((j for j in grid if self._applyFilter(j, metadata)), None)
            return False if first_or_default is None else True

    def _metadataResultFilter(self, recipe, metadata):
        output = False
        if 'metadata' in recipe['result']:
            if recipe['result']['metadata'] == metadata:
                output = True
        return output

    def _metadataListFiltering(self, recipe_list, metadata, result_or_ingredients=1):
        if result_or_ingredients == 1:
            return [recipe for recipe in recipe_list if not(self._metadataIngregientFilter(recipe, metadata))]
        else:
            return [recipe for recipe in recipe_list if not(self._metadataResultFilter(recipe, metadata))]

    def getItemOrBlockRecipeInclusions(self, name, get_names_instead_of_ids=False, metadata_filter=None, what_to_filter=1):
        item = self._getBlockIdByName(name)
        result = [recipe for recipe_list in self.mcd.recipes.values() for recipe in recipe_list if self._inclusionCheck(item,
                                                                                                     recipe)]
        if not(metadata_filter is None):
            result = self._metadataListFiltering(result, metadata_filter, what_to_filter)
        if get_names_instead_of_ids:
            result = self._convertIdsToNamesForRecipeList(result)
        return result


mcdata_wrp = MCDataWrapper(mcd)

# print(len(mcdata_wrp.getItemOrBlockRecipeInclusions('stone')))
print(mcdata_wrp.getItemOrBlockRecipeInclusions('planks', metadata_filter=5, what_to_filter=mcdata_wrp.what_to_filter['ingredients']))
# print(mcd.version)
#
print(mcd.find_item_or_block(5))

# print(mcd.find_item_or_block('log2'))
# print(mcd.blocks_list)

#
# print(mcd.recipes['5'])
# print(mcd.recipes)