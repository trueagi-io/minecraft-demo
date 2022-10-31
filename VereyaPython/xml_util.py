import copy
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass



def get(el, path, must_be=False, _type=None):
    result = el
    xmlattr = False
    path = path.split('.')
    assert el.tag == path[0] 
    for item in path[1:]:
        if item == '<xmlattr>':
            xmlattr = True
            continue
        if xmlattr:
            result = elem.attrib.get(item, None)
            break
        elem = result.find(item)
        if elem:
            result = elem
        else:
            if must_be:
                raise RuntimeError('element {0} not found'.format(path))
            result = ET.SubElement(result, item)
    if _type:
        if xmlattr:
            return _type(result)
        else:
            return _type(result.text)
    return result


class Result:
    def __init__(self, value, _type=str):
        self.value = value
        self._type = _type

    def get_value_or(self, default):
        if self.value:
            return self._type(self.value)
        return default

    def get(self):
        assert self.value is not None
        return self._type(self.value)

    def __bool__(self):
        return bool(self.value)


def get_optional(_type, el, path):
    elem = get(el, path, must_be=False)
    if '<xmlattr>' in path:
        return Result(elem, _type)
    return Result(elem.text if elem else None, _type)


def get_child_optional(el, path):
    path = path.split('.')
    assert path[0] == el.tag
    elem = el
    for item in path[1:]:
        elem = elem.find(item) 
        if elem is None:
            return elem
    return elem
