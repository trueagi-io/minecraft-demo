import copy
from io import StringIO
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
            result = result.attrib.get(item, None)
            break
        elem = result.find(item)
        if elem is not None:
            result = elem
        else:
            if must_be:
                return None
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
        if self.value is not None:
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
    return Result(elem.text if elem is not None else None, _type)


def get_child_optional(el, path):
    path = path.split('.')
    assert path[0] == el.tag
    elem = el
    for item in path[1:]:
        elem = elem.find(item) 
        if elem is None:
            return elem
    return elem


def str2xml(xml: str) -> Element:
    it = ET.iterparse(StringIO(xml))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition('}') # strip ns
    root = it.root
    return root


def remove_namespaces(el):
    if el.tag.startswith('{'):
        _, _, el.tag = el.tag.rpartition('}')
    for child in el:
        remove_namespaces(child)


def xml_to_dict(el):
    result = {}
    
    if el.attrib:
        result.update({k: v for k, v in el.attrib.items()})

    for child in el:
        child_dict = xml_to_dict(child)
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict

    if el.text and el.text.strip():
        result['text'] = el.text.strip()

    return result