import logging
from dataclasses import dataclass
from typing import Dict
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


logger = logging.getLogger()


@dataclass(slots=True, init=False)
class RewardXML:
    reward_values: Dict[int, float]

    def __init__(self, xml_text: str=''):
        self.reward_values = dict()
        if xml_text:
            self.parse_rewards(xml_text)

    def parse_rewards(self, xml_text) -> None:
        logger.debug('parsing reward: \n %s', xml_text)
        root = ET.fromstring(xml_text)
        for child in root.findall("./Reward"):
            if child.tag == "Value":
                dimension = int(child.attrib.get("dimension"))
                value = float(child.attrib.get("value"))
                self.reward_values[dimension] = value

    def add_rewards(self, reward_element: Element) -> None:
        if reward_element.tag == 'Rewards':
            sub1 = reward_element
        else:
            sub1 = ET.SubElement(reward_element, "Rewards")
        for (rk, rv) in self.reward_values.items():
            reward_value = ET.SubElement(sub1, "Value")
            reward_value.attrib["dimension"] = rk
            reward_value.attrib["value"] = rv

    def toXml(self) -> str:
        xml = Element('Rewards')
        self.add_rewards(xml)
        result = ET.tostring(xml, encoding='unicode', method='xml')
        return result.strip()

    def size(self) -> int:
        return len(self.reward_values)
