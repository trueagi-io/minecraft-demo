import xml.etree.ElementTree as ET

class MissionInitXML:
    def __init__(self, xml_text):
        self.parse(xml_text)

    def parse(self, xml_text: str) -> None:
        print(xml_text)
        tree = ET.parse('country_data.xml')
        root = tree.getroot()
        self.mission = root.findall("./MissionInit.Mission");
