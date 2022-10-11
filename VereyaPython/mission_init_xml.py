import xml.etree.ElementTree as ET


class MissionInitXML:
    def __init__(self, xml_text):
        self.tree = self.parse(xml_text)

    def parse(self, xml_text: str) -> None:
        print(xml_text)
        tree = ET.fromstring(xml_text)
        return tree


if __name__ == '__main__':
    m = MissionInitXML(open('miss.xml', 'rt').read())

