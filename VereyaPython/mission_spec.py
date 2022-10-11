import xml.etree.ElementTree as ET


class MissionSpec:
    def __init__(self, xml: str,  validate: bool):
        self.validate(xml)
        self.mission = ET.fromstring(xml)

    def validate(self, xml):
        pass

    def isVideoRequested(role: int) -> bool:
        return getRoleValue(role, "AgentHandlers.VideoProducer", 'x') is not None

    def getRoleValue(self, role: int, videoType: str, what: int) -> None:
        import pdb;pdb.set_trace() 
