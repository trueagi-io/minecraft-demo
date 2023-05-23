from typing import Optional
import xml.etree.ElementTree as ET


class MissionSpec:
    def __init__(self, xml: str,  validate: bool):
        self.validate(xml)
        self.mission = ET.fromstring(xml)

    def validate(self, xml):
        pass

    def isVideoRequested(self, role: int) -> bool:
        return self.getRoleValue(role, "AgentHandlers.VideoProducer", 'x') is not None

    def getRoleValue(self, role: int, videoType: str, what: str) -> Optional[int]:
        for elem in self.mission.findall('./{*}AgentSection'):
            if role == 0:
                for vid in elem.findall('.//{*}' + videoType.split('.')[1]):
                    if what == 'x':
                        return 1
                    elif what == 'w':
                        tmp = vid.find('./{*}Width')
                        assert tmp is not None
                        return int(tmp.text)
                    elif what == 'h':
                        tmp = vid.find('./{*}Height')
                        assert tmp is not None
                        return int(tmp.text)
                    elif what == 'c':
                        if vid.attrib.get('want_depth', 'false') in ('true', '1'):
                            return 1
                        return 0
                    else:
                        raise RuntimeError("Invalid video attribute")
                return None
            role -= 1
        raise RuntimeError("No such role in agent section")

    def getNumberOfAgents(self) -> int:
        """Returns the number of agents involved in this mission.
        returns The number of agents."""
        return len(self.mission.findall('./{*}AgentSection'))

    def getVideoWidth(self, role: int) -> int:
        return self._getVideoHW(role, 'w')

    def getVideoHeight(self, role: int) -> int:
        return self._getVideoHW(role, 'h')

    def _getVideoHW(self, role: int, hw) -> int:
        w = self.getRoleValue(role, "AgentHandlers.VideoProducer", hw)
        if w is not None:
            return w
        w = self.getRoleValue(role, "AgentHandlers.DepthProducer", hw)
        if w is not None:
            return w
        w = self.getRoleValue(role, "AgentHandlers.LuminanceProducer", hw)
        if w is not None:
            return w
        w = self.getRoleValue(role, "AgentHandlers.ColourMapProducer", hw)
        if w is not None:
            return w
        raise RuntimeError("MissionInitSpec::getVideoWidth : video has not been requested for this role")

    def getVideoChannels(self, role: int) -> int:
        c = self.getRoleValue(role, "AgentHandlers.VideoProducer", 'c')
        if c is not None:
            if c == 1:
                return 4
            return 3
        raise RuntimeError("MissionInitSpec::getVideoChannels : video has not been requested for this role")

    def isDepthRequested(self, role: int) -> bool:
        return self.getRoleValue(role, "AgentHandlers.DepthProducer", 'x') is not None

    def isLuminanceRequested(self, role: int) -> bool:
        return self.getRoleValue(role, "AgentHandlers.LuminanceProducer", 'x') is not None

    def isColourMapRequested(self, role: int) -> bool:
        return self.getRoleValue(role, "AgentHandlers.ColourMapProducer", 'x') is not None
