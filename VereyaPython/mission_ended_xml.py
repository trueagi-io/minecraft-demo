from typing import ClassVar, List
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass
from .reward_xml import RewardXML
from .xml_util import get, get_optional, get_child_optional
from .consts import *


@dataclass(slots=True, init=False)
class VideoDataAttributes:
    def __init__(self):
        self.frames_sent = 0
        self.frame_type = ''
        self.frames_received = None
        self.frames_written = None

    frame_type: str
    frames_sent: int
    frames_received: int
    frames_written: int


@dataclass(slots=True, init=False)
class MissionEndedXML:
    ENDED: ClassVar[str] = "ENDED"
    PLAYER_DIED: ClassVar[str] = "PLAYER_DIED"
    AGENT_QUIT: ClassVar[str] = "AGENT_QUIT"
    MOD_FAILED_TO_INSTANTIATE_HANDLERS: ClassVar[str] = "MOD_FAILED_TO_INSTANTIATE_HANDLERS"
    MOD_HAS_NO_WORLD_LOADED: ClassVar[str] = "MOD_HAS_NO_WORLD_LOADED"
    MOD_FAILED_TO_CREATE_WORLD: ClassVar[str] = "MOD_FAILED_TO_CREATE_WORLD"
    MOD_HAS_NO_AGENT_AVAILABLE: ClassVar[str] = "MOD_HAS_NO_AGENT_AVAILABLE"
    MOD_SERVER_UNREACHABLE: ClassVar[str] = "MOD_SERVER_UNREACHABLE"
    MOD_SERVER_ABORTED_MISSION: ClassVar[str] = "MOD_SERVER_ABORTED_MISSION"
    MOD_CONNECTION_FAILED: ClassVar[str] = "MOD_CONNECTION_FAILED"
    MOD_CRASHED: ClassVar[str] = "MOD_CRASHED"

    def __init__(self, xml_text: str):
        self.schema_version = ''
        self.status = ''
        self.human_readable_status = ''
        self.have_rewards = False
        self.reward = RewardXML()
        self.video_data_attributes = list()
        root = ET.fromstring(xml_text)
        self.schema_version = get_optional(str, root, "MissionEnded.<xmlattr>.SchemaVersion")
        self.status = get(root, "MissionEnded.Status", True, str)
        self.human_readable_status = get(root, "MissionEnded.HumanReadableStatus", True, str)

        reward_element = get_child_optional(root, "MissionEnded.Reward")
        self.have_rewards = reward_element is not None
        if self.have_rewards:
            self.reward.parse_rewards(reward_element)
            if self.reward.size() == 0:
                raise RuntimeError("Reward must have at least one value")

        for v in root.find("MissionDiagnostics"):
            if v.tag == "VideoData":
                attributes = VideoDataAttributes()

                attributes.frame_type = v.attrib["frameType"]
                attributes.frames_sent = int(v.attrib["framesSent"])
                attributes.frames_received = int(v.attrib.get("framesReceived", 0))
                attributes.frames_written = int(v.attrib.get("framesWritten", 0))

                self.video_data_attributes.append(attributes)


    def getStatus(self) -> str:
        return self.status

    def getHumanReadableStatus(self) -> str:
        return self.human_readable_status

    def getReward(self) -> RewardXML:
        return self.reward

    def videoDataAttributes(self) -> List[VideoDataAttributes]:
        return self.video_data_attributes

    def toXml(self) -> str:
        el = Element('MissionEnded')
        el.attrib['xmlns'] = MALMO_NAMESPACE
        el.attrib['xmlns:xsi'] = XMLNS_XSI
        if self.schema_version:
            el.attrib["SchemaVersion"] = self.schema_version
        sub = ET.SubElement(el, "Status")
        sub.text = self.status
        sub = ET.SubElement(el, 'HumanReadableStatus')
        sub.text = self.human_readable_status
        if self.have_rewards:
            self.reward.add_rewards(el)

        for d in self.video_data_attributes:
            videoData = ET.SubElement(el, 'MissionDiagnostics')
            videoData.attrib["frameType"] = d.frame_type
            videoData.attrib["framesSent"] = d.frames_sent


        xml_str = ET.tostring(el, encoding='unicode', method='xml')
        return xml_str

    schema_version: str
    status: str
    human_readable_status: str
    have_rewards: bool
    reward: RewardXML
    video_data_attributes: List[VideoDataAttributes]
