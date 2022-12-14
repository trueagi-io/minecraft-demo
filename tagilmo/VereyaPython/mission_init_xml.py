import logging
from typing import Optional
import copy
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass
from .xml_util import get, get_optional, str2xml


logger = logging.getLogger()

MALMO_NAMESPACE = 'http://ProjectMalmo.singularitynet.io'
XMLNS_XSI = "http://www.w3.org/2001/XMLSchema-instance"

@dataclass(init=True, slots=True)
class MinecraftServer:
    connection_address: Optional[str] = None
    connection_port: Optional[int] = None


@dataclass(init=True, slots=True)
class ClientAgentConnection:
    client_mission_control_port: int = 0
    client_commands_port: int = 0
    agent_mission_control_port: int = 0
    agent_video_port: int = 0
    agent_depth_port: int = 0
    agent_observations_port: int = 0
    agent_lumunance_port: int = 0
    agent_rewards_port: int = 0
    agent_colour_map_port: int = 0

    client_ip_address: str = ''
    agent_ip_address: str = ''


@dataclass(slots=True)
class MissionInitXML:
    schema_version: str
    platform_version: str
    mission: Element
    experiment_uid: str
    client_role: int

    minecraft_server: MinecraftServer
    client_agent_connection: ClientAgentConnection

    def __init__(self, xml_text=None):
        self.client_agent_connection = ClientAgentConnection()
        self.minecraft_server = MinecraftServer()
        self.mission = None
        self.client_role = 0
        self.schema_version = ''
        self.platform_version = ''
        self.experiment_uid = ''
        if xml_text:
            self.parse(xml_text)

    def parse(self, xml_text: str) -> None:
        el = str2xml(xml_text)
        self.mission = get(el, "MissionInit.Mission")
        el = copy.deepcopy(el)
        self.experiment_uid = get(el, "MissionInit.ExperimentUID", True, str)
        address_element = get_optional(str, el, "MissionInit.MinecraftServerConnection.<xmlattr>.address")
        if address_element:
            self.minecraft_server.connection_address = address_element.get().strip()
        else:
            self.minecraft_server.connection_address = None

        self.minecraft_server.connection_port = get_optional(int, el, "MissionInit.MinecraftServerConnection.<xmlattr>.port").get_value_or(None)
        self.client_role = get(el, "MissionInit.ClientRole", True, int)
        self.schema_version = get(el, "MissionInit.<xmlattr>.SchemaVersion", True, str)
        self.platform_version = get(el, "MissionInit.<xmlattr>.PlatformVersion", True, str)

        self.client_agent_connection.client_ip_address = get_optional(str, el, "MissionInit.ClientAgentConnection.ClientIPAddress").get_value_or("")
        self.client_agent_connection.client_mission_control_port = get_optional(int, el, "MissionInit.ClientAgentConnection.ClientMissionControlPort").get_value_or(0)
        self.client_agent_connection.client_commands_port = get_optional(int, el, "MissionInit.ClientAgentConnection.ClientCommandsPort").get_value_or(0)
        self.client_agent_connection.agent_ip_address = get_optional(str, el, "MissionInit.ClientAgentConnection.AgentIPAddress").get_value_or("")
        self.client_agent_connection.agent_mission_control_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentMissionControlPort").get_value_or(0)
        self.client_agent_connection.agent_video_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentVideoPort").get_value_or(0)
        self.client_agent_connection.agent_depth_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentDepthPort").get_value_or(0)
        self.client_agent_connection.agent_lumunance_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentLuminancePort").get_value_or(0)
        self.client_agent_connection.agent_observations_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentObservationsPort").get_value_or(0)
        self.client_agent_connection.agent_rewards_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentRewardsPort").get_value_or(0)
        self.client_agent_connection.agent_colour_map_port = get_optional(int, el, "MissionInit.ClientAgentConnection.AgentColourMapPort").get_value_or(0)

    def toXml(self) -> str:
        el = ET.Element('MissionInit')
        el.attrib['xmlns'] =  MALMO_NAMESPACE
        # el.attrib['xmlns:xsi'] = XMLNS_XSI
        el.attrib['SchemaVersion'] = self.schema_version
        el.attrib['PlatformVersion'] = self.platform_version
        el.append(self.mission)
        get(el, "MissionInit.ExperimentUID").text = self.experiment_uid
        if self.minecraft_server.connection_address:
            get(el, "MissionInit.MinecraftServerConnection").attrib['address'] = self.minecraft_server.connection_address
        if self.minecraft_server.connection_port:
            get(el, "MissionInit.MinecraftServerConnection").attrib['port'] = str(self.minecraft_server.connection_port)
        get(el, "MissionInit.ClientRole").text = str(self.client_role)
        get(el, "MissionInit.ClientAgentConnection.ClientIPAddress").text = self.client_agent_connection.client_ip_address
        assert self.client_agent_connection.client_mission_control_port != 0
        get(el, "MissionInit.ClientAgentConnection.ClientMissionControlPort").text = str(self.client_agent_connection.client_mission_control_port)
        get(el, "MissionInit.ClientAgentConnection.ClientCommandsPort").text = str(self.client_agent_connection.client_commands_port)
        get(el, "MissionInit.ClientAgentConnection.AgentIPAddress").text = self.client_agent_connection.agent_ip_address
        get(el, "MissionInit.ClientAgentConnection.AgentMissionControlPort").text = str(self.client_agent_connection.agent_mission_control_port)
        get(el, "MissionInit.ClientAgentConnection.AgentVideoPort").text = str(self.client_agent_connection.agent_video_port)
        get(el, "MissionInit.ClientAgentConnection.AgentDepthPort").text = str(self.client_agent_connection.agent_depth_port)
        get(el, "MissionInit.ClientAgentConnection.AgentLuminancePort").text = str(self.client_agent_connection.agent_lumunance_port)
        get(el, "MissionInit.ClientAgentConnection.AgentObservationsPort").text = str(self.client_agent_connection.agent_observations_port)
        get(el, "MissionInit.ClientAgentConnection.AgentRewardsPort").text = str(self.client_agent_connection.agent_rewards_port)
        get(el, "MissionInit.ClientAgentConnection.AgentColourMapPort").text = str(self.client_agent_connection.agent_colour_map_port)
        el.attrib.pop('xmlns')
        # somehow there are two xmlns attributes
        result = ET.tostring(el, encoding='unicode', method='xml')
        logger.debug("mission xml is \n %s \n", result)
        return result.replace('\n', '')

if __name__ == '__main__':
    MissionInitXML(open('miss1.xml', 'rt').read()) 
