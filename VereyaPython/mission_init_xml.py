import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from dataclasses import dataclass


@dataclass(init=True, slots=True)
class MinecraftServer: 
    connection_address: str = None
    connection_port: int = None


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
            self.mission = self.parse(xml_text)

    def parse(self, xml_text: str) -> None:
        print(xml_text)
        tree = ET.fromstring(xml_text)
        import pdb;pdb.set_trace()
        return tree


if __name__ == '__main__':
    m = MissionInitXML(open('miss.xml', 'rt').read())

