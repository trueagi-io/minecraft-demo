from dataclasses import dataclass
from mission_spec import MissionSpec


@dataclass(slots=True, frozen=True)
class MissionInitSpec:
    mission_init: MissionSpec
    unique_experiment_id: str
    role: int

    def getAsXML(self, prettyPrint: bool) -> str: 
        return self.mission_init.toXml()
    
    def getExperimentID(self) -> str:
        return self.unique_experiment_id

    def getClientAddress(self) -> str:
        return self.mission_init.client_agent_connection.client_ip_address

    def setClientAddress(self, address: str) -> None:
        self.mission_init.client_agent_connection.client_ip_address = address.strip()

    def getClientMissionControlPort(self, ) -> None: 
        return self.mission_init.client_agent_connection.client_mission_control_port

    def setClientMissionControlPort(self, port: int):
        self.mission_init.client_agent_connection.client_mission_control_port = port
    
    def getClientCommandsPort(self) -> int:
        return self.mission_init.client_agent_connection.client_commands_port

    def setClientCommandsPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.client_commands_port = port
    
    def getAgentAddress(self) -> str: 
        return self.mission_init.client_agent_connection.agent_ip_address
    
    def setAgentAddress(self, address: str) -> None:
        self.mission_init.client_agent_connection.agent_ip_address = address

    def getAgentMissionControlPort(self) -> int:
        return self.mission_init.client_agent_connection.agent_mission_control_port

    def setAgentMissionControlPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_mission_control_port = port

    def getAgentVideoPort(self) -> int: 
        return self.mission_init.client_agent_connection.agent_video_port
    
    def getAgentDepthPort(self) -> int: 
        return self.mission_init.client_agent_connection.agent_depth_port

    def getAgentLuminancePort(self) -> int: 
        return self.mission_init.client_agent_connection.agent_lumunance_port

    def getAgentColourMapPort(self) -> int: 
        return self.mission_init.client_agent_connection.agent_colour_map_port

    def setAgentVideoPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_video_port = port
    
    def setAgentDepthPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_depth_port = port

    def setAgentLuminancePort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_lumunance_port = port
    
    def setAgentColourMapPort(self, port: int) -> int:
        self.mission_init.client_agent_connection.agent_colour_map_port = port

    def getAgentObservationsPort(self, ) -> int: 
        return self.mission_init.client_agent_connection.agent_observations_port
    
    def setAgentObservationsPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_observations_port = port
    
    def getAgentRewardsPort(self, ) -> int: 
        return self.mission_init.client_agent_connection.agent_rewards_port

    def setAgentRewardsPort(self, port: int) -> None:
        self.mission_init.client_agent_connection.agent_rewards_port = port

    def hasMinecraftServerInformation(self) -> bool:
        return bool(self.mission_init.minecraft_server.connection_address) or self.mission_init.minecraft_server.connection_port

    def setMinecraftServerInformation(self, address: str, port: int) -> None:
        self.mission_init.minecraft_server.connection_address = address.strip()
        self.mission_init.minecraft_server.connection_port = port

