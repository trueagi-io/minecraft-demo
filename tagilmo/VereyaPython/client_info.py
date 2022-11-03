from dataclasses import dataclass


default_client_mission_control_port = 10000


@dataclass(frozen=True, slots=True, init=False, eq=True)
class ClientInfo:
    '''Structure containing information about a simulation client's address and port'''
    ip_address: str
    control_port: int
    command_port: int

    def __init__(self, ip_address: str,
                       control_port: int=default_client_mission_control_port, 
                       command_port: int=0):
        object.__setattr__(self, "ip_address", ip_address)
        object.__setattr__(self, "control_port", control_port)
        object.__setattr__(self, "command_port", command_port)
