from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ClientInfo:
    '''Structure containing information about a simulation client's address and port'''
    ip_address: str
    control_port: int
    command_port: int
