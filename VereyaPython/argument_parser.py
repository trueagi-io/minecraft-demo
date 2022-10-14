from typing import List


class ArgumentParser:
    def parse(self, args: List[str]) -> None:
        """ Parses a list of strings. Throws std::exception if parsing fails.
            In C++: takes a std::vector<std::string>.
            In Python: takes a list of strings.
            param args The arguments to parse.
        """
        pass

    def receivedArgument(self, name: str) -> bool: 
        """
        Gets whether a named argument was parsed on the command-line arguments.
        param name The name of the argument.
        returns True if the named argument was received
        """
        return False
