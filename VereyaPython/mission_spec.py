
class MissionSpec:
    def __init__(self, xml: str,  validate: bool):
        self.validate(xml)
        self.xml = xml

    def validate(self, xml):
        pass
