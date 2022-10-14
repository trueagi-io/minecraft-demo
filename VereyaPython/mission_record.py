from .mission_record_spec import MissionRecordSpec


class MissionRecord:
    def __init__(self, record: MissionRecordSpec):
        self.record = record
