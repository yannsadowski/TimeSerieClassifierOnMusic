from dataclasses import dataclass


@dataclass
class WandbConfig:
    api_key: str
    project_name: str