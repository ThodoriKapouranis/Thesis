from dataclasses import dataclass


class ConfigError(Exception):
    """Exception raised for errors in the training configuration
    
    Attributes:
        flag    --  flag the error originates from.
    """
    def __init__(self, flag, message="Error creating configuration. Problem flags: "):
        super().__init__(message + flag)

@dataclass 
class TrainingConfig:
    channels: int 
    model: str

def create_config(scenario:int, model:str):
    if scenario not in [1,2,3]:
        raise ConfigError("scenario")
    channels = {1:2, 2:4, 3:6}[scenario]

    if model not in ['xgboost', 'unet', 'a-unet']:
        raise ConfigError("model")

    return TrainingConfig(channels, model)
