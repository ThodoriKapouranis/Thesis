from dataclasses import dataclass
from Models.XGB import Batched_XGBoost
from absl import app, flags

class ConfigError(Exception):
    """Exception raised for errors in the training configuration
    
    Attributes:
        flag    --  flag the error originates from.
    """
    def __init__(self, flag, message="Error creating configuration."):
        super().__init__(f'\033[91m {flag} \033[97m: {message}')

# @dataclass 
# class TrainingConfig:
#     channels: int 
#     model: str

#     def initialize_model(self) -> any:
#         if self.model=='xgboost':
#             return Batched_XGBoost()


def validate_config(FLAGS:flags.FLAGS):
    if FLAGS.scenario not in [1,2,3]:
        raise ConfigError("scenario")
    channels = {1:2, 2:4, 3:6}[FLAGS.scenario]

    if FLAGS.model not in ['xgboost', 'unet', 'transunet']:
        raise ConfigError("model", "Model either not supported or not defined")
    
    if FLAGS.savename == None:
        raise ConfigError("savename", "Save name cannot be None ")

    return 0
