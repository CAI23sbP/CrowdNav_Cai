from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn 
import torch as th 
from typing import Union, Dict
from stable_baselines3.common.type_aliases import TensorDict

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 1, -1)  
        
class ExampleExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: Union[spaces.Box, spaces.Dict], 
        features_dim: int = 128
        ):
        super().__init__(observation_space, features_dim)

        if isinstance(observation_space, spaces.Box):

            self.layer = nn.Sequential(nn.Linear(24, 128),
                            nn.ReLU(),
                            nn.Linear(128, 128),
                            nn.ReLU(),
                        )
            
        elif isinstance(observation_space, spaces.Dict):
            extractors: Dict[str, nn.Module] = {}
            for key in observation_space.spaces.keys():
                if key == 'ranges':
                    extractors[key] = nn.Sequential(
                                                    nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, stride=3),  ## [1, 32, 49, 49]
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3),  ## [1, 64, 23, 23]
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  ## [1, 64, 21, 21]
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),  ## [1, 64, 21, 21]
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
                                                    Flatten(),
                                                    nn.Linear(992,256),
                                                    nn.ReLU(),
                                                    )
                elif key == 'robotstate_obs':
                    extractors[key] = nn.Sequential(
                                                    nn.Linear(5,16),
                                                    nn.ReLU(),
                                                    )
            self.extractors = nn.ModuleDict(extractors)
            self.concat_layer = nn.Sequential(
                                            nn.Linear(16+256, 128), 
                                            nn.ReLU()
                                            )

        else:
            raise TypeError("it is not supported types, except (Dict, Box).")

    def forward(self, observations: Union[TensorDict, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key == 'ranges':
                encoded_tensor_list.append(extractor(observations[key]).squeeze(1))

            else:
                robot_info = observations[key][:, -1, :]
                encoded_tensor_list.append(extractor(robot_info))
        
        return self.concat_layer(th.cat(encoded_tensor_list, dim= -1)) 
    