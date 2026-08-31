
from .base.legged_robot import LeggedRobot






from .go2.go2 import Go2
from .go2.go2_config_baseline import Go2BaseCfg, Go2BaseCfgPPO
from .dog import DogBackflip, DogBackflipCfg, DogBackflipCfgPPO



from legged_gym.utils.task_registry import task_registry


task_registry.register("go2", Go2, Go2BaseCfg(), Go2BaseCfgPPO())
task_registry.register(
    "dog_backflip",
    DogBackflip,
    DogBackflipCfg(),
    DogBackflipCfgPPO(),
)
task_registry.register(
    "dog",
    DogBackflip,
    DogBackflipCfg(),
    DogBackflipCfgPPO(),
)
