from core.trainer import Trainer
from core.configs import default_config


class TestDefaultConfig:
    def test_family(self):
        trainer = Trainer(default_config.cfg)
        trainer.start()

