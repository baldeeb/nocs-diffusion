import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize


class ConfigLoader:
    """Loads config and initializes pieces when requested. 
    Provides an easy and efficient way to access the package's models."""

    def __init__(self, version_base=None, config_path=None, config_name=None, job_name=None):
        # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
        with initialize(version_base=version_base, config_path=config_path, job_name=job_name):
            self.cfg = compose(config_name=config_name)
        self._model = None
        self._dataloader = None
        
    @staticmethod
    def from_config_path(path):
        cfg_path = Path(path)
        rel_path = os.path.relpath(str(cfg_path.parent), str(Path(__file__).parent))
        return ConfigLoader(config_name=str(cfg_path.name), 
                            config_path=str(rel_path))
    @property
    def model(self):
        if self._model is None:
            self._model = hydra.utils.instantiate(self.cfg.model).to(self.cfg.device)
        return self._model

    @property
    def dataloader(self):
        if self._dataloader is None: 
            self._dataloader = hydra.utils.instantiate(self.cfg.dataloader).to(self.cfg.device)
        return self._dataloader
