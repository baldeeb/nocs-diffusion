import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf


class ConfigLoader:
    """Loads config and initializes pieces when requested. 
    Provides an easy and efficient way to access the package's models."""

    def __init__(self, version_base=None, config_path=None, config_name=None, job_name=None):
        # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
        with initialize(version_base=version_base, config_path=config_path, job_name=job_name):
            self.cfg = compose(config_name=config_name)
        self._model = None
        self._dataloader = None
        
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
