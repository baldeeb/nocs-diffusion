import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize

import torch 

class ConfigLoader:
    """Loads config and initializes pieces when requested. 
    Provides an easy and efficient way to access the package's models."""

    def __init__(self, 
                 config: DictConfig,
                 checkpoint_path=None):
        self.cfg = config
        self._ckpt_path = checkpoint_path
        self._model = None
        self._dataloader = None

    @staticmethod
    def using_hydra_main(version_base=None, 
                         config_path=None, 
                         config_name=None, 
                         job_name=None, 
                         checkpoint_path=None):
        # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
        with initialize(version_base=version_base, 
                        config_path=config_path, 
                        job_name=job_name):
            cfg = compose(config_name=config_name)
        return ConfigLoader(cfg, checkpoint_path)

    @staticmethod
    def from_config_path(path):
        cfg_path = Path(path)
        rel_path = os.path.relpath(str(cfg_path.parent), 
                                   str(Path(__file__).parent))
        return ConfigLoader.using_hydra_main(config_name=str(cfg_path.name), 
                            config_path=str(rel_path))
    @property
    def model(self):
        if self._model is None:
            self._model = hydra.utils.instantiate(self.cfg.model)
            self._model = self._model.to(self.cfg.device)
            if self._ckpt_path:
                sd = torch.load(self._ckpt_path)
                self._model.load_state_dict(sd)
        return self._model

    @property
    def dataloader(self):
        if self._dataloader is None: 
            self._dataloader = hydra.utils.instantiate(self.cfg.dataloader)
            self._dataloader = self._dataloader.to(self.cfg.device)
        return self._dataloader


    @staticmethod
    def load_from_checkpoint(path):
        ''' 
        @brief Assumes that the checkpoint directory contains the .hydra
            config directory that has already been composed. Constructs the 
            model using .hydra config 
        '''
        ckpt_path = Path(path)
        cfg_path = ckpt_path.parent / ".hydra/config.yaml"
        return ConfigLoader(OmegaConf.load(cfg_path), ckpt_path)

    @staticmethod
    def load_model_from_checkpoint(path): 
        ckpt_path = Path(path)
        cfg = OmegaConf.load(ckpt_path.parent / ".hydra/config.yaml")
        model = hydra.utils.instantiate(cfg.model).to(cfg.device)
        model.load_state_dict(torch.load(ckpt_path))
        return model
 
    @staticmethod
    def load_dataloader_from_checkpoint(path): 
        return ConfigLoader.load_from_checkpoint(path).dataloader()