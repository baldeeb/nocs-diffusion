

from utils.visualization import viz_image_batch
from utils.load_save import save_model

from omegaconf import OmegaConf
from tqdm import tqdm
import pathlib as pl
import wandb

def train(config, model, optimizer, lr_scheduler, dataloader):

    # Logger
    if config.log: 
        wandb.init(**config.logger, config=OmegaConf.to_container(config))
        log = wandb.log
    else: log = lambda x: None

    epoch = 0
    batch_tqdm = tqdm(range(config.num_epochs), 
                      desc='Training Step Loop')
    for batch_i in batch_tqdm:
        log({'step': batch_i+1})

        x, y = dataloader()
        loss = model.loss(x, y)
        log(loss)
        
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        lr_scheduler.step()

        if config.steps_before_save and (batch_i % config.steps_before_save) == 0:
            batch_tqdm.set_description(f'Saving batch {batch_i}, loss {loss["loss"]:.2f}')
            save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                        retain_n=config.get('retain_n_checkpoints', None))

    save_model(model, pl.Path(config.checkpoint_dir), epoch, batch_i,
                retain_n=config.get('retain_n_checkpoints', None))



def visualize_sample(dataloader, model,):
    x, y = dataloader()
    y_hat, _, _, = model(x)
    as_np = lambda x: (x/2 + 0.5).permute(0,2,3,1).detach().cpu().numpy()
    viz_image_batch(as_np(y), block=False, title='Original')
    viz_image_batch(as_np(y_hat), title='Reconstructed')