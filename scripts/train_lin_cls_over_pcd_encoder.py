
from nocs_diffusion.utils import viz_image_batch, ConfigDirectoriesManager, ConfigLoader
from argparse import ArgumentParser

from tqdm import tqdm 

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser(
                    prog='PCD Encoder Linear Classifier',
                    description='Trains a linear classifier over a pre-trained pcd encoder.',
                    epilog='Text at the bottom of help')
    parser.add_argument("checkpoint",
                    help="First argument should point to a checkpoint." + \
                         "The director of the checkpoint is expected to house" + \
                         ".hydra/config.yaml",)
    args = parser.parse_args()
    print(args.checkpoint)
    loader = ConfigLoader.load_from_checkpoint(args.checkpoint)
    
    cfg = loader.cfg
    model = loader.model.ctxt_net
    dataloader = loader.dataloader
    val_dataloder = loader.validator.dataloader
    
    dataloader.return_dict.append('category_ids')
    val_dataloder.return_dict.append('category_ids')
    num_cats = len(dataloader.renderer.dataset.cate_synsetids)
    lin_classifier = nn.Sequential(
        nn.Linear(model.dims[-1], num_cats),
        nn.Sigmoid(),
    ).to(cfg["device"])

    loss_fx = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(lin_classifier.parameters(), lr=1e-4)

    loss_tracker = []    
    batch_tqdm = tqdm(range(5000), desc='Training Step Loop')
    for batch_i in batch_tqdm:

        data = dataloader()
        point_embeddings = model(data['face_points']).squeeze(1)
        pred = lin_classifier(point_embeddings)
        loss = loss_fx(pred, data['category_ids']) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.append(loss.item())
        if (batch_i % 50) == 0: 
            batch_tqdm.set_description(f'loss {loss:.3f}')

    
    plt.plot(loss_tracker)
    plt.title("Training Loss")
    plt.savefig("./data/sandbox/eval_depth_encoding_training_curve")

    result_dict = {i:0 for i in range(num_cats)}
    pred_counts = {i:0 for i in range(num_cats)}
    val_range = tqdm(range(cfg['validate']['num_inference_steps']))
    for _ in val_range:
        data = val_dataloder()
        
        point_embeddings = model(data['face_points']).squeeze(1)
        pred = lin_classifier(point_embeddings)
        
        pred_labels = torch.max(pred, dim=0).indices
        gt_labels = data['category_ids'].to(cfg["device"])

        for pred, gt in zip(pred_labels, gt_labels):
            g, p = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
            result_dict[gt.item()] += int( pred.item() == gt.item() )
            pred_counts[gt.item()] += 1
    
    
    print('Evaluating....')
    print("results")
    for cat_id in range(num_cats):
        print(f"category {cat_id}: {result_dict[cat_id]} / {pred_counts[cat_id]}")

