import torch
from torchview import draw_graph

from nocs_diffusion.models.blocks import ConditionedPointNetEncoder, PointNetEncoder

device = 'cuda'
B = 10
N = 20
D = 3
Z = 1

GRAPH_OUT_FORMAT = 'png'
OUT_DIRECTORY = './data/sandbox/'

def test_and_graph_pointnet():
    data = torch.rand(B, N, D).to(device)
    net = PointNetEncoder([D, 32, 64]).to(device)
    print(f'network:\n{net}')

    output = net(data)
    print(f'result shape: {output.shape}')
    
    model_graph = draw_graph(net, input_data=(data), expand_nested=True)
    model_graph.visual_graph.render('PointNetEncder', 
                                    directory=OUT_DIRECTORY,
                                    format=GRAPH_OUT_FORMAT)


def test_and_graph_coditioned_pointnet():
    data = torch.rand(B, N, D).to(device)
    ctxt = torch.rand(B, Z).to(device)
    net = ConditionedPointNetEncoder(D, [32, 64, 128], 256, Z, [128]).to(device)
    net(data, ctxt)
    net(data)

    model_graph = draw_graph(net, input_data=(data, ctxt), expand_nested=True)
    model_graph.visual_graph.render('ConditionedPointNetEncoder', 
                                    directory=OUT_DIRECTORY,
                                    format=GRAPH_OUT_FORMAT)


if __name__ == "__main__":
    test_and_graph_pointnet()
    # test_and_graph_coditioned_pointnet()