import torch.jit
from matplotlib import pyplot as plt
from torch import nn
import networkx
from torch.nn import modules

from utils.conv_unit import ConvUnit


class TestNW(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = ConvUnit(1, 10, 3, padding=1, bn=True)
        print(self.first.steps)
        self.left = ConvUnit(10, 10, 3, padding=1, bn=True)

        self.right = ConvUnit(10, 10, 3, padding=1, bn=True)
        self.last = ConvUnit(20, 1, 3, padding=1, bn=True)

    def forward(self, x):
        x = self.first(x)

        a = self.left(x)
        b = self.right(x)

        c = torch.concat((a, b), dim=1)
        return self.last(c)


if __name__ == '__main__':
    net = TestNW()
    net.eval()

    t = torch.ones((16, 16)).view(1, 1, 16, 16)

    source_of_output = {
        t: "START"
    }
    digraph = networkx.DiGraph()
    digraph.add_node("START")

    def hook(module, inp, outp):
        # find the soure of this input and add to that stream
        inp = inp[0]
        if isinstance(module, (nn.Sigmoid, nn.BatchNorm2d, nn.Conv2d)):
            try:
                parent_node = source_of_output[inp]
                digraph.add_node(module)
                digraph.add_edge(parent_node, module)
                source_of_output[outp] = module
            except KeyError:
                print(f"No originator found for {module.__class__.__name__}")

    for m in net.modules():
        m.register_forward_hook(hook)

    labels = {
        "START": "START",
        **{x: x.__class__.__name__ for x in net.modules()}
    }

    net(t)
    networkx.draw_kamada_kawai(digraph, with_labels=True)
    plt.show()

    # trace, _ = torch.jit._get_trace_graph(net, (t,))
    # graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    # print(type(graph))
    # for node in graph.nodes():
    #     print(dir(node))
    #     outputs = [o.unique() for o in node.outputs()]
    #     for target_node in graph.nodes():
    #         target_inputs = [i.unique() for i in target_node.inputs()]
    #         if set(outputs) & set(target_inputs):
    #             print(node.kind(), "connects to", target_node.kind())
