import argparse
import sys
from timeit import default_timer as timer

import dgl
import dgl.function as fn
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

dgl_builtin = {
    "sum": fn.sum,
    "max": fn.max,
    "min": fn.min,
    "mean": fn.mean,
    "mul": fn.u_mul_e,
    "add": fn.u_add_e,
    "sub": fn.u_sub_e,
    "div": fn.u_div_e,
    "copy": fn.copy_e,
}
torch_builtin = {
    "mul": torch.mul,
    "add": torch.add,
    "sub": torch.sub,
    "div": torch.div,
    "copy": None,
}


class RelationalGNN(nn.Module):
    def __init__(self, embeddings, n_layers, op, reduce_op, use_fused):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                RelationalLayer(embeddings[i], op, reduce_op, use_fused)
                for i in range(n_layers)
            ]
        )

    def forward(self, graph, node_features, use_fused):
        for layer in self.layer_stack:
            node_features = layer(graph, node_features, use_fused)
        return node_features


class RelationalLayer(nn.Module):
    def __init__(self, embeddings, op, reduce_op, use_fused):
        super().__init__()
        self.relation_embeddings = embeddings
        if use_fused:
            self.op = dgl_builtin[op]
        else:
            self.op = torch_builtin[op]
        self.reduce_op = dgl_builtin[reduce_op]

    def message(self, edges):
        node_input = edges.src["h"]
        edge_embeddings = self.relation_embeddings[edges.data["etype"]]

        if self.op == None:
            message = edge_embeddings
        else:
            message = self.op(node_input, edge_embeddings)
        return {"edge_msg": message}

    def forward(self, graph, node_features, use_fused):
        if use_fused:
            graph.ndata["h"] = node_features
            if self.op == fn.copy_e:
                graph.update_all_relationwise(
                    self.op("etype", "edge_msg"),
                    self.reduce_op("edge_msg", "out"),
                    relation_feats=self.relation_embeddings,
                )
            else:
                graph.update_all_relationwise(
                    self.op("h", "etype", "edge_msg"),
                    self.reduce_op("edge_msg", "out"),
                    relation_feats=self.relation_embeddings,
                )
        else:
            graph.ndata["h"] = node_features
            graph.update_all(self.message, self.reduce_op("edge_msg", "out"))
        output = graph.ndata["out"]
        return output


@pytest.mark.parametrize("op", ["add", "mul", "sub", "div", "copy"])
@pytest.mark.parametrize("reduce_op", ["sum", "mean", "min", "max"])
def test_update_all_relationwise(op, reduce_op):
    dataset = dgl.data.FB15k237Dataset()
    graph = dataset[0]

    n_relations = graph.edata["etype"].max().item() + 1
    dim = 128
    layers = 4

    node_features = torch.randn(
        graph.number_of_nodes(), dim, requires_grad=True
    )
    node_labels = torch.randint(0, dim - 1, (graph.number_of_nodes(),))

    # creation of relation embeddings for each layer
    rel_embeddings = []
    rel_fused_embeddings = []
    for _ in range(layers):
        rels = torch.randn(n_relations, dim)
        rel_embed = nn.Parameter(rels)
        rel_fused_embed = nn.Parameter(rels)
        rel_embeddings += [rel_embed]
        rel_fused_embeddings += [rel_fused_embed]

    # getting results for fused version
    lfunc_fused = nn.CrossEntropyLoss()
    inst_fused = RelationalGNN(
        rel_fused_embeddings, layers, op, reduce_op, use_fused=True
    )
    opt_fused = torch.optim.Adam(inst_fused.parameters())

    inst_fused.train()
    logits_fused = inst_fused(graph, node_features, use_fused=True)
    loss_fused = lfunc_fused(logits_fused, node_labels)
    opt_fused.zero_grad()
    loss_fused.backward()
    opt_fused.step()

    # getting results for unfused version
    lfunc = nn.CrossEntropyLoss()
    inst = RelationalGNN(rel_embeddings, layers, op, reduce_op, use_fused=False)
    opt = torch.optim.Adam(inst.parameters())

    inst.train()
    logits = inst(graph, node_features, use_fused=False)
    loss = lfunc(logits, node_labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # comparing the results
    torch.allclose(logits_fused, logits)
    if (
        inst_fused.layer_stack[0].relation_embeddings.grad is not None
        or inst.layer_stack[0].relation_embeddings.grad is not None
    ):
        torch.allclose(
            inst_fused.layer_stack[0].relation_embeddings.grad,
            inst.layer_stack[0].relation_embeddings.grad,
        )
