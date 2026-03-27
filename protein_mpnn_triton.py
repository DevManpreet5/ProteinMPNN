import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_mpnn_utils import (
    ProteinMPNN, EncLayer, DecLayer, PositionWiseFeedForward,
    cat_neighbors_nodes, PositionalEncodings
)
import triton_kernels


class TritonProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = min(self.top_k, X.shape[1])
        D_neighbors, E_idx = torch.topk(D_adjust, sampled_top_k, dim=-1, largest=False)
        return D_neighbors, E_idx

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        Cb = triton_kernels.virtual_cb(X)
        Ca = X[:,:,1,:].contiguous()
        N = X[:,:,0,:].contiguous()
        C = X[:,:,2,:].contiguous()
        O = X[:,:,3,:].contiguous()

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(triton_kernels.rbf(D_neighbors, 2., 22., self.num_rbf))

        atoms_list = [
            (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca),
            (C, N), (O, N), (Cb, N), (C, Cb), (O, Cb), (C, O)
        ]

        for A, B in atoms_list:
            RBF_all.append(triton_kernels.fused_rbf_dist(A, B, E_idx, 2., 22., self.num_rbf))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        from protein_mpnn_utils import gather_edges
        offset = residue_idx[:,:,None] - residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)

        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class TritonProteinMPNN(ProteinMPNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.features = TritonProteinFeatures(
            self.node_features,
            self.edge_features,
            top_k=kwargs.get('k_neighbors', 64),
            augment_eps=kwargs.get('augment_eps', 0.05)
        )
