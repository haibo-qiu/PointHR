from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
import pointops

from pcr.models.builder import MODELS
from pcr.models.utils import offset2batch, batch2offset
import pdb


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat

class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPoolNoParams(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, grid_size):
        super(GridPoolNoParams, self).__init__()
        self.grid_size = grid_size

    def forward(self, points, cluster=None, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        if cluster is not None and type(cluster) == dict:
            sorted_cluster_indices = cluster['sorted_cluster_indices']
            idx_ptr = cluster['idx_ptr']
        else:
            start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                                reduce="min") if start is None else start
            cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
            unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            cluster = {'cluster': cluster,
                       'idx_ptr': idx_ptr,
                       'sorted_cluster_indices': sorted_cluster_indices,
                       }
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        cluster['coord'] = coord
        cluster['offset'] = offset
        return [coord, feat, offset], cluster


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, cluster=None, start=None, return_cluster=False):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        if cluster is not None and type(cluster) == dict:
            sorted_cluster_indices = cluster['sorted_cluster_indices']
            idx_ptr = cluster['idx_ptr']
        else:
            start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                                reduce="min") if start is None else start
            cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
            unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            cluster = {'cluster': cluster,
                       'idx_ptr': idx_ptr,
                       'sorted_cluster_indices': sorted_cluster_indices,
                       }
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        cluster['coord'] = coord
        cluster['offset'] = offset
        if return_cluster:
            return [coord, feat, offset], cluster
        else:
            return [coord, feat, offset]


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            cluster = cluster['cluster'] if type(cluster) == dict else cluster
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Unpool(nn.Module):
    """
    Map Unpooling
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 proj=False,
                 backend="map"
                 ):
        super(Unpool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True)) if proj else nn.Identity()

    def forward(self, points, cluster=None, skip_points=None):
        coord, feat, offset = points
        if self.backend == "map" and cluster is not None:
            cluster = cluster['cluster'] if type(cluster) == dict else cluster
            feat = self.proj(feat)[cluster]
        elif self.backend == "interp" and skip_points is not None:
            skip_coord, skip_feat, skip_offset = skip_points
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        else:
            raise ValueError("Unknown unpool!")
        return [coord, feat, offset]


class MRStreamBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_streams,
                 num_blocks,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(MRStreamBlock, self).__init__()

        self.stream_list = nn.ModuleList()
        for i in range(num_streams):
            blocks = BlockSequence(
                depth=num_blocks,
                embed_channels=in_channels*(2**i),
                groups=groups*(2**i),
                neighbours=neighbours,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
                drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
                enable_checkpoint=enable_checkpoint,
            )
            self.stream_list.append(blocks)

    def forward(self, x_list):
        return [f(x) for f, x in zip(self.stream_list, x_list)]


class MRFusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_streams
                 ):
        super(MRFusionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.act = nn.ReLU(inplace=True)
        self.fusion_list = self._make_layers(num_streams)

    def _make_layers(self, num_streams):
        fusion_list = nn.ModuleList()
        for i in range(num_streams):
            out_channels = self.out_channels * (2**i)
            layer_list = nn.ModuleList()
            for j in range(num_streams):
                in_channels = self.in_channels * (2**j)
                if j > i:
                    layers = nn.ModuleList()
                    for k in range(j-i):
                        layers.append(Unpool(in_channels, out_channels, proj=(k==0)))
                elif j == i:
                    layers = nn.Identity()
                else:
                    layers = nn.ModuleList()
                    for k in range(i-j):
                        layers.append(GridPool(in_channels, in_channels*2, None))
                        in_channels *= 2
                    assert in_channels == out_channels
                layer_list.append(layers)
            fusion_list.append(layer_list)
        return fusion_list


    def forward(self, input):
        x_list, lookup = input
        y_list = []
        for i in range(len(x_list)):
            layers = self.fusion_list[i]
            x = x_list[i]
            results = []
            for j in range(len(x_list)):
                if j > i:
                    out = x_list[j]
                    for k in range(j-i):
                        out = layers[j][k](out, lookup[(j-k-1, j-k)])
                elif j == i:
                    out = layers[j](x)
                else:
                    out = x_list[j]
                    for k in range(i-j):
                        out = layers[j][k](out, lookup[(j+k, j+k+1)])
                results.append(out[1])      # only works on feature
            y = self.act(torch.stack(results).sum(0))
            y_list.append([x[0], y, x[-1]])

        return y_list


class MRModule(nn.Module):
    def __init__(self,
                 in_channels,
                 num_streams,
                 num_blocks,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(MRModule, self).__init__()

        self.stream_layer = MRStreamBlock(
                in_channels=in_channels,
                num_streams=num_streams,
                num_blocks=num_blocks,
                groups=groups,
                grid_size=grid_size,
                neighbours=neighbours,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                enable_checkpoint=enable_checkpoint
            )
        self.fusion_layer = MRFusionBlock(in_channels, num_streams)

    def forward(self, input):
        x_list, lookup = input
        y_list = self.stream_layer(x_list)
        y_list = self.fusion_layer([y_list, lookup])
        return y_list, lookup


class MRPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_streams
                 ):
        super(MRPool, self).__init__()
        self.num_streams = num_streams
        self.transform_layers = nn.ModuleList()
        for i in range(num_streams):
            if in_channels != out_channels:
                layer = nn.Sequential(
                    nn.Linear(in_channels*(2**i), out_channels*(2**i)),
                    PointBatchNorm(out_channels*(2**i)),
                    nn.ReLU(inplace=True)
                )
                self.transform_layers.append(layer)
            else:
                self.transform_layers.append(nn.Identity())

        self.new_stream = GridPool(in_channels*(2**(num_streams-1)),
                                   out_channels*(2**num_streams), None)
    def forward(self, input):
        x_list, lookup = input
        y_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            out = self.transform_layers[i](x[1])
            y_list.append([x[0], out, x[-1]])
        y_new = self.new_stream(x_list[-1], lookup[(self.num_streams-1, self.num_streams)])
        y_list.append(y_new)
        return y_list, lookup


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


@MODELS.register_module("pointhr_semseg")
class PointHR(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(1, 1, 3, 2),
                 enc_blocks=(2, 2, 2, 2),
                 enc_streams=(1, 2, 3, 4),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(PointHR, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.streams = enc_streams
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )
        self.down = GridPool(
            in_channels=patch_embed_channels,
            out_channels=enc_channels[0],
            grid_size=grid_sizes[0],
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        # enc_channels = [patch_embed_channels] + list(enc_channels)
        enc_channels = list(enc_channels)
        dec_channels = [patch_embed_channels] + [enc_channels[-1]*(2**i) for i in range(len(enc_channels))]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stages):
            layers = []
            for j in range(enc_depths[i]):
                layer = MRModule(
                    in_channels=enc_channels[i],
                    num_streams=enc_streams[i],
                    num_blocks=enc_blocks[i],
                    groups=enc_groups[i],
                    grid_size=grid_sizes[i],
                    neighbours=enc_neighbours[i],
                    qkv_bias=attn_qkv_bias,
                    pe_multiplier=pe_multiplier,
                    pe_bias=pe_bias,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=enc_dp_rates[cur],
                    enable_checkpoint=enable_checkpoint
                )
                cur += 1
                layers.append(layer)
            if i < self.num_stages - 1:
                pool = MRPool(in_channels=enc_channels[i],
                              out_channels=enc_channels[i + 1],
                              num_streams=enc_streams[i])
                layers.append(pool)
            self.enc_stages.append(nn.Sequential(*layers))

            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=dec_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.dec_stages.append(dec)

        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

    def _cal_updown_idx(self, points):
        updown_idx = {}
        max_stream = self.streams[-1]

        # cal updown sampling index
        for i in range(max_stream-1):
            grid_size = self.grid_sizes[i+1]
            downsample = GridPoolNoParams(grid_size)
            points, cluster = downsample(points)
            updown_idx[(i, i+1)] = cluster
        return updown_idx

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        p0 = self.patch_embed(points)    # N
        points, cluster = self.down(p0, return_cluster=True)  # N/6
        cluster_ori = cluster['cluster']
        del cluster

        lookup = self._cal_updown_idx(points)

        x_list = [points]
        for i in range(self.num_stages):
            x_list, lookup = self.enc_stages[i]([x_list, lookup])

        # add back to ori resolution index
        lookup[(-1, 0)] = cluster_ori
        x_list.insert(0, p0)
        del p0, cluster_ori

        points = x_list.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = x_list.pop(i), lookup.pop((i-1, i))
            points = self.dec_stages[i](points, skip_points, cluster)

        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits
