import tvm
from topi.util import get_const_tuple


def vanilla_spmm(SrcFeat,
                 Adj_s1_pos,
                 Adj_s1_idx,
                 Adj_vals,
                 d1_size,
                 d2_size,
                 num_feat_partitions=1):
    """Comput sparse-dense matrix multiplication of Adj and SrcFeat

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_src_vertices, feat_len]

    Adj_s1_pos : tvm.Tensor
        1-D with shape [d1_size * d2_size] (DDS)

    Adj_s1_idx : tvm.Tensor
        1-D with shape [nnz] (DDS)

    Adj_vals : tvm.Tensor
        1-D with shape [nnz] (DDS)

    d1_size : int
        Number of src vertex partitions

    d2_size : int
        num_dst_vertices + 1

    num_feat_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_len]
    """
    assert d1_size * d2_size == Adj_s1_pos.shape[0].value
    assert Adj_s1_idx.shape[0].value == Adj_vals.shape[0].value
    num_src_vertices, feat_len = get_const_tuple(SrcFeat.shape)
    num_src_vertex_partitions = d1_size
    num_dst_vertices = d2_size - 1
    oshape = (num_dst_vertices, feat_len)

    feat_len_per_partition = feat_len // num_feat_partitions  # we assume feat_len % num_feat_partitions = 0
    num_src_vertices_per_partition = (num_src_vertices + num_src_vertex_partitions - 1) // num_src_vertex_partitions

    ReshapedSrcFeat = tvm.compute((num_feat_partitions, num_src_vertices, feat_len_per_partition), \
        lambda fo, nn, fi: SrcFeat[nn, fo * feat_len_per_partition + fi], name='ReshapedSrcFeat')

    def msgfunc(fo, src_vertex_partition_idx, row, fi):
        row_start = Adj_s1_pos[src_vertex_partition_idx * d2_size + row]
        row_end = Adj_s1_pos[src_vertex_partition_idx * d2_size + row + 1]
        row_num_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = Adj_vals[row_start + elem_idx]
        feat_val = ReshapedSrcFeat[fo, \
                                   Adj_s1_idx[row_start + elem_idx] + src_vertex_partition_idx * num_src_vertices_per_partition, \
                                   fi]
        return tvm.sum(adj_val * feat_val, axis=elem_idx)

    Intermediate = tvm.compute((num_feat_partitions, num_src_vertex_partitions, num_dst_vertices, feat_len_per_partition), \
        msgfunc, name='Intermediate')

    k = tvm.reduce_axis((0, num_src_vertex_partitions), name='src_vertex_partition_reduce')
    ReshapedOut = tvm.compute((num_feat_partitions, num_dst_vertices, feat_len_per_partition),
        lambda fo, nn, fi: tvm.sum(Intermediate[fo, k, nn, fi], axis=k), \
        name='ReshapedOut')

    Out = tvm.compute(oshape, \
        lambda nn, ff: ReshapedOut[ff // feat_len_per_partition, nn, ff % feat_len_per_partition], \
        name='Out')

    return Out


def schedule_vanilla_spmm_x86(Out, num_feat_partitions=1):
    s = tvm.create_schedule([Out.op])

    ReshapedOut = Out.op.input_tensors[0]
    Intermediate = ReshapedOut.op.input_tensors[0]
    ReshapedSrcFeat = Intermediate.op.input_tensors[3]

    I = Intermediate
    RO = ReshapedOut
    s[I.op].reorder(I.op.axis[0], I.op.axis[1], I.op.axis[2], I.op.reduce_axis[0], I.op.axis[3])
    s[RO.op].reorder(RO.op.axis[0], RO.op.reduce_axis[0], RO.op.axis[1], RO.op.axis[2])
    s[I.op].compute_at(s[RO], RO.op.reduce_axis[0])

    s[ReshapedSrcFeat.op].parallel(ReshapedSrcFeat.op.axis[1])
    s[Intermediate.op].parallel(Intermediate.op.axis[2])
    s[ReshapedOut.op].parallel(ReshapedOut.op.axis[1])
    s[Out.op].parallel(Out.op.axis[0])

    return s
