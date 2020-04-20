import tvm
from topi.util import get_const_tuple

def mlp_conv_spmm_ir_x86(SrcFeat,
                         DstFeat,
                         Weight,
                         Adj_s1_pos,
                         Adj_s1_idx,
                         Adj_vals,
                         d1_size,
                         d2_size,
                         num_feat_1_partitions=1,
                         num_feat_2_partitions=1):
    """Message function is MLP. Here has only 1 layer.

    Parameters
    ----------
    SrcFeat : tvm.Tensor
        2-D with shape [num_src_vertices, feat_1_len]

    DstFeat : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_1_len]

    Weight : tvm.Tensor
        2-D with shape [feat_2_len, feat_1_len]

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

    num_feat_1_partitions : int
        Doing feature dimension tiling (not implemented yet)

    num_feat_2_partitions : int
        Doing feature dimension tiling

    Returns
    -------
    Out : tvm.Tensor
        2-D with shape [num_dst_vertices, feat_2_len]
    """
    num_src_vertices, feat_1_len = get_const_tuple(SrcFeat.shape)
    num_src_vertex_partitions = d1_size
    num_dst_vertices = d2_size - 1
    feat_2_len = get_const_tuple(Weight.shape)[0]
    oshape = (num_dst_vertices, feat_2_len)
    assert d1_size * d2_size == Adj_s1_pos.shape[0].value
    assert Adj_s1_idx.shape[0].value == Adj_vals.shape[0].value
    assert num_dst_vertices == DstFeat.shape[0].value
    assert feat_1_len == DstFeat.shape[1].value
    assert feat_1_len == Weight.shape[1].value

    # feat_1_len_per_partition = feat_1_len // num_feat_1_partitions  # we assume feat_1_len % num_feat_1_partitions = 0
    feat_2_len_per_partition = feat_2_len // num_feat_2_partitions  # we assume feat_2_len % num_feat_2_partitions = 0
    num_src_vertices_per_partition = (num_src_vertices + num_src_vertex_partitions - 1) // num_src_vertex_partitions

    def _mlp_conv_spmm_x86_ir(SrcFeat,
                              DstFeat,
                              Weight,
                              Adj_s1_pos,
                              Adj_s1_idx,
                              Adj_vals,
                              Out):
        ib = tvm.ir_builder.create()

        SrcFeat_ptr = ib.buffer_ptr(SrcFeat)
        DstFeat_ptr = ib.buffer_ptr(DstFeat)
        Weight_ptr = ib.buffer_ptr(Weight)
        Adj_s1_pos_ptr = ib.buffer_ptr(Adj_s1_pos)
        Adj_s1_idx_ptr = ib.buffer_ptr(Adj_s1_idx)
        Adj_vals_ptr = ib.buffer_ptr(Adj_vals)
        Out_ptr = ib.buffer_ptr(Out)

        ReshapedOut = ib.allocate(Out.dtype, (num_feat_2_partitions, num_dst_vertices, feat_2_len_per_partition), \
            name='ReshapedOut', scope='global')

        Intermediate= ib.allocate(SrcFeat.dtype, (num_dst_vertices, feat_2_len_per_partition), \
            name='Intermediate', scope='global')

        MLPbuffer = ib.allocate(SrcFeat.dtype, (feat_2_len_per_partition,), name='MLPbuffer', scope='local')

        # initialize ReshapedOut
        with ib.for_range(0, num_feat_2_partitions, name='feat_2_outer_idx') as feat_2_outer_idx:
            with ib.for_range(0, num_dst_vertices, name='row_idx') as row_idx:
                with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                    ReshapedOut[feat_2_outer_idx*num_dst_vertices*feat_2_len_per_partition + row_idx*feat_2_len_per_partition + feat_2_inner_idx] = 0.

        with ib.for_range(0, num_feat_2_partitions, name='feat_2_outer_idx') as feat_2_outer_idx:
            with ib.for_range(0, num_src_vertex_partitions, name='src_vertex_partition_idx') as src_vertex_partition_idx:
                # reset Intermediate
                with ib.for_range(0, num_dst_vertices, name='row_idx') as row_idx:
                    with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                        Intermediate[row_idx*feat_2_len_per_partition + feat_2_inner_idx] = 0.
                # handle each 1D partition
                with ib.for_range(0, num_dst_vertices, name='row_idx') as row_idx:
                    row_start = Adj_s1_pos_ptr[src_vertex_partition_idx*d2_size + row_idx]
                    row_end = Adj_s1_pos_ptr[src_vertex_partition_idx*d2_size + row_idx + 1]
                    row_num_elems = row_end - row_start
                    with ib.for_range(0, row_num_elems, name='elem_idx') as elem_idx:
                        # reset MLPbuffer
                        with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                            MLPbuffer[feat_2_inner_idx] = 0.
                        with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                            with ib.for_range(0, feat_1_len, name='feat_1_idx') as feat_1_idx:
                                MLPbuffer[feat_2_inner_idx] += (SrcFeat_ptr[(Adj_s1_idx_ptr[row_start + elem_idx] + src_vertex_partition_idx*num_src_vertices_per_partition)*feat_1_len + feat_1_idx] \
                                                                + DstFeat_ptr[row_idx*feat_1_len + feat_1_idx]) * Weight_ptr[(feat_2_inner_idx + feat_2_outer_idx*feat_2_len_per_partition)*feat_1_len + feat_1_idx]
                            # ReLU
                            MLPbuffer[feat_2_inner_idx] = tvm.max(MLPbuffer[feat_2_inner_idx], 0.)
                        # max aggregation
                        with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                            Intermediate[row_idx*feat_2_len_per_partition + feat_2_inner_idx] = tvm.max(Intermediate[row_idx*feat_2_len_per_partition + feat_2_inner_idx], MLPbuffer[feat_2_inner_idx])
                # merge partitions
                with ib.for_range(0, num_dst_vertices, name='row_idx') as row_idx:
                    with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                        ReshapedOut[feat_2_outer_idx*num_dst_vertices*feat_2_len_per_partition + row_idx*feat_2_len_per_partition + feat_2_inner_idx] = \
                            tvm.max(ReshapedOut[feat_2_outer_idx*num_dst_vertices*feat_2_len_per_partition + row_idx*feat_2_len_per_partition + feat_2_inner_idx], \
                                Intermediate[row_idx*feat_2_len_per_partition + feat_2_inner_idx])

        # copy from ReshapedOut to Out_ptr
        with ib.for_range(0, num_feat_2_partitions, name='feat_2_outer_idx') as feat_2_outer_idx:
            with ib.for_range(0, num_dst_vertices, name='row_idx') as row_idx:
                with ib.for_range(0, feat_2_len_per_partition, name='feat_2_inner_idx') as feat_2_inner_idx:
                    Out_ptr[row_idx*feat_2_len + (feat_2_outer_idx*feat_2_len_per_partition + feat_2_inner_idx)] = \
                        ReshapedOut[feat_2_outer_idx*num_dst_vertices*feat_2_len_per_partition + row_idx*feat_2_len_per_partition + feat_2_inner_idx]

        return ib.get()

    Out = tvm.extern([oshape],
                     [SrcFeat, DstFeat, Weight, Adj_s1_pos, Adj_s1_idx, Adj_vals],
                     lambda ins, outs: _mlp_conv_spmm_x86_ir(
                         ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], outs[0]),
                     dtype=SrcFeat.dtype,
                     name="Out")

    return Out
