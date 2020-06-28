import scipy
import scipy.sparse
import numpy as np
import time
import os
import tvm
from tvm import te
from topi.util import get_const_tuple

from featgraph.module import VanillaSDDMMx86, SDDMMcuda_elemwise, SDDMMcuda_reduce, MultiHeadSDDMMx86, MultiHeadSDDMMcuda
from featgraph.util import calc_bcast


def test_vanilla_sddmm(adj_scipy_coo, binary_op, src_feat_shape, dst_feat_shape, target):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    # doing 2D graph partitioning during initialization
    # note that 2D partitioning is mainly useful for CPU since it optimizes cache
    if target == 'x86':
        num_row_partitions = 1
        num_col_partitions = 1
        vanilla_sddmm_module = VanillaSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
    elif target == 'cuda':
        vanilla_sddmm_module = SDDMMcuda_reduce(adj_scipy_coo)

    use_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = calc_bcast(binary_op, np.zeros((1,)+src_feat_shape), np.zeros((1,)+dst_feat_shape))
    print(lhs_len, rhs_len, lhs_off, rhs_off)
    # tvm func is built for a specific feat_len and num_feat_partitions
    # feat_len = 4
    SrcFeat = te.placeholder((num_rows, lhs_len), dtype='float32', name='SrcFeat')
    DstFeat = te.placeholder((num_cols, rhs_len), dtype='float32', name='DstFeat')
    input_placeholders = [SrcFeat, DstFeat]
    if target == 'x86':
        num_feat_partitions = 1
        compute_args = {'num_feat_partitions': num_feat_partitions, 'binary_op':binary_op, 
                        'out_len': out_len, 'reduce_size': reduce_size, 
                        'use_bcast': use_bcast}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 128
        num_threads_per_cuda_block = 64
        compute_args = {'binary_op':binary_op, 
                        'out_len': out_len, 'reduce_size': reduce_size, 
                        'use_bcast': use_bcast}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks}
    if use_bcast:
        compute_args['lhs_off'] = te.placeholder((out_len,), dtype='int32', name='lhs_off')
        compute_args['rhs_off'] = te.placeholder((out_len,), dtype='int32', name='rhs_off')
    vanilla_sddmm_module.build(input_placeholders, compute_args, schedule_args)
    # print(vanilla_sddmm_module.lower_to_ir(input_placeholders, compute_args, schedule_args))
    print(vanilla_sddmm_module.cuda_source())
    # vanilla_sddmm_module.export_library(os.path.dirname(os.path.abspath(os.path.expanduser(__file__))) + '/libsddmm.so')
    # print(os.path.dirname(os.path.abspath(os.path.expanduser(__file__))) + '/libsddmm.so')

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, vanilla_sddmm_module.ctx)
    lhs_off_tvm = tvm.nd.array(lhs_off, ctx=vanilla_sddmm_module._ctx)
    rhs_off_tvm = tvm.nd.array(rhs_off, ctx=vanilla_sddmm_module._ctx)
    feat_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm, lhs_off_tvm, rhs_off_tvm]
    out_tvm = vanilla_sddmm_module.run(feat_tvm_ndarrays).asnumpy()
    print(out_tvm.shape)
    # print('warming up')
    # for i in range(10):
    #     vanilla_sddmm_module.run(feat_tvm_ndarrays).asnumpy()
    # print("finished warmup")
    # start = time.time()
    # for i in range(100):
    #     vanilla_sddmm_module.run(feat_tvm_ndarrays).asnumpy()
    # print('Elapsed time: {} seconds'.format((time.time() - start) / 100))
    # be careful here
    # if target == 'x86':
    #     out_tvm = out_tvm[vanilla_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np.reshape((num_rows,)+src_feat_shape)[adj_scipy_coo.row]
    rhs = dst_feat_np.reshape((num_cols,)+dst_feat_shape)[adj_scipy_coo.col]
    out_scipy = (lhs * rhs).sum(axis=-1)
    out_scipy = out_scipy.reshape(out_scipy.shape[0], out_len)
    # for i in range(100):
    #     for j in range(100):
    #         if abs(out_scipy[i][j] - out_tvm[i][j]) > 0.01:
    #             print(i, j, out_scipy[i][j], out_tvm[i][j])
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-2, atol=1e-2)


def test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, target):
    num_rows = adj_scipy_coo.shape[0]
    num_cols = adj_scipy_coo.shape[1]

    # doing 2D graph partitioning during initialization
    # note that 2D partitioning is mainly useful for CPU since it optimizes cache
    if target == 'x86':
        num_row_partitions = 4
        num_col_partitions = 4
        multi_head_sddmm_module = MultiHeadSDDMMx86(adj_scipy_coo, num_row_partitions, num_col_partitions)
    elif target == 'cuda':
        multi_head_sddmm_module = MultiHeadSDDMMcuda(adj_scipy_coo)

    # tvm func is built for a specific num_heads, num_head_partitions, feat_len, num_feat_partitions
    num_heads = 16
    feat_len = 64
    SrcFeat = te.placeholder((num_rows, num_heads, feat_len))
    DstFeat = te.placeholder((num_cols, num_heads, feat_len))
    input_placeholders = [SrcFeat, DstFeat]
    if target == 'x86':
        num_head_partitions = 2
        num_feat_partitions = 8
        compute_args = {'num_head_partitions': num_head_partitions,
                        'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_head_partitions': num_head_partitions,
                         'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 4096
        num_threads_per_cuda_block = 64
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks,
                         'num_threads_per_cuda_block': num_threads_per_cuda_block}
    multi_head_sddmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, multi_head_sddmm_module.ctx)
    dst_feat_tvm = tvm.nd.array(dst_feat_np, multi_head_sddmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm]
    out_tvm = multi_head_sddmm_module.run(input_tvm_ndarrays).asnumpy()
    # be careful here
    if target == 'x86':
        out_tvm = out_tvm[multi_head_sddmm_module.edge_mapping]

    # check correctness against scipy
    lhs = src_feat_np[adj_scipy_coo.col]
    rhs = dst_feat_np[adj_scipy_coo.row]
    out_scipy = (lhs * rhs).sum(axis=-1)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    adj_scipy_coo = scipy.sparse.random(128, 128, density=0.1, format='coo').astype('int32')
    # import dgl
    # from dgl.data import RedditDataset
    # data = RedditDataset()
    # data._load()
    # adj_scipy_coo = data.graph.adjacency_matrix_scipy(fmt='coo')
    test_vanilla_sddmm(adj_scipy_coo, 'dot', (1,32), (8, 32), 'cuda')
    # test_multi_head_dot_product_attention_sddmm(adj_scipy_coo, 'x86')
