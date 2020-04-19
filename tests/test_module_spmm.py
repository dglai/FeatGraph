import scipy
import scipy.sparse
import numpy as np
import time
import tvm
from topi.util import get_const_tuple

from featgraph.module import VanillaSpMMx86, VanillaSpMMcuda, MLPConvSpMMx86


def test_vanilla_spmm(adj_scipy_csr, target):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    if target == 'x86':
        # doing 1D graph partitioning during initialization
        num_col_partitions = 4
        vanilla_spmm_module = VanillaSpMMx86(adj_scipy_csr, num_col_partitions)
    elif target == 'cuda':
        module = VanillaSpMMcuda
        vanilla_spmm_module = VanillaSpMMcuda(adj_scipy_csr)

    # tvm func is built for a specific feat_len and num_feat_partitions
    feat_len = 128
    SrcFeat = tvm.placeholder((num_cols, feat_len))
    input_placeholders = [SrcFeat]
    if target == 'x86':
        num_feat_partitions = 4
        compute_args = {'num_feat_partitions': num_feat_partitions}
        schedule_args = {'num_feat_partitions': num_feat_partitions}
    elif target == 'cuda':
        num_cuda_blocks = 4096
        num_threads_per_cuda_block = 64
        compute_args = {}
        schedule_args = {'num_cuda_blocks': num_cuda_blocks,
                         'num_threads_per_cuda_block': num_threads_per_cuda_block}
    vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm]
    out_tvm = vanilla_spmm_module.run(input_tvm_ndarrays).asnumpy()

    # check correctness against scipy
    out_scipy = adj_scipy_csr.dot(src_feat_np)
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


def test_mlp_conv_spmm(adj_scipy_csr, target):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]

    if target == 'x86':
        # doing 1D graph partitioning during initialization
        num_col_partitions = 4
        mlp_conv_spmm_module = MLPConvSpMMx86(adj_scipy_csr, num_col_partitions)
    elif target == 'cuda':
        pass

    # tvm func is built for a specific feat_len
    feat_1_len = 32
    feat_2_len = 128
    SrcFeat = tvm.placeholder((num_cols, feat_1_len))
    DstFeat = tvm.placeholder((num_rows, feat_1_len))
    Weight = tvm.placeholder((feat_2_len, feat_1_len))
    input_placeholders = [SrcFeat, DstFeat, Weight]
    if target == 'x86':
        compute_args = {}
        schedule_args = {}
    elif target == 'cuda':
        pass
    mlp_conv_spmm_module.build(input_placeholders, compute_args, schedule_args)

    # run
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, mlp_conv_spmm_module.ctx)
    dst_feat_np = np.random.random(get_const_tuple(DstFeat.shape)).astype('float32')
    dst_feat_tvm = tvm.nd.array(dst_feat_np, mlp_conv_spmm_module.ctx)
    weight_np = np.random.random(get_const_tuple(Weight.shape)).astype('float32')
    weight_tvm = tvm.nd.array(weight_np, mlp_conv_spmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm, dst_feat_tvm, weight_tvm]
    out_tvm = mlp_conv_spmm_module.run(input_tvm_ndarrays).asnumpy()

    # check correctness against scipy
    out_scipy = np.zeros((num_rows, feat_2_len))
    for row_id in range(num_rows):
        for x in range(adj_scipy_csr.indptr[row_id], adj_scipy_csr.indptr[row_id + 1]):
            col_id = adj_scipy_csr.indices[x]
            mlp = (dst_feat_np[row_id, :] + src_feat_np[col_id, :]).dot(weight_np.T)
            mlp = np.maximum(mlp, 0)  # ReLu
            out_scipy[row_id, :] = np.maximum(out_scipy[row_id, :], mlp)  # max aggregation
    np.testing.assert_allclose(out_scipy, out_tvm, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    adj_scipy_csr = scipy.sparse.random(127, 255, density=0.1, format='csr').astype('float32')
    test_vanilla_spmm(adj_scipy_csr, 'x86')
    test_mlp_conv_spmm(adj_scipy_csr, 'x86')
