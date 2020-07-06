import tvm
from tvm import te
from tvm import autotvm
import logging
import numpy as np

from featgraph.util import calc_bcast

binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y,
    'copy_u' : lambda x,y : x,
    'copy_e' : lambda x,y : y,
}
reduce_op_map = {
    'sum' : te.sum,
    'max' : te.max,
    'min' : te.min
}
binary_ops = ['add', 'sub', 'mul', 'div', 'copy_u', 'copy_e']
reduce_ops = ['sum', 'max', 'min']
indice_types = ['int32', 'int64']
feature_types = ['float32', 'float64']

def find_num_threads(l, ntx):
    if ntx <= 1:
        return 1
    return te.if_then_else(ntx > l, find_num_threads(l, ntx >> 1), ntx)

def gspmm(binary_op, reduce_op, indice_type='int32', feature_type='float32', use_bcast=False, target='llvm'):
    # sparse matrix variables
    nnz = te.var('nnz', 'int64')
    num_rows = te.var('num_rows', indice_type)
    num_cols = te.var('num_cols', indice_type)
    # feature length variables
    # dtype set to be the same as the other dim in the shape, i.e. nnz, num_rows, num_cols
    # otherwise cannot pass tvm type check 
    lhs_len = te.var('lhs_len', indice_type)
    rhs_len = te.var('rhs_len', 'int64')
    out_len = te.var('out_len', indice_type)
    # placeholder for sparse matrix
    adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    edge_feat = te.placeholder((nnz, rhs_len), feature_type, 'edge_feat')
    # placeholder for dense features
    node_feat = te.placeholder((num_cols, lhs_len), feature_type, 'src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    def msgfunc(row, fid):
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = edge_feat[row_start + elem_idx, rhs_off[fid] if use_bcast else fid]
        feat_val = node_feat[adj_indices[row_start + elem_idx], lhs_off[fid] if use_bcast else fid]
        return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
    out = te.compute((num_rows, out_len), msgfunc, name='out')
    # prepare input
    f_input = [adj_indptr, adj_indices]
    if binary_op == 'copy_u':
        f_input.append(node_feat)
    elif binary_op == 'copy_e':
        f_input.append(edge_feat)
    else:
        f_input += [node_feat, edge_feat]
    f_name = 'spmm_{}_{}_{}_{}'.format(binary_op, reduce_op, indice_type, feature_type)
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    if target == 'llvm':
        pass
    elif target == 'cuda':
        edge_axis, feat_axis = out.op.axis[0], out.op.axis[1]
        feat_outer, feat_inner = s[out].split(edge_axis, factor=out_len)
        s[out].bind(feat_outer, te.thread_axis('blockIdx.x'))
        s[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
    return tvm.lower(s, f_input, name=f_name)

def spmm(binary_op, reduce_op, nnz, num_rows, num_cols, 
         lhs_len, rhs_len, out_len,
         indice_type, feat_type, use_bcast=False, target='llvm'):
    # placeholder for sparse matrix
    adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    edge_feat = te.placeholder((nnz, rhs_len), feat_type, 'edge_feat')
    # placeholder for dense features
    node_feat = te.placeholder((num_cols, lhs_len), feat_type, 'src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    def msgfunc(row, fid):
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = edge_feat[row_start + elem_idx, rhs_off[fid] if use_bcast else fid]
        feat_val = node_feat[adj_indices[row_start + elem_idx], lhs_off[fid] if use_bcast else fid]
        return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
    out = te.compute((num_rows, out_len), msgfunc, name='out')
    # prepare input
    f_input = [adj_indptr, adj_indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, nnz, num_rows, 
        num_cols, out_len, indice_type, feat_type
        ])
    if binary_op == 'copy_u':
        f_input.append(node_feat)
    elif binary_op == 'copy_e':
        f_input.append(edge_feat)
    else:
        f_input += [node_feat, edge_feat]
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    edge_axis, feat_axis = out.op.axis
    if target == 'cuda':
        # cuda schedule
        ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
        ntx = 1024 if ntx > 1024 else ntx
        nty = 1024 // ntx
        fo, fi = s[out].split(feat_axis, factor=ntx)
        eo, ei = s[out].split(edge_axis, factor=nty)
        nby = (nnz + nty - 1) // nty
        if nby > 65535:
            eo, e = s[out].split(eo, nparts = 65535)
            s[out].reorder(e, eo, ei, fo, fi)
        s[out].bind(fi, te.thread_axis('threadIdx.x'))
        s[out].bind(fo, te.thread_axis('blockIdx.x'))
        s[out].bind(ei, te.thread_axis('threadIdx.y'))
        s[out].bind(eo, te.thread_axis('blockIdx.y'))
        # fo, fi = s[out].split(feat_axis, factor=out_len)
        # s[out].bind(fi, te.thread_axis('threadIdx.x'))
        # s[out].bind(fo, te.thread_axis('blockIdx.x'))
    else:
        # llvm schedule
        pass
    return tvm.build(s, f_input, target=target, name=f_name)

def build_all(target, dir = None):
    # build a .so library packing all kinds of kernel for a target(cuda/llvm)
    modules = []
    for b_op in binary_ops:
        for r_op in reduce_ops:
            for id_type in indice_types:
                for f_type in feature_types:
                    for ub in [True, False]:
                        print('Building spmm_{}_{}_{}_{}{}'.format(b_op, r_op, id_type, f_type, '_bcast' if ub else ''))
                        modules.append(gspmm(b_op, r_op, id_type, f_type, ub, target))
    f = tvm.build(modules, target=target, name='gspmm_' + target)
    if dir:
        f.export_library(dir + '/libgspmm_tvm_' + target + '.so')
    return f


if __name__ == '__main__':
    import scipy, logging
    target = 'cuda'
    # f = build_all(target)
    # ir = gspmm('add', 'sum', indice_type='int32', feature_type='float32', target=target, use_bcast=False)
    # print(ir)
    # f = tvm.build(ir, target=target)
    # print(f.imported_modules[0].get_source())
    adj_scipy_csr = scipy.sparse.random(2**10, 2**10, density=0.1, format='csr').astype('int32')
    # evaluate_time()
    nnz = adj_scipy_csr.indices.shape[0]
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]
    indice_type = str(adj_scipy_csr.indices.dtype)
    feat_len = 64
    feat_type = 'float32'
    f = spmm('copy_u', 'sum', nnz, num_rows, num_cols, feat_len, feat_len, feat_len, indice_type, feat_type, target=target)
    print(f.imported_modules[0].get_source())
