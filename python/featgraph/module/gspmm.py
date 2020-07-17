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
    'copy_lhs' : lambda x,y : x,
    'copy_rhs' : lambda x,y : y,
}
reduce_op_map = {
    'sum' : te.sum,
    'max' : te.max,
    'min' : te.min
}
binary_ops = ['add', 'sub', 'mul', 'div', 'copy_lhs', 'copy_rhs']
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
    dst_feat = te.placeholder((nnz, rhs_len), feature_type, 'dst_feat')
    # placeholder for dense features
    src_feat = te.placeholder((num_cols, lhs_len), feature_type, 'src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    def msgfunc(row, fid):
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = dst_feat[row_start + elem_idx, rhs_off[fid] if use_bcast else fid]
        feat_val = src_feat[adj_indices[row_start + elem_idx], lhs_off[fid] if use_bcast else fid]
        return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
    out = te.compute((num_rows, out_len), msgfunc, name='out')
    # prepare input
    f_input = [adj_indptr, adj_indices]
    if binary_op == 'copy_lhs':
        f_input.append(src_feat)
    elif binary_op == 'copy_rhs':
        f_input.append(dst_feat)
    else:
        f_input += [src_feat, dst_feat]
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
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    if '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    # placeholder for sparse matrix
    adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    dst_feat = te.placeholder((nnz, rhs_len), feat_type, 'dst_feat')
    # placeholder for dense features
    src_feat = te.placeholder((num_cols, lhs_len), feat_type, 'src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    def msgfunc(row, fid):
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = dst_feat[row_start + elem_idx, rhs_off[fid] if use_bcast else fid]
        feat_val = src_feat[adj_indices[row_start + elem_idx], lhs_off[fid] if use_bcast else fid]
        return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
    out = te.compute((num_rows, out_len), msgfunc, name='out')
    # prepare input
    f_input = [adj_indptr, adj_indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, nnz, num_rows, 
        num_cols, out_len, indice_type, feat_type
        ])
    if binary_op == 'copy_lhs':
        f_input.append(src_feat)
    elif binary_op == 'copy_rhs':
        f_input.append(dst_feat)
    else:
        f_input += [src_feat, dst_feat]
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    edge_axis, feat_axis = out.op.axis
    reduce_axis = out.op.reduce_axis[0]
    if target == 'cuda':
        # cuda schedule
        if out_len < 16:
            # use tree reduce if feat_len is small
            ro, ri = s[out].split(reduce_axis, factor=32)
            s[out].bind(ri, te.thread_axis('threadIdx.x'))
            s[out].bind(feat_axis, te.thread_axis('threadIdx.y'))
        else:
            #othrewise just parallel on feature dimension
            s[out].bind(feat_axis, te.thread_axis('threadIdx.x'))
        s[out].bind(edge_axis, te.thread_axis('blockIdx.x'))
    else:
        # llvm schedule
        # pass
        s[out].reorder(edge_axis, reduce_axis, feat_axis)
        # s[out].parallel(edge_axis)
        # s[out].pragma(edge_axis, 'parallel_launch_point')
        # s[out].pragma(edge_axis, 'parallel_stride_pattern', 8)
        # s[out].vectorize(feat_axis)
    # print(tvm.lower(s, f_input))
    return tvm.build(s, f_input, target=target, name=f_name)

def spmm_dds(binary_op, reduce_op, d1_size, d2_size, num_cols,
             nnz, lhs_len, rhs_len, out_len,
             indice_type, feat_type, 
             num_feat_partitions=1, use_bcast=False, target='llvm'):
    # placeholder for dds format
    adj_s1_pos = te.placeholder((d1_size, d2_size), dtype=indice_type, name='adj_s1_pos')
    adj_s1_idx = te.placeholder((nnz,), dtype=indice_type, name='adj_s1_idx')
    adj_vals = te.placeholder((nnz, rhs_len), dtype=feat_type, name='adj_vals')
    num_rows = d2_size - 1
    # num_src_vertices_per_partition = (num_cols + d1_size - 1) // d1_size
    # placeholder for dense features
    src_feat = te.placeholder((num_cols, lhs_len), dtype=feat_type, name='src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    if num_feat_partitions > 1:
        feat_len_per_partition = out_len // num_feat_partitions
        reshaped_src_feat = te.compute((num_feat_partitions, num_cols, feat_len_per_partition),
                                        lambda fo, cid, fi: src_feat[cid, lhs_off[fo*feat_len_per_partition+fi] \
                                        if use_bcast else fo*feat_len_per_partition+fi], name='reshaped_src_feat')
        def msgfunc(fo, src_vertex_partition_idx, row, fi):
            row_start = adj_s1_pos[src_vertex_partition_idx, row]
            row_end = adj_s1_pos[src_vertex_partition_idx, row + 1]
            row_num_elems = row_end - row_start
            elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
            adj_val = adj_vals[row_start + elem_idx, rhs_off[fo*feat_len_per_partition+fi] \
                            if use_bcast else fo*feat_len_per_partition+fi]
            feat_val = reshaped_src_feat[fo, adj_s1_idx[row_start + elem_idx], fi]
            return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
        intermediate = te.compute((num_feat_partitions, d1_size, num_rows, feat_len_per_partition),
                                msgfunc, name='intermediate')
        k = te.reduce_axis((0, d1_size), name='src_vertex_partition_reduce')
        reshaped_out = te.compute((num_feat_partitions, num_rows, feat_len_per_partition), 
                                lambda fo, nn, fi: reduce_op_map[reduce_op](intermediate[fo, k, nn, fi], axis=k),
                                name='reshaped_out')
        out = te.compute((num_rows, out_len), 
                        lambda nn, ff: reshaped_out[ff // feat_len_per_partition, nn, ff % feat_len_per_partition], name='out')
    else:
        def msgfunc(src_vertex_partition_idx, row, fid):
            row_start = adj_s1_pos[src_vertex_partition_idx, row]
            row_end = adj_s1_pos[src_vertex_partition_idx, row + 1]
            row_num_elems = row_end - row_start
            elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
            adj_val = adj_vals[row_start + elem_idx, rhs_off[fid] if use_bcast else fid]
            feat_val = src_feat[adj_s1_idx[row_start + elem_idx], fid]
            return reduce_op_map[reduce_op](binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
        k = te.reduce_axis((0, d1_size), name='src_vertex_partition_reduce')
        intermediate = te.compute((d1_size, num_rows, out_len), msgfunc, name='intermediate')
        out = te.compute((num_rows, out_len), 
            lambda nn, fid: reduce_op_map[reduce_op](intermediate[k, nn, fid], axis=k), name='out')
    # schedule
    s = te.create_schedule([out.op])
    if num_feat_partitions > 1:
        I, RO = intermediate, reshaped_out
        s[I].reorder(I.op.axis[0], I.op.axis[1], I.op.axis[2], I.op.reduce_axis[0], I.op.axis[3])
        s[RO].reorder(RO.op.axis[0], RO.op.reduce_axis[0], RO.op.axis[1], RO.op.axis[2])
        s[I].compute_at(s[RO], RO.op.reduce_axis[0])
        # Parallelize the rows of the sparse matrix
        s[reshaped_src_feat].parallel(reshaped_src_feat.op.axis[1])
        s[I].parallel(I.op.axis[2])
        s[I].vectorize(I.op.axis[3])
        s[RO].parallel(RO.op.axis[1])
        s[out].parallel(out.op.axis[0])
    else:
        I = intermediate
        s[I].reorder(I.op.axis[0], I.op.axis[1], I.op.reduce_axis[0], I.op.axis[2])
        s[out].reorder(out.op.reduce_axis[0], out.op.axis[0], out.op.axis[1])
        s[I].compute_at(s[out], out.op.reduce_axis[0])
        s[I].parallel(I.op.axis[1])
        s[I].vectorize(I.op.axis[2])
        s[out].parallel(out.op.axis[0])

    # prepare input
    f_input = [adj_s1_pos, adj_s1_idx]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, nnz, num_rows, 
        num_cols, d1_size, 'partitioned', out_len, indice_type, feat_type
        ])
    if binary_op == 'copy_lhs':
        f_input.append(src_feat)
    elif binary_op == 'copy_rhs':
        f_input.append(adj_vals)
    else:
        f_input += [src_feat, adj_vals]
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # print(tvm.lower(s, f_input))
    return tvm.build(s, f_input, target=target, name=f_name)
    # else:
    #     def msgfunc(src_vertex_partition_idx, row, fid):
    #         row_start = adj_s1_pos[src_vertex_partition_idx * d2_size + row]
    #         row_end = adj_s1_pos[src_vertex_partition_idx * d2_size + row + 1]
    #         row_num_elems = row_end - row_start
    #         elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
    #         adj_val = adj_vals[row_start + elem_idx]
    #         feat_val = src_feat[adj_s1_idx[row_start + elem_idx], fid]
    #         return te.sum(adj_val * feat_val, axis=elem_idx)


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
    f = spmm('copy_lhs', 'sum', nnz, num_rows, num_cols, feat_len, feat_len, feat_len, indice_type, feat_type, target=target)
    print(f.imported_modules[0].get_source())
