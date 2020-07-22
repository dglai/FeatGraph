import tvm
from tvm import te
from tvm import autotvm
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
binary_ops = ['add', 'sub', 'mul', 'div', 'copy_lhs', 'copy_rhs', 'dot']
# binary_ops = ['add', 'sub', 'mul', 'div', 'copy_lhs', 'copy_rhs']
indice_types = ['int32', 'int64']
feature_types = ['float32', 'float64']

def find_num_threads(l, ntx):
    if ntx <= 1:
        return 1
    return te.if_then_else(ntx > l, find_num_threads(l, ntx >> 1), ntx)

def gsddmm(binary_op, indice_type='int32', feature_type='float32', use_bcast=False, target='llvm'):
    # sparse matrix variables
    nnz = te.var('nnz', 'int64')
    num_rows = te.var('num_rows', indice_type)
    num_cols = te.var('num_cols', indice_type)
    # feature length variables
    # dtype set to be the same as the other dim in the shape, i.e. nnz, num_rows, num_cols
    # otherwise cannot pass tvm type check 
    lhs_len = te.var('lhs_len', indice_type)
    rhs_len = te.var('rhs_len', indice_type)
    out_len = te.var('out_len', 'int64')
    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')
    # placeholder for dense features
    src_feat = te.placeholder((num_rows, lhs_len), feature_type, 'src_feat')
    dst_feat = te.placeholder((num_cols, rhs_len), feature_type, 'dst_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    if binary_op == 'dot':
        reduce_size = te.var('reduce_size', indice_type)
        k = te.reduce_axis((0, reduce_size), name='k')
        out = te.compute(
            (nnz, out_len),
            lambda eid, fid: te.sum(
                src_feat[adj_row_indices[eid], (lhs_off[fid] if use_bcast else fid) * reduce_size + k] * \
                dst_feat[adj_col_indices[eid], (rhs_off[fid] if use_bcast else fid) * reduce_size + k],
                axis=k
            ),
            name='out'
        )
    else:
        out = te.compute(
            (nnz, out_len), 
            lambda eid, fid: binary_op_map[binary_op](
                src_feat[adj_row_indices[eid], lhs_off[fid] if use_bcast else fid], \
                dst_feat[adj_col_indices[eid], rhs_off[fid] if use_bcast else fid]
            ),
            name='out'
        )
    # prepare input
    f_input = [adj_row_indices, adj_col_indices]
    f_name = 'sddmm_{}_{}_{}'.format(binary_op, indice_type, feature_type)
    if binary_op == 'copy_lhs':
        f_input.append(src_feat)
    elif binary_op == 'copy_rhs':
        f_input.append(dst_feat)
    else:
        f_input += [src_feat, dst_feat]
    if binary_op == 'dot':
        f_input.append(reduce_size)
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    if target == 'llvm':
        pass
    elif target == 'cuda':
        # ntx = find_num_threads(out_len, 1024)
        # nty = 1024 // ntx
        # edge_axis, feat_axis = out.op.axis[0], out.op.axis[1]
        # edge_outer, edge_inner = s[out].split(edge_axis, factor=ntx)
        # feat_outer, feat_inner = s[out].split(feat_axis, factor=nty)
        # s[out].bind(feat_outer, te.thread_axis('blockIdx.x'))
        # s[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
        # s[out].bind(edge_outer, te.thread_axis('blockIdx.y'))
        # s[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
        edge_axis, feat_axis = out.op.axis[0], out.op.axis[1]
        edge_outer, edge_inner = s[out].split(edge_axis, factor=1024)
        s[out].bind(edge_outer, te.thread_axis('blockIdx.x'))
        s[out].bind(edge_inner, te.thread_axis('threadIdx.x'))
    return tvm.lower(s, f_input, name=f_name)

def sddmm(binary_op, nnz, num_rows, num_cols, 
          lhs_len, rhs_len, out_len, indice_type, feat_type,
           reduce_size=1, lhs_target=0, rhs_target=2,
          use_bcast=False, target='llvm'):
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    if '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')
    # placeholder for dense features
    def switch_target(t, l, name):
        if t == 0:
            return te.placeholder((num_rows, l), feat_type, name)
        elif t == 1:
            return te.placeholder((nnz, l), feat_type, name)
        elif t == 2:
            return te.placeholder((num_cols, l), feat_type, name)
    lhs = switch_target(lhs_target, lhs_len, 'lhs')
    rhs = switch_target(rhs_target, rhs_len, 'rhs')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    def idx_target(t, eid):
        if t == 0:
            return adj_row_indices[eid]
        elif t == 1:
            return eid
        elif t == 2:
            return adj_col_indices[eid]
    # compute
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        out = te.compute(
            (nnz, out_len),
            lambda eid, fid: te.sum(
                lhs[idx_target(lhs_target, eid), \
                    (lhs_off[fid] if use_bcast else fid) * reduce_size + k] * \
                rhs[idx_target(rhs_target, eid), \
                    (rhs_off[fid] if use_bcast else fid) * reduce_size + k],
                axis=k
            ),
            name='out'
        )
    else:
        out = te.compute(
            (nnz, out_len), 
            lambda eid, fid: binary_op_map[binary_op](
                lhs[idx_target(lhs_target, eid), \
                    lhs_off[fid] if use_bcast else fid], \
                rhs[idx_target(rhs_target, eid), \
                    rhs_off[fid] if use_bcast else fid]
            ),
            name='out'
        )
    # prepare input
    f_input = []
    if lhs_target == 0 or rhs_target == 0:
        f_input.append(adj_row_indices)
    if lhs_target == 2 or rhs_target == 2:
        f_input.append(adj_col_indices)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, nnz, num_rows, num_cols,
         lhs_len, rhs_len, out_len, indice_type, feat_type
         ])
    if binary_op != 'copy_rhs':
        f_input.append(lhs)
    if binary_op != 'copy_lhs':
        f_input.append(rhs)
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    edge_axis, feat_axis = out.op.axis
    if target == 'cuda':
        # cuda schedule
        if binary_op == 'dot' and reduce_size >= 32:
            # if dot product, use tree reduction
            reduce_axis = out.op.reduce_axis[0]
            eo, ei = s[out].split(edge_axis, factor = (1024 // out_len))
            s[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
            s[out].bind(ei, te.thread_axis('threadIdx.y'))
            s[out].bind(eo, te.thread_axis('blockIdx.x'))
        else:
            ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
            ntx = 1024 if ntx > 1024 else ntx
            nty = 1024 // ntx
            fo, fi = s[out].split(feat_axis, factor=ntx)
            eo, ei = s[out].split(edge_axis, factor=nty)
            nby = (nnz + nty - 1) // nty
            if nby > 65535:
                eo, e = s[out].split(eo, nparts = 65535)
                s[out].reorder(eo, e, ei, fo, fi)
            s[out].bind(fi, te.thread_axis('threadIdx.x'))
            s[out].bind(fo, te.thread_axis('blockIdx.x'))
            s[out].bind(ei, te.thread_axis('threadIdx.y'))
            s[out].bind(eo, te.thread_axis('blockIdx.y'))
    elif target == 'llvm':
        pass
        # TODO only parallel, i.e. without vectorize on feat_axis, will segfault, why?
        # s[out].parallel(edge_axis)
        # s[out].pragma(edge_axis, 'parallel_launch_point')
        # s[out].pragma(edge_axis, 'parallel_stride_pattern', 8)
        # if binary_op != 'dot':
        #     s[out].vectorize(feat_axis)
    # print(tvm.lower(s, f_input))
    return tvm.build(s, f_input, target=target, name=f_name)

def build_all(target, dir = None):
    # build a .so library packing all kinds of kernel for a target(cuda/llvm)
    modules = []
    for op in binary_ops:
        for id_type in indice_types:
            for f_type in feature_types:
                for ub in [True, False]:
                    # print('Building sddmm_{}_{}_{}{}'.format(op, id_type, f_type, '_bcast' if ub else ''))
                    modules.append(gsddmm(op, id_type, f_type, ub, target))
    f = tvm.build(modules, target=target, name='gsddmm_' + target)
    if dir:
        f.export_library(dir + '/libgsddmm_tvm_' + target + '.so')
    return f

# def sddmm_tune(adj_scipy_coo, feat_len, feat_type, binary_op, reduce_size=1, target='llvm'):
#     nnz = adj_scipy_coo.row.shape[0]
#     num_rows = adj_scipy_coo.shape[0]
#     num_cols = adj_scipy_coo.shape[1]
#     indice_type = str(adj_scipy_coo.row.dtype)
#     sddmm = autotvm.task.create('sddmm', args=(binary_op, nnz, num_rows, num_cols, feat_len, indice_type, feat_type, reduce_size), target=target)
#     return sddmm

if __name__ == '__main__':
    target = 'cuda'
    # import dgl
    target = 'cuda'
    # g = dgl.rand_graph(100,30)
    lhs_len, rhs_len = 3, 3
    out_len = 1
    use_bcast = False
    nnz = 30
    num_rows = 100
    num_cols = 100
    indice_type = 'int32'
    feat_type = 'float32'
    f = sddmm('dot', nnz, num_rows, num_cols, lhs_len, rhs_len, out_len, indice_type, feat_type,\
         reduce_size=3, lhs_target=0, rhs_target=0, use_bcast=use_bcast, target=target)
    print(f.imported_modules[0].get_source())




