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
indice_types = ['int32', 'int64']
feature_types = ['float32', 'float64']

def sddmm(binary_op, nnz, num_rows, num_cols, 
          lhs_len, rhs_len, out_len, indice_type, feat_type,
           reduce_size=1, lhs_target=0, rhs_target=2,
          use_bcast=False, target='llvm'):
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    else:
        raise NotImplementedError
    if '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    else:
        raise NotImplementedError
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
            # eo, ei = s[out].split(edge_axis, factor = (1024 // reduce_size))
            # s[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
            ro, ri = s[out].split(reduce_axis, factor = 32)
            eo, ei = s[out].split(edge_axis, factor = 32)
            s[out].bind(ri, te.thread_axis('threadIdx.x'))
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




