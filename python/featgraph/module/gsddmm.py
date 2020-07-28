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

def _sddmm_compute(out_shp, binary_op, lhs, rhs, 
                  lhs_idx, rhs_idx, feat_bcast, 
                  reduce_size=1, num_feat_partitions=1):
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        out = te.compute(
            out_shp,
            lambda eid, fid: te.sum(
                lhs[lhs_idx(eid), feat_bcast(fid, True) * reduce_size + k] * \
                rhs[rhs_idx(eid), feat_bcast(fid, False) * reduce_size + k],
                axis=k
            ),
            name='out'
        )
    else:
        out = te.compute(
            out_shp, 
            lambda eid, fid: binary_op_map[binary_op](
                lhs[lhs_idx(eid), feat_bcast(fid, True)], \
                rhs[rhs_idx(eid), feat_bcast(fid, False)]
            ),
            name='out'
        )
    return out

def _sddmm_compute_feat_partition(out_shp, binary_op, lhs, rhs, 
                                  lhs_idx, rhs_idx, feat_bcast, 
                                  reduce_size=1, num_feat_partitions=1):
    # assume out_len is a multiply of num_feat_partitions
    feat_len_per_partition = out_shp[1] // num_feat_partitions
    reshaped_lhs = te.compute((num_feat_partitions, lhs.shape[0], feat_len_per_partition * reduce_size), \
                               lambda fo, idx, fi: lhs[idx, feat_bcast(fo*feat_len_per_partition+(fi // reduce_size), True) \
                                                       * reduce_size + (fi % reduce_size)], name='reshaped_lhs')
    reshaped_rhs = te.compute((num_feat_partitions, rhs.shape[0], feat_len_per_partition * reduce_size), \
                               lambda fo, idx, fi: rhs[idx, feat_bcast(fo*feat_len_per_partition+(fi // reduce_size), False) \
                                                       * reduce_size + (fi % reduce_size)], name='reshaped_rhs')
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        out = te.compute(
            out_shp,
            lambda eid, fid: te.sum(
                reshaped_lhs[fid // feat_len_per_partition, \
                             lhs_idx(eid), (fid % feat_len_per_partition) * reduce_size + k] * \
                reshaped_rhs[fid // feat_len_per_partition, \
                             rhs_idx(eid), (fid % feat_len_per_partition) * reduce_size + k],
                axis=k
            ),
            name='out'
        )
    else:
        out = te.compute(
            out_shp, 
            lambda eid, fid: binary_op_map[binary_op](
                reshaped_lhs[fid // feat_len_per_partition, \
                             lhs_idx(eid), fid % feat_len_per_partition], \
                reshaped_rhs[fid // feat_len_per_partition, \
                             rhs_idx(eid), fid % feat_len_per_partition]
            ),
            name='out'
        )
    return out

def _sddmm_cuda_general(s, out):
    out_len = out.shape[1]
    edge_axis, feat_axis = out.op.axis
    nnz = out.shape[0]
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

def _sddmm_cuda_tree_reduce(s, out):
    edge_axis = out.op.axis[0]
    reduce_axis = out.op.reduce_axis[0]
    # eo, ei = s[out].split(edge_axis, factor = (1024 // reduce_size))
    # s[out].bind(reduce_axis, te.thread_axis('threadIdx.x'))
    ro, ri = s[out].split(reduce_axis, factor = 32)
    eo, ei = s[out].split(edge_axis, factor = 32)
    s[out].bind(ri, te.thread_axis('threadIdx.x'))
    s[out].bind(ei, te.thread_axis('threadIdx.y'))
    s[out].bind(eo, te.thread_axis('blockIdx.x'))

def _sddmm_cpu_general(s, out):
    edge_axis = out.op.axis[0]
    s[out].parallel(edge_axis)
    s[out].pragma(edge_axis, 'parallel_launch_point')
    s[out].pragma(edge_axis, 'parallel_stride_pattern', 8)
    
def _sddmm_cpu_feat_partition(s, out, num_feat_paritions):
    pass


def sddmm(binary_op, nnz, num_rows, num_cols, 
          lhs_len, rhs_len, out_len, indice_type, feat_type,
           reduce_size=1, lhs_target=0, rhs_target=2,
          use_bcast=False, target='llvm', num_feat_partitions=1):
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
    def idx_target(t):
        def foo(eid):
            if t == 0:
                return adj_row_indices[eid]
            elif t == 1:
                return eid
            elif t == 2:
                return adj_col_indices[eid]
        return foo
    def feat_bcast(fid, left):
        if not use_bcast:
            return fid
        elif left:
            return lhs_off[fid]
        else:
            return rhs_off[fid]
    # compute
    if num_feat_partitions == 1:
        out = _sddmm_compute((nnz, out_len), binary_op, lhs, rhs, \
                        idx_target(lhs_target), idx_target(rhs_target), feat_bcast, \
                        reduce_size)
    else:
        out = _sddmm_compute_feat_partition((nnz, out_len), binary_op, lhs, rhs, \
                        idx_target(lhs_target), idx_target(rhs_target), feat_bcast, \
                        reduce_size, num_feat_partitions)
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
    if target == 'cuda':
        # cuda schedule
        if binary_op == 'dot' and reduce_size >= 32:
            # if dot product, use tree reduction
            _sddmm_cuda_tree_reduce(s, out)
        else:
            _sddmm_cuda_general(s, out)
    elif target == 'llvm':
        if num_feat_partitions == 1:
            _sddmm_cpu_general(s, out)
        else:
            _sddmm_cpu_feat_partition(s, out, num_feat_partitions)
    print(tvm.lower(s, f_input))
    return tvm.build(s, f_input, target=target, name=f_name)

if __name__ == '__main__':
    target = 'llvm'
    lhs_len, rhs_len = 8, 16
    out_len = 2
    use_bcast = True
    nnz = 5
    num_rows = 10
    num_cols = 10
    indice_type = 'int32'
    feat_type = 'float32'
    f = sddmm('dot', nnz, num_rows, num_cols, lhs_len, rhs_len, out_len, indice_type, feat_type,\
         reduce_size=8, lhs_target=0, rhs_target=2, use_bcast=use_bcast, target=target, num_feat_partitions=2)
    # print(f.imported_modules[0].get_source())




