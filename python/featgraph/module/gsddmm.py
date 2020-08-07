import tvm
from tvm import te
from tvm import topi

binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y
}

def _sddmm_compute(out_shp, binary_op, lhs, rhs, 
                  lhs_idx, rhs_idx, 
                  num_feat_partitions=1):
    reduce_size = lhs.shape[-1]
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        out = te.compute(
            out_shp,
            lambda *args: te.sum(
                lhs.__getitem__((lhs_idx(args[0]),) + args[1:-1] +(k,)) * \
                rhs.__getitem__((rhs_idx(args[0]),) + args[1:-1] +(k,)),
                axis=k
            ),
            name='out'
        )
    else:
        out = te.compute(out_shp, 
                lambda *args: binary_op_map[binary_op](
                    lhs.__getitem__((lhs_idx(args[0]),) + args[1:]), 
                    rhs.__getitem__((rhs_idx(args[0]),) + args[1:])
                ),
                name='out')
    return out

def _sddmm_compute_feat_partition(out_shp, binary_op, lhs, rhs, 
                                  lhs_idx, rhs_idx, 
                                  num_feat_partitions=1):
    reduce_size = lhs.shape[-1] if binary_op == 'dot' else 1
    if binary_op != 'dot':
        feat_shp = out_shp[1:]
    else:
        if out_shp[1:] != (1,):
            feat_shp = out_shp[1:] + (reduce_size,)
        else:
            feat_shp = (reduce_size,)
    bcast_lhs = topi.broadcast_to(lhs, (lhs.shape[0],) + feat_shp)
    bcast_rhs = topi.broadcast_to(rhs, (rhs.shape[0],) + feat_shp)
    feat_len = 1
    for d in out_shp[1:]:
        feat_len *= d
    feat_len *= reduce_size
    flatten_lhs = topi.reshape(bcast_lhs, (lhs.shape[0], feat_len))
    flatten_rhs = topi.reshape(bcast_rhs, (rhs.shape[0], feat_len))
    # assume feat_len is a multiply of num_feat_partitions
    feat_len_per_partition = feat_len // num_feat_partitions
    reshaped_lhs = te.compute((num_feat_partitions, lhs.shape[0], feat_len_per_partition), \
                               lambda fo, idx, fi: flatten_lhs[idx, fo * feat_len_per_partition + fi],
                               name='reshaped_lhs')
    reshaped_rhs = te.compute((num_feat_partitions, rhs.shape[0], feat_len_per_partition), \
                               lambda fo, idx, fi: flatten_rhs[idx, fo * feat_len_per_partition + fi],
                               name='reshaped_rhs')
    if binary_op == 'dot':
        k = te.reduce_axis((0, reduce_size), name='k')
        def dot_edge_func(*indice):
            eid = indice[0]
            fid = topi.util.ravel_index(indice[1:], out_shp[1:])
            fid *= reduce_size
            return te.sum(
                reshaped_lhs[(fid + k) // feat_len_per_partition, \
                            lhs_idx(eid), (fid + k) % feat_len_per_partition] * \
                reshaped_rhs[(fid + k) // feat_len_per_partition, \
                            rhs_idx(eid), (fid + k) % feat_len_per_partition],
                axis=k
            )
        out = te.compute(out_shp, dot_edge_func, name='out')
    else:
        def edge_func(*indice):
            eid = indice[0]
            fid = topi.util.ravel_index(indice[1:], out_shp[1:])
            return binary_op_map[binary_op](
                reshaped_lhs[fid // feat_len_per_partition, \
                             lhs_idx(eid), fid % feat_len_per_partition], \
                reshaped_rhs[fid // feat_len_per_partition, \
                             rhs_idx(eid), fid % feat_len_per_partition]
            )
        out = te.compute(out_shp, edge_func, name='out')
    return out, feat_len_per_partition, [reshaped_lhs, reshaped_rhs], [bcast_lhs, bcast_rhs, flatten_lhs, flatten_rhs]

def _sddmm_cuda_general(s, out):
    out_len = topi.util.get_const_int(topi.util.prod(out.shape[1:]))
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    ntx = 1024 if ntx > 1024 else ntx
    nty = 1024 // ntx
    fo, fi = s[out].split(feat_axis, factor=ntx)
    eo, ei = s[out].split(edge_axis, factor=nty)
    s[out].bind(fi, te.thread_axis('threadIdx.x'))
    s[out].bind(fo, te.thread_axis('blockIdx.y'))
    s[out].bind(ei, te.thread_axis('threadIdx.y'))
    s[out].bind(eo, te.thread_axis('blockIdx.x'))

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
    # s[out].pragma(edge_axis, 'parallel_launch_point')
    # s[out].pragma(edge_axis, 'parallel_stride_pattern', 8)
    
def _sddmm_cpu_feat_partition(s, out, reshaped, inline, reduce_size, feat_len_per_partition):
    for t in inline:
        s[t].compute_inline()
    for t in reshaped:
        edge_axis = t.op.axis[1]
        s[t].parallel(edge_axis)
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    if op != 'dot':
        fo, fi = s[out].split(feat_axis, factor=feat_len_per_partition)
        s[out].reorder(fo, edge_axis, fi)
    else:
        reduce_axis = out.op.reduce_axis[0]
        if reduce_size == feat_len_per_partition:
            s[out].reorder(feat_axis, edge_axis, reduce_axis)
        elif reduce_size < feat_len_per_partition:
            fo, fi = s[out].split(feat_axis, factor=feat_len_per_partition // reduce_size)
            s[out].reorder(fo, edge_axis, fi, reduce_axis)
        else:
            ro, ri = s[out].split(reduce_axis, factor=feat_len_per_partition)
            s[out].reorder(ro, feat_axis, edge_axis, ri)
    s[out].parallel(edge_axis)

def sddmm(binary_op, nnz, num_rows, num_cols, 
          lhs_shp, rhs_shp, out_shp, indice_type, feat_type,
          lhs_target=0, rhs_target=2,
          target='llvm', num_feat_partitions=1):
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
    # check should be done in infer_out_shape
    if binary_op == 'dot':
        reduce_size = lhs_shp[-1]
    else:
        reduce_size = 1
    # placeholder for sparse matrix
    adj_row_indices = te.placeholder((nnz,), indice_type, 'adj_row_indices')
    adj_col_indices = te.placeholder((nnz,), indice_type, 'adj_col_indices')
    # placeholder for dense features
    def switch_target(t, s, name):
        if t == 0:
            return te.placeholder((num_rows,) + s, feat_type, name)
        elif t == 1:
            return te.placeholder((nnz,) + s, feat_type, name)
        elif t == 2:
            return te.placeholder((num_cols,) + s, feat_type, name)
    lhs = switch_target(lhs_target, lhs_shp, 'lhs')
    rhs = switch_target(rhs_target, rhs_shp, 'rhs')
    # idx wrapper for corresponding target
    def idx_target(t):
        def foo(eid):
            if t == 0:
                return adj_row_indices[eid]
            elif t == 1:
                return eid
            elif t == 2:
                return adj_col_indices[eid]
        return foo
    # compute
    if num_feat_partitions == 1:
        out = _sddmm_compute((nnz,) + out_shp, binary_op, lhs, rhs, \
            idx_target(lhs_target), idx_target(rhs_target))
    else:
        out, feat_len_per_partition, reshaped, inline = _sddmm_compute_feat_partition((nnz,) + out_shp, binary_op, lhs, rhs, \
            idx_target(lhs_target), idx_target(rhs_target), num_feat_partitions)
    # prepare input
    f_input = []
    if lhs_target == 0 or rhs_target == 0:
        f_input.append(adj_row_indices)
    if lhs_target == 2 or rhs_target == 2:
        f_input.append(adj_col_indices)
    f_name = '_'.join(str(x) for x in [
        'sddmm', binary_op, nnz, num_rows, num_cols,
         lhs_target, rhs_target,
         indice_type, feat_type
         ])
    f_input += [lhs, rhs, out]
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
            _sddmm_cpu_feat_partition(s, out, reshaped, inline, reduce_size, feat_len_per_partition)
    # bind autobroadcast buffer
    lhs_buffer = tvm.tir.decl_buffer(lhs.shape, lhs.dtype, name='lhs_buf', buffer_type='auto_broadcast')
    rhs_buffer = tvm.tir.decl_buffer(rhs.shape, rhs.dtype, name='rhs_buf', buffer_type='auto_broadcast')
    # print(tvm.lower(s, f_input, binds={lhs:lhs_buffer, rhs:rhs_buffer}))
    return tvm.build(s, f_input, target=target, name=f_name, binds={lhs:lhs_buffer, rhs:rhs_buffer})

if __name__ == '__main__':
    print('hello')
    import numpy as np
    import dgl
    import dgl.backend as F
    target = 'llvm'
    lhs_shp = (1024,)
    rhs_shp = (1024,)
    out_shp = (1024,)
    nnz = 3000
    num_rows = 100
    num_cols = 100
    g = dgl.rand_graph(num_rows, nnz).astype(F.int32)
    gidx = g._graph
    row, col, _ = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_coo_dlpack(0))
    indice_type = 'int32'
    feat_type = 'float32'
    op = 'add'
    f1 = sddmm(op, nnz, num_rows, num_cols, lhs_shp, rhs_shp, out_shp, indice_type, feat_type,\
                lhs_target=0, rhs_target=2, target='cuda', num_feat_partitions=1)
    # print(f1.imported_modules[0].get_source())