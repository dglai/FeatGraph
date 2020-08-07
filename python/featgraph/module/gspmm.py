import tvm
from tvm import te
from tvm import topi

from featgraph.util import calc_bcast

binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y,
    'copy_lhs' : lambda x,y : x,
    'copy_rhs' : lambda x,y : y,
}

def max_combine(x, y):
    if len(x) == 3:
        eid = tvm.tir.Select(x[2] > y[2], x[0], y[0])
        cid = tvm.tir.Select(x[2] > y[2], x[1], y[1])
        val = tvm.tir.Select(x[2] > y[2], x[2], y[2])
        return eid, cid, val
    else:
        idx = tvm.tir.Select(x[1] > y[1], x[0], y[0])
        val = tvm.tir.Select(x[1] > y[1], x[1], y[1])
        return idx, val

def max_identity(t0, t1, t2=None):
    if t2:
        return tvm.tir.const(0, t0), tvm.tir.const(0, t1), tvm.te.min_value(t2)
    else:
        return tvm.tir.const(0, t0), tvm.te.min_value(t1)

def min_combine(x, y):
    if len(x) == 3:
        eid = tvm.tir.Select(x[2] < y[2], x[0], y[0])
        cid = tvm.tir.Select(x[2] < y[2], x[1], y[1])
        val = tvm.tir.Select(x[2] < y[2], x[2], y[2])
        return eid, cid, val
    else:
        idx = tvm.tir.Select(x[1] < y[1], x[0], y[0])
        val = tvm.tir.Select(x[1] < y[1], x[1], y[1])
        return idx, val

def min_identity(t0, t1, t2=None):
    if t2:
        return tvm.tir.const(0, t0), tvm.tir.const(0, t1), tvm.te.max_value(t2)
    else:
        return tvm.tir.const(0, t0), tvm.te.max_value(t1)

argmax = te.comm_reducer(max_combine, max_identity, name='argmax')
argmin = te.comm_reducer(min_combine, min_identity, name='argmin')

reduce_op_map = {
    'max': argmax,
    'min': argmin
}

def _spmm(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
          ufeat, efeat, edge_id):
    def msgfunc(*args):
        row = args[0]
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        elem_idx = te.reduce_axis((0, row_end-row_start), name="elem_idx")
        u_val = ufeat.__getitem__((adj_indices[row_start + elem_idx],) + args[1:])
        e_val = efeat.__getitem__((edge_id(row_start + elem_idx),) + args[1:])
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](u_val, e_val), axis=elem_idx)
        else:
            if binary_op == 'copy_lhs':
                return reduce_op_map[reduce_op]((adj_indices[row_start + elem_idx], u_val), axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), e_val), axis=elem_idx)
            else:
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), adj_indices[row_start + elem_idx], \
                            binary_op_map[binary_op](u_val, e_val)), axis=elem_idx)
    return te.compute(out_shp, msgfunc, name='out')

def _spmm_feat(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
          ufeat, efeat, edge_id, num_feat_partitions):
    num_cols = ufeat.shape[0]
    nnz = efeat.shape[0]
    bcast_ufeat = topi.broadcast_to(ufeat, (num_cols,) + out_shp[1:])
    bcast_efeat = topi.broadcast_to(efeat, (num_cols,) + out_shp[1:])
    feat_len = 1
    for d in out_shp[1:]:
        feat_len *= d
    flatten_ufeat = topi.reshape(bcast_ufeat, (num_cols, feat_len))
    flatten_efeat = topi.reshape(bcast_efeat, (nnz, feat_len))
    feat_len_per_partition = feat_len // num_feat_partitions
    reshaped_ufeat = te.compute((num_feat_partitions, num_cols, feat_len_per_partition), \
                            lambda fo, idx, fi: flatten_ufeat[idx, fo * feat_len_per_partition + fi],
                            name='reshaped_ufeat')
    reshaped_efeat = te.compute((num_feat_partitions, nnz, feat_len_per_partition), \
                            lambda fo, idx, fi: flatten_efeat[idx, fo * feat_len_per_partition + fi],
                            name='reshaped_efeat')
    def msgfunc(*args):
        row = args[0]
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        elem_idx = te.reduce_axis((0, row_end-row_start), name="elem_idx")
        fid = topi.util.ravel_index(args[1:], out_shp[1:])
        u_val = reshaped_ufeat[fid // feat_len_per_partition, adj_indices[row_start + elem_idx], \
                                  fid % feat_len_per_partition]
        e_val = reshaped_efeat[fid // feat_len_per_partition, edge_id(row_start + elem_idx), \
                                  fid % feat_len_per_partition]
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](u_val, e_val), axis=elem_idx)
        else:
            if binary_op == 'copy_lhs':
                return reduce_op_map[reduce_op]((adj_indices[row_start + elem_idx], u_val), axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), e_val), axis=elem_idx)
            else:
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), adj_indices[row_start + elem_idx], \
                            binary_op_map[binary_op](u_val, e_val)), axis=elem_idx)
    rst = te.compute(out_shp, msgfunc, name='out')
    return rst, [bcast_ufeat, bcast_efeat, flatten_ufeat, flatten_efeat], [reshaped_ufeat, reshaped_efeat]
    
def _spmm_dds(out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
              ufeat, efeat, edge_id, num_feat_partitions):
    num_col_partitions = adj_indptr[0]
    def msgfunc(*args):
        col_part_idx = args[0]
        row = args[1]
        row_start = adj_indptr[col_part_idx, row]
        row_end = adj_indptr[col_part_idx, row + 1]
        elem_idx = te.reduce_axis((0, row_end-row_start), name="elem_idx")
        u_val = ufeat.__getitem__((adj_indices[row_start + elem_idx],) + args[1:])
        e_val = efeat.__getitem__((edge_id(row_start + elem_idx),) + args[1:])
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](u_val, e_val), axis=elem_idx)
        else:
            if binary_op == 'copy_lhs':
                return reduce_op_map[reduce_op]((adj_indices[row_start + elem_idx], u_val), axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), e_val), axis=elem_idx)
            else:
                return reduce_op_map[reduce_op]((edge_id(row_start + elem_idx), adj_indices[row_start + elem_idx], \
                            binary_op_map[binary_op](u_val, e_val)), axis=elem_idx)
    if reduce_op == 'sum':
        intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc, name='out')
    else:
        if binary_op == 'copy_lhs':
            argu, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc, name='out')
        elif binary_op == 'copy_rhs':
            arge, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc, name='out')
        else:
            arge, argu, intermediate = te.compute((num_col_partitions,) + out_shp, msgfunc, name='out')
    k = te.reduce_axis((0, num_col_partitions), name='k')
    if reduce_op == 'sum':
        rst =  te.compute(out_shp, lambda *args: te.sum(intermediate.__getitem__((k,) + args), axis=k), name='out')
    else:
        if binary_op == 'copy_lhs':
            rst =  te.compute(out_shp, 
                lambda *args: reduce_op_map[reduce_op](
                    (argu.__getitem__((k,) + args), intermediate.__getitem__((k,) + args)), axis=k), 
                    name='out')
        if binary_op == 'copy_rhs':
            rst =  te.compute(out_shp, 
                lambda *args: reduce_op_map[reduce_op](
                    (arge.__getitem__((k,) + args), intermediate.__getitem__((k,) + args)), axis=k), 
                    name='out')
        else:
            rst =  te.compute(out_shp,
                lambda *args: reduce_op_map[reduce_op](arge.__getitem__((k,) + args)
                    (argu.__getitem__((k,) + args), intermediate.__getitem__((k,) + args)), axis=k), 
                    name='out')
    return intermediate, rst
    
def _spmm_cuda_general(s, out):
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    s[out].bind(edge_axis, te.thread_axis('blockIdx.x'))
    s[out].bind(feat_axis, te.thread_axis('threadIdx.x'))

def _spmm_cuda_tree_reduce(s, out):
    reduce_axis = out.op.reduce_axis[0]
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    ro, ri = s[out].split(reduce_axis, factor=32)
    s[out].bind(ri, te.thread_axis('threadIdx.x'))
    s[out].bind(feat_axis, te.thread_axis('threadIdx.y'))
    s[out].bind(edge_axis, te.thread_axis('blockIdx.x'))

def _spmm_cpu(s, out):
    reduce_axis = out.op.reduce_axis[0]
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    s[out].reorder(edge_axis, reduce_axis, feat_axis)
    s[out].parallel(edge_axis)

def _spmm_feat(s, out, inlines, reshapes, num_feat_partitions):
    for t in inlines:
        s[t].compute_inline()
    for t in reshapes:
        s[t].parallel(t.op.axis[1])
    reduce_axis = out.op.reduce_axis[0]
    edge_axis = out.op.axis[0]
    feat_axis = s[out].fuse(*out.op.axis[1:])
    fo, fi = s[out].split(feat_axis, nparts = num_feat_partitions)
    s[out].reorder(fo, edge_axis, reduce_axis, fi)
    s[out].parallel(edge_axis)

def _spmm_dds_sched(s, out, I):
    ifeat_axis = s[I].fuse(*I.op.axis[2:])
    s[I].reorder(I.op.axis[0], I.op.axis[1], I.op.reduce_axis[0], ifeat_axis)
    ofeat_axis = s[out].fuse(out.op.axis[1:])
    s[out].reorder(out.op.reduce_axis[0], out.op.axis[0], ofeat_axis)
    s[I].compute_at(s[out], out.op.reduce_axis[0])
    s[I].parallel(I.op.axis[1])
    s[I].vectorize(I.op.axis[2])
    s[out].parallel(out.op.axis[0])
    

def spmm(binary_op, reduce_op, nnz, num_rows, num_cols, 
         lhs_shp, rhs_shp, out_shp,
         indice_type, feat_type, use_idx=False,
         num_col_partitions=1, num_feat_partitions=1, target='llvm'):
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
    if num_col_partitions > 1:
        adj_indptr = te.placeholder((num_col_partitions, num_rows+1), indice_type, 'adj_indptr')
    else:
        adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    efeat = te.placeholder((nnz,) + rhs_shp, feat_type, 'efeat')
    edge_mapping = te.placeholder((nnz,), indice_type, 'edge_mapping')
    # placeholder for dense features
    ufeat = te.placeholder((num_cols,) + lhs_shp, feat_type, 'ufeat')
    # compute
    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    def edge_id(x):
        return edge_mapping[x] if use_idx else x
    if num_feat_partitions == 1:
        if num_col_partitions == 1:
            rst = _spmm((num_rows,) + out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
                            ufeat, efeat, edge_id)
        else:
            intermediate, rst = _spmm_dds((num_rows,) + out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
                ufeat, efeat, edge_id)
    else:
        if num_col_partitions == 1:
            rst, inlines, reshapes = _spmm_feat((num_rows,) + out_shp, binary_op, reduce_op, adj_indptr, adj_indices,
                ufeat, efeat, edge_id, num_feat_partitions)
        else:
            raise NotImplementedError
    if reduce_op == 'sum':
        out = rst
    else:
        out = rst[-1]
    # prepare input
    f_input = [adj_indptr, adj_indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, nnz, num_rows, 
        num_cols, indice_type, feat_type
        ])
    if use_u:
        f_input.append(ufeat)
    if use_e:
        f_input.append(efeat)
    if reduce_op != 'sum':
        if use_u and use_e:
            f_input += [rst[1], rst[0]]
        else:
            f_input.append(rst[0])
    if use_idx:
        f_input.append(edge_mapping)
        f_name += '_idx'
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    if target == 'cuda':
        # cuda schedule
        if topi.util.get_const_int(topi.util.prod(out.shape[1:])) < 16:
            # use tree reduce if feat_len is small
            _spmm_cuda_tree_reduce(s, out)
        else:
            #othrewise just parallel on feature dimension
            _spmm_cuda_general(s, out)
    else:
        # llvm schedule
        if num_col_partitions == 1:
            if num_feat_partitions > 1:
                _spmm_feat(s, out, inlines, reshapes, num_feat_partitions)
            else:
                _spmm_cpu(s, out)
        else:
            _spmm_dds(s, out, intermediate)
    # bind autobroadcast buffer
    u_buffer = tvm.tir.decl_buffer(ufeat.shape, ufeat.dtype, name='u_buf', buffer_type='auto_broadcast')
    e_buffer = tvm.tir.decl_buffer(efeat.shape, efeat.dtype, name='e_buf', buffer_type='auto_broadcast')
    # print(tvm.lower(s, f_input), binds={ufeat:u_buffer, efeat: e_buffer})
    return tvm.build(s, f_input, target=target, name=f_name, binds={ufeat:u_buffer, efeat: e_buffer})

if __name__ == '__main__':
    target = 'cuda'
    lhs_shp, rhs_shp = (8,), (8,)
    out_shp = (8,)
    nnz = 5
    num_rows = 10
    num_cols = 10
    indice_type = 'int32'
    feat_type = 'float32'
    f = spmm('add', 'sum', nnz, num_rows, num_cols, lhs_shp, rhs_shp, out_shp, indice_type, feat_type, target=target, num_feat_partitions=1)
    print(f.imported_modules[0].get_source())
