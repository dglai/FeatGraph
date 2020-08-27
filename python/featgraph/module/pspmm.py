import tvm
from tvm import te
from tvm.tir import IntImm, StringImm

@te.hybrid.script
def _spmm_partitioned_1d(adj_indptr, adj_indices, edge_mapping, ufeat, efeat, binary_op, reduce_op, out_dim1):
    num_rows = adj_indptr.shape[1] - 1
    out = output_tensor((num_rows, out_dim1), ufeat.dtype, name='out')
    arg_u = output_tensor((num_rows, out_dim1), adj_indices.dtype, name='arg_u')
    arg_e = output_tensor((num_rows, out_dim1), adj_indices.dtype, name='arg_e')
    for col_part in range(adj_indptr.shape[0]):
        for row in range(num_rows):
            cid = allocate((1,), adj_indices.dtype, 'local')
            eid = allocate((1,), adj_indices.dtype, 'local')
            val = allocate((1,), ufeat.dtype, 'local')
            for elem in range(adj_indptr[col_part, row], adj_indptr[col_part, row+1]):
                for fid in range(out_dim1):
                    cid[0] = adj_indices[elem]
                    eid[0] = edge_mapping[elem]
                    val[0] = 0.0
                    if binary_op == 'add':
                        val[0] = ufeat[cid[0], fid] + efeat[eid[0], fid]
                    elif binary_op == 'sub':
                        val[0] = ufeat[cid[0], fid] - efeat[eid[0], fid]
                    elif binary_op == 'mul':
                        val[0] = ufeat[cid[0], fid] * efeat[eid[0], fid]
                    elif binary_op == 'div':
                        val[0] = ufeat[cid[0], fid] / efeat[eid[0], fid]
                    elif binary_op == 'copy_lhs':
                        val[0] = ufeat[cid[0], fid]
                    elif binary_op == 'copy_rhs':
                        val[0] = efeat[eid[0], fid]
                    if reduce_op == 'min':
                        if out[row, fid] < val[0]:
                            out[row, fid] = val[0]
                            if binary_op != 'copy_rhs':
                                arg_u[row, fid] = cid[0]
                            if binary_op != 'copy_lhs':
                                arg_e[row, fid] = eid[0]
                    elif reduce_op == 'max':
                        if out[row, fid] > val[0]:
                            out[row, fid] = val[0]
                            if binary_op != 'copy_rhs':
                                arg_u[row, fid] = cid[0]
                            if binary_op != 'copy_lhs':
                                arg_e[row, fid] = eid[0]
                    elif reduce_op == 'sum':
                        out[row, fid] += val[0]
    return out, arg_u, arg_e

def partitioned_spmm(binary_op, reduce_op, nnz, num_rows, num_cols, 
                     indice_type, feat_type, lhs_dim, rhs_dim, out_dim,
                     num_col_partitions, num_feat_partitions=1):
    adj_indptr = te.placeholder((num_col_partitions, num_rows+1), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    edge_mapping = te.placeholder((nnz,), indice_type, 'edge_mapping')
    ufeat = te.placeholder((num_cols,) + lhs_dim, feat_type, 'ufeat')
    efeat = te.placeholder((nnz,) + rhs_dim, feat_type, 'efeat')
    if len(out_dim) == 1:
        rst = _spmm_partitioned_1d(adj_indptr, adj_indices, edge_mapping, ufeat, efeat,
                                   StringImm(binary_op), StringImm(reduce_op), IntImm(indice_type, out_dim[0]))
    else:
        raise NotImplementedError
    out = rst[0]
    s = te.create_schedule(out.op)
    if num_feat_partitions != 1:
        # fused_feat = s[out].fuse(*out.op.axis[3:])
        fo, fi = s[out].split(out.op.axis[3], nparts=num_feat_partitions)
        s[out].reorder(out.op.axis[0], out.op.axis[1], fo, out.op.axis[2], fi)
        s[out].parallel(out.op.axis[1])
    u_buffer = tvm.tir.decl_buffer(ufeat.shape, ufeat.dtype, name='u_buf', buffer_type='auto_broadcast')
    e_buffer = tvm.tir.decl_buffer(efeat.shape, efeat.dtype, name='e_buf', buffer_type='auto_broadcast')
    ir = tvm.lower(s, [adj_indptr, adj_indices, edge_mapping, ufeat, efeat, out], binds={ufeat:u_buffer, efeat:e_buffer})
    print(ir)
    tvm.build(ir, binds={ufeat:u_buffer, efeat:e_buffer})

if __name__ == '__main__':
    partitioned_spmm('mul', 'sum', 10, 6, 6, 'int32', 'float32', (1,), (8,), (8,), 2, 2)