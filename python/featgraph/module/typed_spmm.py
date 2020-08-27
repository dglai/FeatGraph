import tvm
from tvm import te

@te.hybrid.script
def _typed_spmm_llvm(adj_indptr, adj_indices, type_mapping, ufeat, type_feat):
    out = output_tensor((num_rows, ufeat.shape[1], type_feat.shape[2]))
    for row in parallel(adj_indptr.shape[0] - 1):
        for elem in range(adj_indptr[row], adj_indptr[row+1]):
            for i in range(ufeat.shape[1]):
                for j in range(ufeat.shape[2]):
                    for k in range(type_feat.shape[2]):
                        out[row, i, k] += ufeat[adj_indices[elem], i, j] * type_feat[type_mapping[elem], j, k]
    return out

@te.hybrid.script
def _typed_spmm_cuda(adj_indptr, adj_indices, type_mapping, ufeat, type_feat):
    out = output_tensor((num_rows, ufeat.shape[1], type_feat.shape[2]))
    for row in bind('blockIdx.x', adj_indptr.shape[0] - 1):
        for elem in range(adj_indptr[row], adj_indptr[row+1]):
            for i in bind('threadIdx.y', ufeat.shape[1]):
                for j in range(ufeat.shape[2]):
                    for k in bind('threadIdx.x', type_feat.shape[2]):
                        out[row, i, k] += ufeat[adj_indices[elem], i, j] * type_feat[type_mapping[elem], j, k]
    return out

def typed_spmm(nnz, num_rows, num_cols, 
               num_types, lhs_shp, rhs_shp,
               indice_type, feat_type, target='llvm'):
    if '32' in indice_type:
        indice_type = 'int32'
    elif '64' in indice_type:
        indice_type = 'int64'
    else:
        raise NotImplementedError
    if '16' in feat_type:
        feat_type = 'float16'
    elif '32' in feat_type:
        feat_type = 'float32'
    elif '64' in feat_type:
        feat_type = 'float64'
    else:
        raise NotImplementedError
    adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    type_feat = te.placeholder((num_types,) + rhs_shp, feat_type, 'type_feat')
    type_mapping = te.placeholder((nnz,), indice_type, 'type_mapping')
    ufeat = te.placeholder((num_cols,) + lhs_shp, feat_type, 'ufeat')
    if target == 'llvm':
        out = _typed_spmm_llvm(adj_indptr, adj_indices, type_mapping, ufeat, type_feat)
    else:
        out = _typed_spmm_cuda(adj_indptr, adj_indices, type_mapping, ufeat, type_feat)
    s = te.create_schedule(out.op)
    ir = tvm.lower(s, [adj_indptr, adj_indices, type_feat, type_mapping, ufeat, out])
    print(ir)
    f = tvm.build(ir, target=target)
    # print(f.imported_modules[0].get_source())