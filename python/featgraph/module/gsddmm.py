import tvm
from tvm import te
import numpy as np
from featgraph.util import calc_bcast

binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y,
    'copy_u' : lambda x,y : x,
    'copy_v' : lambda x,y : y,
}
# binary_ops = ['add', 'sub', 'mul', 'div', 'copy_u', 'copy_v', 'dot']
binary_ops = ['add', 'sub', 'mul', 'div', 'copy_u', 'copy_v']
indice_types = ['int32', 'int64']
feature_types = ['float32', 'float64']

def find_num_threads(l, ntx):
    # return 1024 if l >= 1024 else (
    #             512 if l >= 512 else (
    #                 256 if l >= 256 else (
    #                     128 if l >= 128 else (
    #                         64 if l >= 64 else (
    #                             32 if l >= 32 else (
    #                                 16 if l >= 16 else (
    #                                     8 if l >= 8 else (
    #                                         4 if l >= 4 else (
    #                                             2 if l >= 2 else 1
    #                                         )
    #                                     )
    #                                 )
    #                             )
    #                         )
    #                     )
    #                 )
    #             )
    #         )
    # return te.if_then_else(l >= 1024, 1024, \
    #        te.if_then_else(l >= 512, 512, \
    #        te.if_then_else(l >= 256, 256, \
    #        te.if_then_else(l >= 128, 128, \
    #        te.if_then_else(l >= 64, 64, \
    #        te.if_then_else(l >= 32, 32, \
    #        te.if_then_else(l >= 16, 16, \
    #        te.if_then_else(l >= 8, 8, \
    #        te.if_then_else(l >= 4, 4, \
    #        te.if_then_else(l >= 2, 2, 1))))))))))
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
    f_input = [adj_row_indices, adj_col_indices, src_feat, dst_feat]
    f_name = 'sddmm_{}_{}_{}'.format(binary_op, indice_type, feature_type)
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    if binary_op == 'dot':
        f_input.append(reduce_size)
    f_input.append(out)
    # schedule
    s = te.create_schedule(out.op)
    if target == 'llvm':
        pass
    elif target == 'cuda':
        ntx = find_num_threads(out_len, 1024)
        nty = 1024 // ntx
        edge_axis, feat_axis = out.op.axis[0], out.op.axis[1]
        edge_outer, edge_inner = s[out].split(edge_axis, factor=nty)
        feat_outer, feat_inner = s[out].split(feat_axis, factor=ntx)
        s[out].bind(feat_outer, te.thread_axis('blockIdx.x'))
        s[out].bind(feat_inner, te.thread_axis('threadIdx.x'))
        s[out].bind(edge_outer, te.thread_axis('blockIdx.y'))
        s[out].bind(edge_inner, te.thread_axis('threadIdx.y'))
        # edge_axis, feat_axis = out.op.axis[0], out.op.axis[1]
        # edge_outer, edge_inner = s[out].split(edge_axis, factor=1024)
        # s[out].bind(edge_outer, te.thread_axis('blockIdx.x'))
        # s[out].bind(edge_inner, te.thread_axis('threadIdx.x'))
    return tvm.lower(s, f_input, name=f_name)

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

def evaluate_time(f, adj_scipy_coo, binary_op, src_feat_shape, dst_feat_shape, feat_type, target, num_runs):
    ctx = tvm.cpu(0) if target == 'llvm' else tvm.gpu(0)
    adj_row_indices = adj_scipy_coo.row
    adj_col_indices = adj_scipy_coo.col
    id_type = str(adj_row_indices.dtype)
    tvm_row_indices = tvm.nd.array(adj_row_indices, ctx=ctx)
    tvm_col_indices = tvm.nd.array(adj_col_indices, ctx=ctx)
    use_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = \
        calc_bcast(binary_op, np.zeros((1,)+src_feat_shape), np.zeros((1,)+dst_feat_shape))
    f_name = 'sddmm_{}_{}_{}'.format(binary_op, id_type, feat_type)
    if use_bcast:
        f_name += '_bcast'
    src_feat = np.random.random((adj_scipy_coo.shape[0], lhs_len)).astype(feat_type)
    dst_feat = np.random.random((adj_scipy_coo.shape[1], rhs_len)).astype(feat_type)
    tvm_src_feat = tvm.nd.array(src_feat, ctx=ctx)
    tvm_dst_feat = tvm.nd.array(dst_feat, ctx=ctx)
    f_input = [tvm_row_indices, tvm_col_indices, tvm_src_feat, tvm_dst_feat]
    if use_bcast:
        tvm_lhs_off = tvm.nd.array(lhs_off.astype(id_type), ctx=ctx)
        tvm_rhs_off = tvm.nd.array(rhs_off.astype(id_type), ctx=ctx)
        f_input += [tvm_lhs_off, tvm_rhs_off]
    if binary_op == 'dot':
        f_input.append(reduce_size)
    tvm_out = tvm.nd.array(np.zeros(shape=(adj_row_indices.shape[0], out_len), dtype=feat_type), ctx=ctx)
    f_input.append(tvm_out)
    # first verify result is correct
    f.get_function(f_name)(*f_input)
    lhs = src_feat.reshape((src_feat.shape[0],)+src_feat_shape)[adj_scipy_coo.row]
    rhs = dst_feat.reshape((dst_feat.shape[0],)+dst_feat_shape)[adj_scipy_coo.col]
    if binary_op != 'dot':
        np_result = binary_op_map[binary_op](lhs, rhs)
    else:
        np_result = (lhs * rhs).sum(axis=-1)
    if th_ctx.type == 'cuda':
        np_result = np_result.cpu()
    np.testing.assert_allclose(np_result.reshape(np_result.shape[0], out_len), tvm_out.asnumpy(), rtol=1e-4, atol=1e-4)
    # measure average time
    timer = f.time_evaluator(f_name, ctx=ctx, number=num_runs)
    tcost = timer(*f_input).mean
    print(tcost)
    return tcost


if __name__ == '__main__':
    import scipy
    target = 'cuda'
    # f = build_all(target)
    ir = gsddmm('add', indice_type='int32', feature_type='float32', target=target, use_bcast=False)
    # print(ir)
    f = tvm.build(ir, target=target)
    print(f.imported_modules[0].get_source())
    # adj_scipy_coo = scipy.sparse.random(2**15, 2**15, density=0.001, format='coo').astype('int32')
    # evaluate_time()


