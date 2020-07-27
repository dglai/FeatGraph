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

binary_ops = ['add', 'sub', 'mul', 'div', 'copy_lhs', 'copy_rhs']
reduce_ops = ['sum', 'max', 'min']
indice_types = ['int32', 'int64']
feature_types = ['float32', 'float64']

def spmm(binary_op, reduce_op, nnz, num_rows, num_cols, 
         lhs_len, rhs_len, out_len,
         indice_type, feat_type, use_bcast=False, use_idx=False, target='llvm'):
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
    adj_indptr = te.placeholder((num_rows+1,), indice_type, 'adj_indptr')
    adj_indices = te.placeholder((nnz,), indice_type, 'adj_indices')
    dst_feat = te.placeholder((nnz, rhs_len), feat_type, 'dst_feat')
    edge_mapping = te.placeholder((nnz,), indice_type, 'edge_mapping')
    # placeholder for dense features
    src_feat = te.placeholder((num_cols, lhs_len), feat_type, 'src_feat')
    # placeholder for possible broadcasting offset
    lhs_off = te.placeholder((out_len,), indice_type, 'lhs_off')
    rhs_off = te.placeholder((out_len,), indice_type, 'rhs_off')
    # compute
    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    def msgfunc(row, fid):
        row_start = adj_indptr[row]
        row_end = adj_indptr[row + 1]
        row_num_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_num_elems), name="elem_idx")
        adj_val = dst_feat[edge_mapping[row_start+elem_idx] if use_idx else row_start + elem_idx, \
            rhs_off[fid] if use_bcast else fid]
        feat_val = src_feat[adj_indices[row_start + elem_idx], lhs_off[fid] if use_bcast else fid]
        if reduce_op == 'sum':
            return te.sum(binary_op_map[binary_op](feat_val, adj_val), axis=elem_idx)
        elif reduce_op == 'max':
            if binary_op == 'copy_lhs':
                return argmax((adj_indices[row_start + elem_idx], feat_val), axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return argmax((edge_mapping[row_start+elem_idx] if use_idx else row_start+elem_idx, \
                    adj_val), axis=elem_idx)
            else:
                return argmax((edge_mapping[row_start+elem_idx] if use_idx else row_start+elem_idx, \
                    adj_indices[row_start + elem_idx], \
                    binary_op_map[binary_op](feat_val, adj_val)), axis=elem_idx)
        elif reduce_op == 'min':
            if binary_op == 'copy_lhs':
                return argmin((adj_indices[row_start + elem_idx], feat_val), axis=elem_idx)
            elif binary_op == 'copy_rhs':
                return argmin((edge_mapping[row_start+elem_idx] if use_idx else row_start+elem_idx, \
                    adj_val), axis=elem_idx)
            else:
                return argmin((edge_mapping[row_start+elem_idx] if use_idx else row_start+elem_idx, \
                    adj_indices[row_start + elem_idx], \
                    binary_op_map[binary_op](feat_val, adj_val)), axis=elem_idx)
        else:
            raise NotImplementedError
    if reduce_op == 'sum':
        out = te.compute((num_rows, out_len), msgfunc, name='out')
    else:
        if binary_op == 'copy_lhs':
            argu, out = te.compute((num_rows, out_len), msgfunc, name='out')
        elif binary_op == 'copy_rhs':
            arge, out = te.compute((num_rows, out_len), msgfunc, name='out')
        else:
            arge, argu, out = te.compute((num_rows, out_len), msgfunc, name='out')
    # prepare input
    f_input = [adj_indptr, adj_indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, nnz, num_rows, 
        num_cols, out_len, indice_type, feat_type
        ])
    ops = [out.op]
    if use_u:
        f_input.append(src_feat)
    if use_e:
        f_input.append(dst_feat)
    if use_bcast:
        f_input += [lhs_off, rhs_off]
        f_name += '_bcast'
    if reduce_op != 'sum':
        if use_u:
            f_input.append(argu)
            ops.append(argu.op)
        if use_e:
            f_input.append(arge)
            ops.append(arge.op)
    if use_idx:
        f_input.append(edge_mapping)
        f_name += '_idx'
    f_input.append(out)
    # schedule
    edge_axis, feat_axis = out.op.axis
    reduce_axis = out.op.reduce_axis[0]    
    s = te.create_schedule(ops)
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
        s[out].parallel(edge_axis)
        s[out].pragma(edge_axis, 'parallel_launch_point')
        s[out].pragma(edge_axis, 'parallel_stride_pattern', 8)
        # make compiling really slow when shape is not good
        # s[out].vectorize(feat_axis)
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

if __name__ == '__main__':
    # import dgl
    target = 'llvm'
    # g = dgl.rand_graph(100,30)
    lhs_len, rhs_len = 105, 21
    out_len = int(15 * 49)
    use_bcast = True
    nnz = 3000
    num_rows = 80
    num_cols = 160
    indice_type = 'int32'
    feat_type = 'float32'
    f = spmm('add', 'sum', nnz, num_rows, num_cols, lhs_len, rhs_len, out_len, indice_type, feat_type, use_bcast=use_bcast, target=target)
    # print(f.imported_modules[0].get_source())
