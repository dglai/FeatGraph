import numpy as np
import tvm
from topi.util import get_const_tuple
from tvm import te

from ..util import util_partition_adj_coo_2d
from ..op import vanilla_sddmm, schedule_vanilla_sddmm_x86, \
    schedule_vanilla_sddmm_cuda_tree_reduce, schedule_vanilla_sddmm_cuda_single_thread_reduce, \
    multi_head_dot_product_attention_sddmm, schedule_multi_head_dot_product_attention_sddmm_x86


class SDDMMbase():
    """The base class for SDDMM-like computation kernels."""
    def __init__(self, adj_scipy, num_row_partitions=1, num_col_partitions=1):
        """Doing 2D graph partitioning during init.

        Parameters
        ----------
        adj_scipy : scipy.sparse.coo_matrix or scipy.sparse.csr_matrix
            The input scipy sparse matrix

        num_row_partitions : int
            Number of partitions along the row dimension

        num_col_partitions : int
            Number of partitions along the col dimension
        """
        # Use coo format in SDDMM-like kernels
        if adj_scipy.format != 'coo':
            adj_scipy_coo = adj_scipy.tocoo()
        else:
            adj_scipy_coo = adj_scipy
        self._num_rows = adj_scipy_coo.shape[0]
        self._num_cols = adj_scipy_coo.shape[1]
        # assert num_row_partitions >= 1, "num_row_partitions should be larger than or equal to 1"
        # assert num_col_partitions >= 1, "num_col_partitions should be larger than or equal to 1"
        # self._num_row_partitions = num_row_partitions
        # self._num_col_partitions = num_row_partitions
        self._num_row_partitions = 1
        self._num_col_partitions = 1
        # To be updated in self.register
        # TODO uselhs userhs
        self._target = None
        self._ctx = None
        self._compute_func = None
        self._schedule_func = None
        self._register()
        # 2D graph partitioning
        if self._num_row_partitions > 1 or self._num_col_partitions > 1:
            edge_id_list, adj_row_indices, adj_col_indices = self._preprocess_adj(adj_scipy_coo, \
                self._num_row_partitions, self._num_col_partitions)
            # This is smart; credit to Zihao
            self._edge_mapping = np.argsort(edge_id_list)
        else:
            adj_row_indices = adj_scipy_coo.row
            adj_col_indices = adj_scipy_coo.col
        self._adj_row_indices = adj_row_indices
        self._adj_col_indices = adj_col_indices
        self._adj_row_indices_placeholder = te.placeholder(shape=self._adj_row_indices.shape, \
            dtype=str(self._adj_row_indices.dtype), name='adj_row_indices_placeholder')
        self._adj_col_indices_placeholder = te.placeholder(shape=self._adj_col_indices.shape, \
            dtype=str(self._adj_col_indices.dtype), name='adj_col_indices_placeholder')
        self._adj_row_indices_tvm = tvm.nd.array(self._adj_row_indices, ctx=self._ctx)
        self._adj_col_indices_tvm = tvm.nd.array(self._adj_col_indices, ctx=self._ctx)
        # To be updated in self.build
        self._func = None
        # To be updated in self.run
        self.out_tvm = None

    def _preprocess_adj(self, adj_scipy_coo, num_row_partitions=1, num_col_partitions=1):
        return util_partition_adj_coo_2d(adj_scipy_coo, num_row_partitions, num_col_partitions)

    def _register(self):
        raise NotImplementedError("Please register target, ctx, compute_func, and schedule_func.")

    def build(self, input_placeholders, compute_args, schedule_args):
        """Build tvm func, update self._func inplace.

        Parameters
        ----------
        input_placeholders : list of tvm.placeholder
            The required input tvm placeholders other than adj (which has been passed in during self.init)

        compute_args : dict
            Arguments required for compute_func, e.g., num_feat_partitions

        schedule_args : dict
            Arguments required for schedule_func, e.g., num_cuda_blocks
        """
        Adj_row_indices_placeholder = te.placeholder(shape=self._adj_row_indices.shape, \
            dtype=str(self._adj_row_indices.dtype), name='Adj_row_indices_placeholder')
        Adj_col_indices_placeholder = te.placeholder(shape=self._adj_col_indices.shape, \
            dtype=str(self._adj_col_indices.dtype), name='Adj_col_indices_placeholder')
        Out_placeholder = self._compute_func(*input_placeholders, Adj_row_indices_placeholder, \
            Adj_col_indices_placeholder, **compute_args)  # use ** to unpack dict into kwargs
        s = self._schedule_func(Out_placeholder, **schedule_args)
        module_input = [*input_placeholders]
        if compute_args['use_bcast']:
            module_input += [compute_args['lhs_off'], compute_args['rhs_off']]
        module_input += [Adj_row_indices_placeholder, Adj_col_indices_placeholder, Out_placeholder]
        self._func = tvm.build(s, module_input, target=self._target, name='sddmm')
        self._adj_row_indices_tvm = tvm.nd.array(self._adj_row_indices, ctx=self._ctx)
        self._adj_col_indices_tvm = tvm.nd.array(self._adj_col_indices, ctx=self._ctx)
        self.out_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Out_placeholder.shape), dtype=str(Out_placeholder.dtype)), ctx=self._ctx)

    def lower_to_ir(self, input_placeholders, compute_args, schedule_args):
        """Return the IR. This can be useful for debug.

        Parameters
        ----------
        input_placeholders : list of tvm.placeholder
            The required input tvm placeholders other than adj (which has been passed in during self.init)

        compute_args : dict
            Arguments required for compute_func, e.g., num_feat_partitions

        schedule_args : dict
            Arguments required for schedule_func, e.g., num_cuda_blocks
        """
        Adj_row_indices_placeholder = te.placeholder(shape=self._adj_row_indices.shape, \
            dtype=str(self._adj_row_indices.dtype), name='Adj_row_indices_placeholder')
        Adj_col_indices_placeholder = te.placeholder(shape=self._adj_col_indices.shape, \
            dtype=str(self._adj_col_indices.dtype), name='Adj_col_indices_placeholder')
        Out_placeholder = self._compute_func(*input_placeholders, Adj_row_indices_placeholder, \
            Adj_col_indices_placeholder, **compute_args)  # use ** to unpack dict into kwargs
        s = self._schedule_func(Out_placeholder, **schedule_args)
        # s[a].compute_inline()
        # s[b].compute_inline()
        ir = tvm.lower(s, [*input_placeholders, Adj_row_indices_placeholder, \
            Adj_col_indices_placeholder, Out_placeholder], simple_mode=True)
        return ir

    def cuda_source(self):
        assert self._target == 'cuda', "Target is not cuda."
        assert self._func is not None, "Module must be built first"
        return self._func.imported_modules[0].get_source()

    def export_library(self, path):
        assert self._func is not None, "Module must be built first"
        self._func.export_library(path)

    def run(self, input_tvm_ndarrays):
        """Run the built func with the given inputs feat_tvm_ndarrays.

        Parameters
        ----------
        input_tvm_ndarrays : list of tvm.ndarray
            The required input tvm ndarrays other than adj (which has been created during self.build)

        Returns
        -------
        self.out_tvm: tvm.ndarray
            The output tvm ndarray
        """
        self._func(*input_tvm_ndarrays, self._adj_row_indices_tvm, self._adj_col_indices_tvm, self.out_tvm)
        return self.out_tvm

    @property
    def edge_mapping(self):
        assert self._target != 'cuda', "no graph partitioning on cuda, no edge_mapping"
        assert self._num_row_partitions > 1 or self._num_col_partitions > 1, \
            "no graph partitioning, no edge_mapping"
        return self._edge_mapping

    @property
    def ctx(self):
        return self._ctx

    @property
    def target(self):
        return self._target


class VanillaSDDMMx86(SDDMMbase):
    def __init__(self, adj_scipy, num_row_partitions=1, num_col_partitions=1):
        super(VanillaSDDMMx86, self).__init__(adj_scipy, num_row_partitions, num_col_partitions)

    def _register(self):
        self._target = 'llvm'
        self._ctx = tvm.cpu(0)
        self._compute_func = vanilla_sddmm
        self._schedule_func = schedule_vanilla_sddmm_x86


class SDDMMcuda_elemwise(SDDMMbase):
    def __init__(self, adj_scipy):
        super(SDDMMcuda_elemwise, self).__init__(adj_scipy, num_row_partitions=1, num_col_partitions=1)

    def _register(self):
        self._target = 'cuda'
        self._ctx = tvm.gpu(0)
        self._compute_func = vanilla_sddmm
        self._schedule_func = schedule_vanilla_sddmm_cuda_single_thread_reduce

class SDDMMcuda_reduce(SDDMMbase):
    def __init__(self, adj_scipy):
        super(SDDMMcuda_reduce, self).__init__(adj_scipy, num_row_partitions=1, num_col_partitions=1)

    def _register(self):
        self._target = 'cuda'
        self._ctx = tvm.gpu(0)
        self._compute_func = vanilla_sddmm
        self._schedule_func = schedule_vanilla_sddmm_cuda_tree_reduce


class MultiHeadSDDMMx86(SDDMMbase):
    def __init__(self, adj_scipy, num_row_partitions=1, num_col_partitions=1):
        super(MultiHeadSDDMMx86, self).__init__(adj_scipy, num_row_partitions, num_col_partitions)

    def _register(self):
        self._target = 'llvm'
        self._ctx = tvm.cpu(0)
        self._compute_func = multi_head_dot_product_attention_sddmm
        self._schedule_func = schedule_multi_head_dot_product_attention_sddmm_x86


class MultiHeadSDDMMcuda(SDDMMbase):
    def __init__(self, adj_scipy):
        super(MultiHeadSDDMMcuda, self).__init__(adj_scipy, num_row_partitions=1, num_col_partitions=1)

    def _register(self):
        pass
