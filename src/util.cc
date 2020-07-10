#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdio>
namespace py = pybind11;

void partition_2d(py::array_t<int> py_indptr, 
                  py::array_t<int> py_indices,
                  py::array_t<int> py_edge_id,
                  py::array_t<int> py_row_indices,
                  py::array_t<int> py_col_indices,
                  py::array_t<int> py_par_edge_id,
                  int num_rows, int num_cols,
                  int num_rows_per_partition,
                  int num_cols_per_partition)
{
    // input
    int *indptr = static_cast<int *>(py_indptr.request().ptr);
    int *indices = static_cast<int *>(py_indices.request().ptr);
    int *edge_id = static_cast<int *>(py_edge_id.request().ptr);
    //output
    int *row_indices = static_cast<int *>(py_row_indices.request().ptr);
    int *col_indices = static_cast<int *>(py_col_indices.request().ptr);
    int *par_edge_id = static_cast<int *>(py_par_edge_id.request().ptr);
    // calculate number of partitions on rows and columns
    int num_row_partitions = (num_rows + num_rows_per_partition - 1) / num_rows_per_partition;
    int num_col_partitions = (num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    // cache for position in last iteration for each row in a row partition
    int **memo = new int*[num_rows_per_partition];
    // running sum of copied edges
    size_t count = 0;
    for (int i = 0; i < num_row_partitions; i++)
    {
        // for each row parition
        int row_start = i * num_rows_per_partition;
        int row_end = row_start + num_rows_per_partition;
        row_end = row_end > num_rows ? num_rows : row_end;
        // initialize memo
        for (int r = row_start; r < row_end; r++)
        {
            memo[r - row_start] = indices + indptr[r];
        }
        // for each col partition
        for (int j = 1; j <= num_col_partitions; j++)
        {
            int col_end = (j * num_cols_per_partition) < num_cols ? \
                          (j * num_cols_per_partition) : num_cols;
            // for each row in the patition
            for (int ri = row_start; ri < row_end; ri++)
            {
                // binary search the col_end
                // assume the indices are sorted
                int *l = memo[ri - row_start];
                int *r = indices + indptr[ri+1];
                if (l == r) continue;
                int *pos = std::lower_bound(l, r, col_end);
                // update memo
                memo[ri - row_start] = pos;
                // update count, copy data
                size_t num_elem = pos - l;
                std::copy(l, pos, col_indices + count);
                std::copy(edge_id+(l-indices), edge_id+(pos-indices), par_edge_id+count);
                for (size_t i = 0; i < num_elem; i++)
                {
                    row_indices[count + i] = ri;
                }
                count += num_elem;
            }
        }
    }
    // printf("%ul\n", count);
    delete[] memo;
}

void partition_1d(py::array_t<int> py_indptr, 
                  py::array_t<int> py_indices,
                  py::array_t<int> py_edge_id,
                  py::array_t<int> py_dense_indptr,
                  py::array_t<int> py_par_indices,
                  py::array_t<int> py_par_edge_id,
                  int num_rows, int num_cols,
                  int num_cols_per_partition)
{
    // input
    int *indptr = static_cast<int *>(py_indptr.request().ptr);
    int *indices = static_cast<int *>(py_indices.request().ptr);
    int *edge_id = static_cast<int *>(py_edge_id.request().ptr);
    // output
    int *dense_indptr = static_cast<int *>(py_dense_indptr.request().ptr);
    int *par_indices = static_cast<int *>(py_par_indices.request().ptr);
    int *par_edge_id = static_cast<int *>(py_par_edge_id.request().ptr);
    // calculate number of partitions on rows and columns
    int num_col_partitions = (num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    // cache for position in last iteration for each row in a row partition
    int *memo = new int[num_rows];
    // running sum of copied edges
    size_t count = 0;
    // initialize memo
    std::copy(indptr, indptr+num_rows, memo);
    // for each row
    // for each col partition
    for (int j = 1; j <= num_col_partitions; j++)
    {
        int col_end = (j * num_cols_per_partition) < num_cols ? \
                        (j * num_cols_per_partition) : num_cols;
        for (int i = 0; i < num_rows; i++)
        {
            // binary search the col_end
            // assume the indices are sorted
            int l = memo[i];
            int r = indptr[i+1];
            dense_indptr[(j-1)*(num_rows+1)+i] = count;
            if (l == r) continue;
            int *pos = std::lower_bound(indices+l, indices+r, col_end);
            // update memo
            memo[i] = pos - indices;
            // update count, copy data
            size_t num_elem = pos - (indices+l);
            std::copy(indices+l, pos, par_indices + count);
            std::copy(edge_id+l, edge_id+(pos-indices), par_edge_id+count);
            count += num_elem;
        }
        dense_indptr[j*(num_rows+1) - 1] = count;
    }
    // printf("%ul\n", count);
    delete[] memo;
}

PYBIND11_MODULE(util, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           partition
    )pbdoc";

    m.def("partition_2d", &partition_2d, "partition a csr matrix in both row and column to COO format");

    m.def("partition_1d", &partition_1d, "partition a csr matrix in column to dds format");
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
