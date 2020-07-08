#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdio>
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

float sum(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    float ret = 0.;
    float *arr_ptr = static_cast<float *>(buf.ptr);
    for (size_t i = 0; i < buf.size; ++i) {
        ret += arr_ptr[i];
    }
    return ret;
} 

void partition(py::array_t<int> py_indptr, 
               py::array_t<int> py_indices,
               py::array_t<int> py_row_indices,
               py::array_t<int> py_col_indices,
               py::array_t<int> py_edge_list,
               int num_rows, int num_cols,
               int num_rows_per_partition,
               int num_cols_per_partition)
{
    int *indptr = static_cast<int *>(py_indptr.request().ptr);
    int *indices = static_cast<int *>(py_indices.request().ptr);
    int *row_indices = static_cast<int *>(py_row_indices.request().ptr);
    int *col_indices = static_cast<int *>(py_col_indices.request().ptr);
    int *edge_list = static_cast<int *>(py_edge_list.request().ptr);
    int num_row_partitions = (num_rows + num_rows_per_partition - 1) / num_rows_per_partition;
    int num_col_partitions = (num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    int **memo = new int*[num_rows_per_partition];
    size_t count = 0;
    for (int i = 0; i < num_row_partitions; i++)
    {
        int row_start = i * num_rows_per_partition;
        int row_end = row_start + num_rows_per_partition;
        // printf("%d %d\n", row_start, row_end);
        row_end = row_end > num_rows ? num_rows : row_end;
        for (int r = row_start; r < row_end; r++)
        {
            memo[r - row_start] = indices + indptr[r];
        }
        for (int j = 1; j <= num_col_partitions; j++)
        {
            int col_end = (j * num_cols_per_partition) < num_cols ? \
                          (j * num_cols_per_partition) : num_cols;
            // printf("%d\n", col_end);
            for (int ri = row_start; ri < row_end; ri++)
            {
                int *l = memo[ri - row_start];
                int *r = indices + indptr[ri+1];
                if (l == r) continue;
                int *pos = std::lower_bound(l, r, col_end);
                // printf("%d %d %d\n", *l, *r, *pos);
                memo[ri - row_start] = pos;
                size_t num_elem = pos - l;
                std::copy(l, pos, col_indices + count);
                for (size_t i = 0; i < num_elem; i++)
                {
                    row_indices[count + i] = ri;
                }
                // edge list
                count += num_elem;
            }
        }
    }
    printf("%ul\n", count);
    delete[] memo;
}

PYBIND11_MODULE(util, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    m.def("sum", &sum, "taking the sum of an array");

    m.def("partition", &partition, "partition a csr matrix");
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
