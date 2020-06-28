#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <cstdio>
#include <cassert>

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  assert(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* adj_row;
  DLTensor* adj_col;
  DLTensor* x;
  DLTensor* y;
  DLTensor* output;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[2] = {4, 4};
  int64_t A_shape[1] = {4};
  TVMArrayAlloc(shape, 2, kDLFloat, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape, 2, kDLFloat, dtype_bits, dtype_lanes, device_type, device_id, &y);
  TVMArrayAlloc(A_shape, 1, kDLFloat, dtype_bits, dtype_lanes, device_type, device_id, &output);
  TVMArrayAlloc(A_shape, 1, kDLInt, dtype_bits, dtype_lanes, device_type, device_id, &adj_row);
  TVMArrayAlloc(A_shape, 1, kDLInt, dtype_bits, dtype_lanes, device_type, device_id, &adj_col);

  for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; j++)
      {
        static_cast<float*>(x->data)[i*shape[0] + j] = 1.0;
        static_cast<float*>(y->data)[i*shape[0] + j] = 1.0;
      }
  }
  static_cast<int*>(adj_row->data)[0] = 0;
  static_cast<int*>(adj_row->data)[1] = 0;
  static_cast<int*>(adj_row->data)[2] = 1;
  static_cast<int*>(adj_row->data)[3] = 1;
  static_cast<int*>(adj_col->data)[0] = 2;
  static_cast<int*>(adj_col->data)[1] = 3;
  static_cast<int*>(adj_col->data)[2] = 2;
  static_cast<int*>(adj_col->data)[3] = 3;  
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(x, y, adj_row, adj_col, output);
  // Print out the output
  for (int i = 0; i < A_shape[0]; ++i) {
    printf("%f ", static_cast<float*>(output->data)[i]);
  }
  printf("\n");
}

int main(void) {
  // Normally we can directly
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("./libsddmm.so");
  Verify(mod_dylib, "sddmm");
  return 0;
}
