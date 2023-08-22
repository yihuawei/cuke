#include <torch/extension.h>

torch::Tensor add_index_A_c7_index_B_c11(torch::Tensor obj_A, torch::Tensor obj_B)
{
    int s2;
s2 = 10 - 1;
int s3;
s3 = s2 / 1;
auto A = obj_A.accessor<float, 2>();
auto B = obj_B.accessor<float, 2>();
torch::Tensor obj_arr4 = torch::empty({s3,20}, at::kFloat);
auto arr4 = obj_arr4.accessor<float, 2>();
for (int _l0 = 0; _l0 < s3; _l0 += 1) {
for (int _l1 = 0; _l1 < 20; _l1 += 1) {
arr4[_l0][_l1] = A[<core.ir.Slice object at 0x7f7a4151b100>][_l0][_l1] + B[<core.ir.Slice object at 0x7f7a4151b280>][_l0][_l1];
} 
} 
return obj_arr4;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &add_index_A_c7_index_B_c11);
}