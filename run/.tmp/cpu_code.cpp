#include <torch/extension.h>

torch::Tensor einsum_a_b_(int d1, int d2, int d4, int d3, torch::Tensor obj_a, torch::Tensor obj_b)
{
    auto a = obj_a.accessor<float, 3>();
auto b = obj_b.accessor<float, 3>();
torch::Tensor obj_arr10 = torch::zeros({d1,d2,d4}, at::kFloat);
auto arr10 = obj_arr10.accessor<float, 3>();
for (int _l0 = 0; _l0 < d1; _l0 += 1) {
for (int _l1 = 0; _l1 < d2; _l1 += 1) {
for (int _l2 = 0; _l2 < d3; _l2 += 1) {
for (int _l3 = 0; _l3 < d4; _l3 += 1) {
arr10[_l0][_l1][_l3] += a[_l0][_l1][_l2] * b[_l0][_l2][_l3];
} 
} 
} 
} 
return obj_arr10;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &einsum_a_b_);
}