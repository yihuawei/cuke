#include <torch/extension.h>

torch::Tensor reduce_a__zeros_0_c3_item1_of_a_item2_of_a(torch::Tensor obj_a)
{
    auto a = obj_a.accessor<float, 2>();
torch::Tensor obj_zeros_0 = torch::zeros({20}, at::kFloat);
auto zeros_0 = obj_zeros_0.accessor<float, 1>();
torch::Tensor obj_arr2 = torch::empty({20}, at::kFloat);
auto arr2 = obj_arr2.accessor<float, 1>();
torch::Tensor obj_arr5 = torch::empty({20}, at::kFloat);
auto arr5 = obj_arr5.accessor<float, 1>();
for (int _l0 = 0; _l0 < 20; _l0 += 1) {
arr2[_l0] = zeros_0[_l0];
} 
for (int _l1 = 0; _l1 < 10; _l1 += 1) {
for (int _l2 = 0; _l2 < 20; _l2 += 1) {
arr5[_l2] = arr2[_l2] + a[_l1][_l2];
} 
for (int _l3 = 0; _l3 < 20; _l3 += 1) {
arr2[_l3] = arr5[_l3];
} 
} 
return obj_arr2;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &reduce_a__zeros_0_c3_item1_of_a_item2_of_a);
}