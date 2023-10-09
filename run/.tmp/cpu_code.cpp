#include <torch/extension.h>

torch::Tensor reduce_a__zeros_0_c46_item1_of_a_item2_of_a(torch::Tensor obj_a)
{
    auto a = obj_a.accessor<float, 2>();
torch::Tensor obj_zeros_0 = torch::zeros({20}, at::kFloat);
auto zeros_0 = obj_zeros_0.accessor<float, 1>();
torch::Tensor obj_arr64 = torch::empty({10}, at::kFloat);
auto arr64 = obj_arr64.accessor<float, 1>();
torch::Tensor obj_arr67 = torch::empty({10}, at::kFloat);
auto arr67 = obj_arr67.accessor<float, 1>();
for (int _l16 = 0; _l16 < 10; _l16 += 1) {
arr64[_l16] = zeros_0[_l16];
} 
for (int _l17 = 0; _l17 < 20; _l17 += 1) {
for (int _l18 = 0; _l18 < 10; _l18 += 1) {
arr67[_l18] = (arr64[_l18] + a[_l18][_l17]);
} 
for (int _l19 = 0; _l19 < 10; _l19 += 1) {
arr64[_l19] = arr67[_l19];
} 
} 
return obj_arr64;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &reduce_a__zeros_0_c46_item1_of_a_item2_of_a);
}