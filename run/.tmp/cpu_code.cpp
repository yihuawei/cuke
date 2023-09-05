#include <torch/extension.h>

torch::Tensor add_index_index_a_c14_idx_t(torch::Tensor obj_a, torch::Tensor obj_idx, torch::Tensor obj_t)
{
    auto a = obj_a.accessor<float, 2>();
auto idx = obj_idx.accessor<int, 1>();
auto t = obj_t.accessor<float, 1>();
torch::Tensor obj_arr24 = torch::empty({5}, at::kFloat);
auto arr24 = obj_arr24.accessor<float, 1>();
for (int _l6 = 0; _l6 < 5; _l6 += 1) {
arr24[_l6] = a[0][idx[_l6]] + t[_l6];
} 
return obj_arr24;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &add_index_index_a_c14_idx_t);
}