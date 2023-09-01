#include <torch/extension.h>

torch::Tensor add_index_m_idx_t(int n, torch::Tensor obj_m, int idx, torch::Tensor obj_t)
{
    auto m = obj_m.accessor<float, 2>();
auto t = obj_t.accessor<float, 1>();
torch::Tensor obj_arr4 = torch::empty({2}, at::kFloat);
auto arr4 = obj_arr4.accessor<float, 1>();
for (int _l0 = 0; _l0 < 2; _l0 += 1) {
arr4[_l0] = m[idx][_l0] + t[_l0];
} 
return obj_arr4;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &add_index_m_idx_t);
}