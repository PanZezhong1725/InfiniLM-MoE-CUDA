﻿fn main() {
    if search_cuda_tools::find_nccl().is_some() {
        search_cuda_tools::detect_nccl();
    }
}
