weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
weights = (weights / weights_sum) * routed_scaling_factor