import numpy as np

def use_bcast(op, lhs, rhs):
    if op in ['copy_u', 'copy_e']:
        return False
    if lhs.ndim != rhs.ndim:
        return True
    for i in range(lhs.ndim):
        if lhs.shape[i] != rhs.shape[i]:
            return True
    return False

def calc_bcast(op, lhs, rhs):
    # remove first dimension to only consider feature
    lhs_len, rhs_len = 1, 1
    for v in lhs.shape[1:]:
        lhs_len *= v
    for v in rhs.shape[1:]:
        rhs_len *= v
    lhs_ndim, rhs_ndim = lhs.ndim, rhs.ndim
    if_bcast = use_bcast(op, lhs, rhs)
    reduce_size = 1
    lhs_off, rhs_off = [], []
    rst_out_len = 1
    if if_bcast:
        max_ndim = max(lhs_ndim, rhs_ndim)-1
        out_len = 1
        j = 0
        if op == 'dot':
            j += 1
            reduce_size = lhs.shape[lhs_ndim - 1]
        stride_l, stride_r = 1, 1
        lhs_off.append(0)
        rhs_off.append(0)
        while j < max_ndim:
            dl = 1 if lhs_ndim - 1 - j < 1 else lhs.shape[lhs_ndim-1-j]
            dr = 1 if rhs_ndim - 1 - j < 1 else rhs.shape[rhs_ndim-1-j]
            for i in range(1, max(dl, dr)):
                for k in range(out_len):
                    lhs_off.append(lhs_off[k] + i * (i < dl) * stride_l)
                    rhs_off.append(rhs_off[k] + i * (i < dr) * stride_r)
            out_len *= max(dl, dr)
            stride_l *= dl
            stride_r *= dr
            j += 1
        rst_out_len = out_len
    else:
        rst_out_len = rhs_len if op == 'copy_e' else lhs_len
        if op == 'dot':
            reduce_size = lhs.shape[lhs_ndim-1]
            rst_out_len //= reduce_size
    return if_bcast, lhs_len, rhs_len, rst_out_len, reduce_size, np.array(lhs_off).astype('int32'), np.array(rhs_off).astype('int32')

if __name__ == '__main__':
    import numpy as np
    a = np.random.random((1, 1, 3, 3))
    b = np.random.random((1, 4, 1, 3))
    # print(a+b)
    use_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = calc_bcast('add', a, b)
    # print(lhs_off, rhs_off)
    c = [0] * out_len
    x = a[0].flatten()
    y = b[0].flatten()
    for i in range(out_len):
        c[i] = x[lhs_off[i]] + y[rhs_off[i]]
    np.testing.assert_allclose(c, (a[0]+b[0]).flatten(), rtol=1e-4, atol=1e-4)

