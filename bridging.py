import numpy as np


def iterate_ras(b, row_sum, col_sum):

    # scale rows
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.divide(row_sum, b.sum(axis=1))
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = 0
    b = np.dot(np.diag(r), b)
    b[np.isnan(b)] = 0
    b[np.isinf(b)] = 0

    # scale columns
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.divide(col_sum, b.sum(axis=0))
    s[np.isnan(s)] = 0
    s[np.isinf(s)] = 0
    b = np.dot(b, np.diag(s))
    b[np.isnan(b)] = 0
    b[np.isinf(b)] = 0
    return b


def test_iterate_ras():
    mat = np.ones((2, 2))
    row_sum = np.asarray([52., 48.])
    col_sum = np.asarray([87., 13.])
    test = abs(iterate_ras(mat, row_sum, col_sum) - np.asarray([[45.24, 6.76], [41.76, 6.24]])) < 1e-6
    assert test.all()
    mat = np.asarray([[1, 0], [0, 1]])
    row_sum = np.asarray([52, 48])
    col_sum = np.asarray([87, 13])
    test = abs(iterate_ras(mat, row_sum, col_sum) - np.asarray([[87, 0], [0, 13]])) < 1e-6
    assert test.all()
    mat = np.asarray([[0, 0], [1, 1]])
    row_sum = np.asarray([0, 100])
    col_sum = np.asarray([30, 70])
    test = abs(iterate_ras(mat, row_sum, col_sum) - np.asarray([[0, 0], [30, 70]])) < 1e-6
    assert test.all()

