import numpy as np
def numpy_vectors():
    def print_val(val):
        print('Vector is {} and it\'s shape is {}'.format(val, val.shape))

    a = np.random.randn(1, 5)
    print_val(val=a)

    b = np.random.randn(6) # 6 rows col is undefined
    print_val(val=b)
    assert a.shape == (1,6), "wrong array shape"


def main():
    numpy_vectors()
if __name__== "__main__":
    main()
