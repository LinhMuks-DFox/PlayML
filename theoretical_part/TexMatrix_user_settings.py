# user settings
"""
Usage:

Normal:
    [
        [1, 2],
        [3, 4]
    ]

To create a 10 raw, 7 colum matrix:
    [
        [i for i in range(j * 7, j * 7 + 7)] # colum var
            for j in range(10)               # row var
    ]

To create 0 matrix in 10 row, 11 col:
    [
        [0 for _ in range(j * 11, j * 11 + 11)] # colum var
            for j in range(10)                  # row var
    ]

To create a 10 colum, 7 row algebra Matrix:
    [
        [f"X_{j}^{i}" for i in range(10)]
            for j in range(7)
    ]

Can Also use NumPy (Only in 2 dim!):
    np.zeros(shape=(3, 4), dtype=np.int32)
    np.ones(shape=(4, 5), dtype=np.int32)
    np.linspace(0, 100, 10, shape=(2, 5))
    ...
"""

user_matrix = [
        [f"\\sum_{{i=1}}^m (X_b^{{(i)}} - y^{{(i)}}) \cdot X_{i}^{{(i)}}"] 
        for i in [1, 2, "n"]
]

user_paste_to_clip_board = True     # True / False
# "Big", "Mid", "Small", "Big-No-Right"
user_bracket = "Big"
user_begin_tag = "matrix"           # "matrix" / "align"
