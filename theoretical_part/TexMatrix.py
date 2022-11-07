# For generating matrices in latex format
# Author: Mux
# Last Edit: 2022-10-30
from TexMatrix_user_settings import *
# Const
LATEX_MATRIX_FORMAT = \
    r"""\left {BRACKETS_LEFT}
    \begin{{{BEGIN_TAG}}}
		{MATRIX}
    \end{{{BEGIN_TAG}}}
\right {BRACKETS_RIGHT}
"""
BRACKETS_TABLE = {
    "Big": ("\{", "\}"),
    "Mid": ("[", "]"),
    "Small": ("(", ")")
}

MATRIX_ROW_ELE_DELIMITER = " & "
MATRIX_NEW_ROW = " \\\\ \n\t\t"

if user_paste_to_clip_board:
    from tkinter import *
if user_preview:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        rcParams["text.usetex"] = True
    except ImportError:
        print("Fatal: matplotlib did not installed, can not plot latex preview.")
        print("Disable preview or use command : pip install matplotlib")
        print("to install matplotlib")
        user_preview = False
# Main Loop


def main():
    user_matrix_tex = LATEX_MATRIX_FORMAT.format(
        MATRIX=MATRIX_NEW_ROW.join([
            MATRIX_ROW_ELE_DELIMITER.join(str_matrix)
            for str_matrix in [
                [str(digit) for digit in digits_row] for digits_row in user_matrix
            ]
        ]),
        BRACKETS_LEFT=BRACKETS_TABLE[user_bracket][0],
        BEGIN_TAG=user_begin_tag,
        BRACKETS_RIGHT=BRACKETS_TABLE[user_bracket][1]
    )
    try:
        if user_preview:
            plt.text(0.0, 0.0, user_matrix_tex, fontsize=14)
            ax = plt.gca()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.show()
    except RuntimeError as rune:
        print("Maybe latex was not installed.")

    if user_paste_to_clip_board:
        r = Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(user_matrix_tex)
        r.after(500, r.destroy)
        r.mainloop()
        print("After Program closed, matrix will be paste on clip board")
        exit()
    else:
        print(user_matrix_tex)


if __name__ == "__main__":
    main()