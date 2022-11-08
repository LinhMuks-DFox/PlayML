# For generating matrices in latex format
# Author: Mux
# Last Edit: 2022-10-30
import platform
from TexMatrix_user_settings import *
if user_paste_to_clip_board:
    if "Win" in (p := platform.platform()) or "win" in p:
        from tkinter import *

        def copy2clipboard(content: str) -> None:
            r = Tk()
            r.withdraw()
            r.clipboard_clear()
            r.clipboard_append(content)
            r.update()
            r.after(1000, r.destroy)
            r.mainloop()
            print("Copy done!(Os = Win)")
            exit()
    elif "Mac" in p or "mac" in p:
        import subprocess

        def copy2clipboard(output):
            process = subprocess.Popen(
                'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
            process.communicate(output.encode('utf-8'))
            print("Copy done!(os = MacOs)")
    else:
        print("Not supported os. Copy disabled.")
        user_paste_to_clip_board = False
        copy2clipboard = None


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
    "Small": ("(", ")"),
    "Big-No-Right": ("\{", ".")
}

MATRIX_ROW_ELE_DELIMITER = " & "
MATRIX_NEW_ROW = " \\\\ \n\t\t"


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

    if user_paste_to_clip_board:
        copy2clipboard(user_matrix_tex)
    else:
        print(user_matrix_tex)


if __name__ == "__main__":
    main()
