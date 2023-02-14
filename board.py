
import numpy as np
from basic_functions import rotate


def parse_fen(fen_string):
    w_castling_right_comparison = ["K", "Q"]
    b_castling_right_comparison = ["k", "q"]

    fen_list = fen_string.strip().split()
    fen_board = fen_list[0]

    turn = fen_list[1]

    new_board = []
    pos = 21

    for i in range(20):
        new_board.append(30)

    new_board.append(30)
    for i in fen_board:
        if i == "/":
            new_board.append(30)
            new_board.append(30)
        elif i.isdigit():
            for j in range(int(i)):
                new_board.append(20)
                pos += 1
        elif i.isalpha():
            idx = 0
            if i.islower():
                idx = 6
            piece = i.lower()
            if piece == "p":
                idx += 0
            elif piece == "n":
                idx += 1
            elif piece == "b":
                idx += 2
            elif piece == "r":
                idx += 3
            elif piece == "q":
                idx += 4
            elif piece == "k":
                idx += 5
            new_board.append(idx)

    new_board.append(30)
    for i in range(20):
        new_board.append(30)

    w_castle = np.array([0, 0])
    b_castle = np.array([0, 0])

    for i, right in enumerate(w_castling_right_comparison):
        if right in fen_list[2]:
            w_castle[i] = 1
    for i, right in enumerate(b_castling_right_comparison):
        if right in fen_list[2]:
            b_castle[i] = 1

    if len(fen_list[3]) > 1:
        square = [8 - int(fen_list[3][1]), ord(fen_list[3][0]) - 97]  # row, col  b1 == [7, 1]

        if turn == "b":
            square = [7 - square[0], 7 - square[1]]
        square = square[0] * 8 + square[1]

        square = square // 8 * 10 + 21 + square % 8

        ep = square
    else:
        ep = -1

    np_board = np.array(new_board)
    if turn == "b":
        return rotate(np_board), turn, w_castle, b_castle, ep
    return np_board, turn, w_castle, b_castle, ep

