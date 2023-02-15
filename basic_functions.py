
import numpy as np
import numba as nb
from collections import deque

DIMENSION = 8

PIECE_VALUES = np.array([100, 290, 324, 502, 976, 10000])
PST = np.array([
        [    0,   0,   0,   0,   0,   0,   0,   0,
            77,  80,  82,  82,  82,  82,  80,  77,
            30,  34,  37,  54,  54,  47,  44,  30,
             3,   4,  11,  16,  16,   9,   4,   3,
             0,  -2,  10,  15,  15,   3,   0,   0,
             2,   2,  -3,  -1,  -1,  -3,   2,   2,
             0,   0,   3, -26, -26,   7,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0],
        [  -50, -40, -30, -30, -30, -30, -40, -50,
           -40, -20,   0,   0,   0,   0, -20, -40,
           -30,   0,  18,  23,  23,  18,   0, -30,
           -30,   4,  23,  30,  30,  23,   4, -30,
           -30,   0,  23,  30,  30,  23,   0, -30,
           -30,   4,  18,  23,  23,  18,   4, -30,
           -40, -20,   0,   5,   5,   0, -20, -40,
           -50, -40, -30, -30, -30, -30, -40, -50],
        [  -20, -10, -10, -10, -10, -10, -10, -20,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -10,   0,   5,   8,   8,   5,   0, -10,
           -10,  15,   5,  10,  10,   5,  15, -10,
           -10,  12,  15,  11,  11,  15,  12, -10,
           -10,  10,  10,   9,   9,  10,  10, -10,
           -10,  10,   0,   0,   0,   0,  10, -10,
           -20, -10, -10, -10, -10, -10, -10, -20],
        [   30,  30,  30,  35,  35,  30,  30,  35,
            25,  30,  40,  40,  45,  40,  30,  30,
             5,  10,  10,  30,  20,  30,  10,   5,
           -20,  -5,  10,  15,  15,  20,  -5, -20,
           -30,  -5,  -1,   0,   5,  -1,  -5, -20,
           -35,   0,   0,   0,   0,   0,   0, -30,
           -30, -10,   4,   6,   6,   4,  -5, -40,
           -10,  -8,   8,  10,  10,   8, -15, -15],
        [  -20, -10, -10,  -5,  -5, -10, -10, -20,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -10,   0,   5,   5,   5,   5,   0, -10,
            -5,   0,   5,   5,   5,   5,   0,  -5,
            -5,   0,   5,   8,   8,   5,   0,  -5,
           -10,   5,   8,   8,   8,   8,   5, -10,
           -10,   0,   8,   0,   0,   0,   0, -10,
           -20, -10, -10,  -5,  -5, -10, -10, -20],
        [  -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -20, -30, -30, -40, -40, -30, -30, -20,
           -10, -20, -20, -40, -40, -20, -20, -10,
            10,  12, -10, -55, -55, -15,  14,  13,
            19,  25,   3, -30,  -5, -20,  27,  22],
])

ENDGAME_PST = np.array([
        [    0,   0,   0,   0,   0,   0,   0,   0,
            90,  94,  98,  98,  98,  98,  94,  90,
            50,  59,  60,  69,  69,  60,  59,  50,
             3,   4,   9,  16,  16,   9,   4,   3,
             0,  -2,   5,  15,  15,   5,   0,   0,
             2,   2,  -2,  -1,  -1,  -2,   2,   2,
            -5,   8,   3, -26, -26,   3,   8,  -5,
             0,   0,   0,   0,   0,   0,   0,   0],
        [  -50, -40, -30, -30, -30, -30, -40, -50,
           -40, -20,   0,   0,   0,   0, -20, -40,
           -30,   0,  20,  25,  25,  20,   0, -30,
           -30,   5,  25,  30,  30,  25,   5, -30,
           -30,   0,  25,  30,  30,  25,   0, -30,
           -30,   5,  20,  25,  25,  20,   5, -30,
           -40, -20,   0,   5,   5,   0, -20, -40,
           -50, -40, -30, -30, -30, -30, -40, -50],
        [  -20, -10, -10, -10, -10, -10, -10, -20,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -10,   0,   5,  10,  10,   5,   0, -10,
           -10,  15,   5,  15,  15,   5,  15, -10,
           -10,   5,  20,  15,  15,  20,   5, -10,
           -10,  15,  15,  10,  10,  15,  15, -10,
           -10,   5,   0,   0,   0,   0,   5, -10,
           -20, -10, -10, -10, -10, -10, -10, -20],
        [   20,  20,  20,  20,  20,  20,  20,  20,
            30,  40,  43,  45,  45,  43,  40,  30,
             4,  18,  23,  25,  25,  23,  18,   4,
            -5,   0,   8,   8,   8,   8,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   5,   5,   5,   5,   0,  -5,
             0,   5,  10,  14,  14,  10,   5,   0],
        [  -20, -10, -10,  -5,  -5, -10, -10, -20,
           -10,   0,   0,   0,   0,   0,   0, -10,
           -10,   0,  20,  20,  20,  20,   0, -10,
            -5,   0,  20,  24,  24,  20,   0,  -5,
            -5,   0,  20,  24,  24,  20,   0,  -5,
           -10,   5,  20,  20,  20,  20,   5, -10,
           -10,   0,   5,   0,   0,   0,   0, -10,
           -20, -10, -10,  -5,  -5, -10, -10, -20],
        [    2,   8,  16,  14,  14,  16,   8,   2,
            14,  16,  20,  26,  26,  20,  16,  14,
            16,  24,  29,  31,  31,  29,  24,  16,
            16,  26,  32,  36,  36,  32,  26,  16,
             8,  24,  30,  33,  33,  30,  24,   8,
             2,   8,  16,  14,  14,  16,   8,   2,
           -18, -14, -10, -10, -10, -10, -14, -18,
           -20, -20, -20, -20, -20, -20, -20, -20],
])

OPP_PST = np.array([
    PST[0][::-1],
    PST[1][::-1],
    PST[2][::-1],
    PST[3][::-1],
    PST[4][::-1],
    PST[5][::-1]
])

OPP_ENDGAME_PST = np.array([
    ENDGAME_PST[0][::-1],
    ENDGAME_PST[1][::-1],
    ENDGAME_PST[2][::-1],
    ENDGAME_PST[3][::-1],
    ENDGAME_PST[4][::-1],
    ENDGAME_PST[5][::-1]
])

A1, H1, A8, H8 = 91, 98, 21, 28

INCREMENTS = np.array([
    [-11,  -9, -10, -20,   0,   0,   0,   0],
    [-21, -19,  -8,  12,  21,  19,   8, -12],
    [-11,  11,   9,  -9,   0,   0,   0,   0],
    [-10,   1,  10,  -1,   0,   0,   0,   0],
    [-11,  11,   9,  -9, -10,   1,  10,  -1],
    [-11, -10,  -9,   1,  11,  10,   9,  -1]
])

ATK_INCREMENTS = np.array([
    [-11,  -9,   0,   0,   0,   0,   0,   0],
    [-21, -19,  -8,  12,  21,  19,   8, -12],
    [-11,  11,   9,  -9,   0,   0,   0,   0],
    [-10,   1,  10,  -1,   0,   0,   0,   0],
    [-11,  11,   9,  -9, -10,   1,  10,  -1],
    [-11, -10,  -9,   1,  11,  10,   9,  -1]
])

OPP_ATK_INCREMENTS = np.array([
    [ 11,   9,   0,   0,   0,   0,   0,   0],
    [-21, -19,  -8,  12,  21,  19,   8, -12],
    [-11,  11,   9,  -9,   0,   0,   0,   0],
    [-10,   1,  10,  -1,   0,   0,   0,   0],
    [-11,  11,   9,  -9, -10,   1,  10,  -1],
    [-11, -10,  -9,   1,  11,  10,   9,  -1]
])


@nb.njit(cache=True)   # with numba nps is increased by ~ 23
def get_pseudo_legal_moves(board, castle, ep):
    moves = []
    for pos, piece in enumerate(board):
        if piece > 5:  # if not own piece
            continue
        for increment in INCREMENTS[piece]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == 30 or occupied < 6:  # standing on own piece or outside of board
                    break
                if piece == 0 and increment in (-10, -20) and board[pos-10] != 20:
                    break
                if piece == 0 and increment == -20 and (pos < 81 or occupied != 20):
                    break
                if piece == 0 and increment in (-11, -9) and (occupied < 6 or occupied > 11)\
                        and new_pos != ep:
                    break
                moves.append((pos, new_pos))
                if piece in (0, 1, 5) or 5 < occupied < 12:  # if is pawn, knight, king or opposing piece
                    break
                if pos == 91 and board[new_pos+1] == 5 and castle[0] == 1:
                    moves.append((new_pos+1, new_pos-1))
                elif pos == 98 and board[new_pos-1] == 5 and castle[1] == 1:
                    moves.append((new_pos-1, new_pos+1))
    return moves


@nb.njit(cache=True)  # with numba nps is increased by ~ 16
def get_pseudo_legal_captures(board):
    moves = []
    for pos, piece in enumerate(board):
        if piece > 5:  # if it's not own piece
            continue
        for increment in ATK_INCREMENTS[piece]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == 30 or occupied < 6:  # if outside of board or own piece
                    break
                if 5 < occupied < 12:
                    moves.append((pos, new_pos))
                    break
                if piece in (0, 1, 5):  # if it is an opposing pawn, knight, or king
                    break
    return moves


@nb.njit(cache=True)  # with numba nps is increased by ~ 19
def get_attacked_squares(board):
    squares = []
    for pos, piece in enumerate(board):
        if piece < 6 or piece > 11:
            continue
        for increment in OPP_ATK_INCREMENTS[piece-6]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == 30 or 5 < occupied < 12:
                    break
                squares.append(new_pos)
                if occupied < 6:
                    break
                if piece in [6, 7, 11] or occupied < 6:
                    break
    return squares


# create a function to get which pieces are attacking which squares. It can include a string of attackers?
# create another function to get which pieces are defending which squares


@nb.njit(cache=True)  # with numba nps is increased by ~ 21
def get_pawn_attack_map(board):
    squares = []
    for pos, piece in enumerate(board):
        if piece != 6:
            continue
        for increment in (11, 9):
            new_pos = pos + increment
            occupied = board[new_pos]
            if occupied == 30:
                continue
            squares.append(new_pos)
    return squares


@nb.njit(cache=True)  # with numba nps is increased by ~ 72
def get_attacked_pieces(board):
    squares = []
    for pos, piece in enumerate(board):
        if piece < 6 or piece > 11:  # if it's not an opposing piece
            continue
        for increment in OPP_ATK_INCREMENTS[piece-6]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == 30 or 5 < occupied < 12:  # if outside of board or opposing piece
                    break
                if occupied < 6:
                    squares.append(new_pos)
                    break
                if piece in [6, 7, 11]:  # if it is an opposing pawn, knight, or king
                    break
    return squares


# in combination, with numba, the above functions give around a 180 nps boost


@nb.njit(cache=True)
def in_check(board, pos):
    for piece in (4, 1):
        for increment in ATK_INCREMENTS[piece]:
            if increment == 0:
                break
            new_pos = pos
            while True:
                new_pos += increment
                occupied = board[new_pos]
                if occupied == 30 or occupied < 6:  # standing on own piece or outside of board
                    break

                if occupied < 12:
                    if piece == occupied - 6:
                        return True

                    if piece == 1:  # if we are checking with knight and opponent piece is not knight
                        break
                    if occupied == 7:  # if we are checking with a queen and opponent piece is a knight
                        break
                    if occupied == 11:  # king
                        if new_pos == pos + increment:
                            return True
                        break

                    if occupied == 6:  # pawn
                        if new_pos == pos - 11 or\
                           new_pos == pos - 9:
                            return True
                        break

                    if occupied == 8:  # bishop
                        if increment in (-11, 11, 9, -9):
                            return True
                        break
                    if occupied == 9:  # rook
                        if increment in (-10, 1, 10, -1):
                            return True
                        break

                if piece == 1:  # if checking with knight
                    break
    return False


@nb.njit(fastmath=True, cache=True)
def get_ordered_pseudo_legal_moves(board, castle, ep, tt_move):
    moves = get_pseudo_legal_moves(board, castle, ep)
    move_scores = np.zeros(len(moves))
    pawn_attack_map = get_pawn_attack_map(board)
    for i, move in enumerate(moves):

        if tt_move[0] == move[0] and tt_move[1] == move[1]:
            move_scores[i] += 100000
            continue

        selected_piece = board[move[0]]
        occupied = board[move[1]]
        if occupied < 12:
            move_scores[i] += 100
            move_scores[i] += 4 * (3 * PIECE_VALUES[occupied - 6] - PIECE_VALUES[selected_piece])
            if move[1] in pawn_attack_map:
                move_scores[i] -= 8 * PIECE_VALUES[selected_piece]
        elif move[1] in pawn_attack_map:
            move_scores[i] += 5 * (PIECE_VALUES[0] - PIECE_VALUES[selected_piece])

        new_correspond_pos = (move[1]-21)//10*8 + (move[1]-21) % 10
        old_correspond_pos = (move[0]-21)//10*8 + (move[1]-21) % 10
        move_scores[i] += 6 * (PST[selected_piece][new_correspond_pos] - PST[selected_piece][old_correspond_pos])
    zipped_moves = zip(move_scores, moves)
    ordered_moves = [move for _, move in sorted(zipped_moves, reverse=True)]

    return ordered_moves


@nb.njit(fastmath=True, cache=True)
def get_ordered_pseudo_legal_captures(board):
    moves = get_pseudo_legal_captures(board)
    move_scores = np.zeros(len(moves))
    pawn_attack_map = get_pawn_attack_map(board)
    for i, move in enumerate(moves):
        selected_piece = board[move[0]]
        occupied = board[move[1]]

        move_scores[i] += 8 * PIECE_VALUES[occupied - 6] - PIECE_VALUES[selected_piece]
        if move[1] in pawn_attack_map:
            move_scores[i] -= 5 * PIECE_VALUES[selected_piece]

    zipped_moves = zip(move_scores, moves)
    ordered_moves = [move for _, move in sorted(zipped_moves, reverse=True)]

    return ordered_moves


def test_for_win(board, move_amt):
    if move_amt > 0:
        return 0
    k_pos = -1
    for pos, piece in enumerate(board):
        if piece == 5:  # if piece is king
            k_pos = pos

    if in_check(board, k_pos):
        return 2
    return 1


@nb.njit(fastmath=True, cache=True)  # without fastmath nps decrease. with fastmath maybe 1 nps boost
def heuristic(board):

    own_mid_score = 0
    opp_mid_score = 0

    own_end_score = 0
    opp_end_score = 0

    own_mid_piece_vals = 0
    opp_mid_piece_vals = 0

    for pos in range(21, 99):
        piece = board[pos]
        correspond_pos = (pos - 21) // 10 * 8 + (pos - 21) % 10
        if piece < 6:
            own_mid_piece_vals += PIECE_VALUES[piece]

            own_mid_score += PST[piece][correspond_pos]
            own_end_score += ENDGAME_PST[piece][correspond_pos]
        elif piece < 12:
            opp_mid_piece_vals += PIECE_VALUES[piece-6]

            opp_mid_score += OPP_PST[piece-6][correspond_pos]
            opp_end_score += OPP_ENDGAME_PST[piece-6][correspond_pos]

    if own_mid_piece_vals + opp_mid_piece_vals > 2600:  # opening or middle game
        own_score = own_mid_score + own_mid_piece_vals
        opp_score = opp_mid_score + opp_mid_piece_vals
    else:  # end game
        own_score = own_end_score + own_mid_piece_vals
        opp_score = opp_end_score + opp_mid_piece_vals

    return own_score-opp_score


@nb.njit(cache=True)  # maybe a 1-5 nps boost with numba
def rotate(board):
    for i in range(21, 99):
        if board[i] < 6:  # own piece
            board[i] += 6
        elif board[i] < 12:  # opponent's piece
            board[i] -= 6
    return np.flip(board)


@nb.njit(cache=True)
def make_move(board, selected_piece, old_pos, new_pos, castle, opp_castle, ep):

    flag = True
    castled_pos = np.array([-1, -1])

    board[new_pos] = selected_piece

    if selected_piece == 0:
        if 21 <= new_pos <= 28:
            board[new_pos] = 4
        elif new_pos == ep:
            board[new_pos + 10] = 20

    if selected_piece == 5:
        if abs(new_pos - old_pos) == 2:
            board[new_pos] = selected_piece
            if new_pos < old_pos:
                castled_pos[0], castled_pos[1] = old_pos, new_pos + 1
                board[old_pos - 1] = board[91]
                board[91] = 20
            else:
                castled_pos[0], castled_pos[1] = old_pos, new_pos - 1
                board[old_pos + 1] = board[98]
                board[98] = 20

    board[old_pos] = 20  # set to empty square

    k_pos = -1
    for pos, piece in enumerate(board):
        if piece == 5:  # if piece is king
            k_pos = pos

    if in_check(board, k_pos):
        flag = False
    elif castled_pos[0] != -1:
        if in_check(board, castled_pos[0]):
            flag = False
        elif in_check(board, castled_pos[1]):
            flag = False

    if flag:
        if selected_piece == 5:
            castle[0], castle[1] = 0, 0

        if old_pos == 91: castle[0] = 0
        if old_pos == 98: castle[1] = 0
        if new_pos == 21: opp_castle[1] = 0
        if new_pos == 28: opp_castle[0] = 0

        if selected_piece == 0 and new_pos - old_pos == -20:
            return 109 - new_pos
        else:
            return -1

    return -10


@nb.njit(cache=True)
def make_capture(board, selected_piece, old_pos, new_pos):

    flag = True

    board[new_pos] = selected_piece
    board[old_pos] = 20

    k_pos = -1
    for pos, piece in enumerate(board):
        if piece == 5:  # if piece is king
            k_pos = pos

    if in_check(board, k_pos):
        flag = False

    if flag:
        return 1

    return 0


@nb.njit(cache=True)
def undo_move(board, selected_piece, old_pos, new_pos, occupied, ep):

    if selected_piece == 0:
        if new_pos == ep:
            board[new_pos + 10] = 6

    if selected_piece == 5:
        if abs(new_pos-old_pos) == 2:
            if new_pos < old_pos:
                board[91] = board[old_pos - 1]
                board[old_pos - 1] = 20
            else:
                board[98] = board[old_pos + 1]
                board[old_pos + 1] = 20
    board[new_pos] = occupied
    board[old_pos] = selected_piece


@nb.njit(cache=True)
def undo_capture(board, selected_piece, old_pos, new_pos, occupied):
    board[new_pos] = occupied
    board[old_pos] = selected_piece


@nb.njit(fastmath=True, cache=True)  # increases nps by like 10-15
def board_hash(board, castle, opp_castle, ep):
    code = nb.uint64(0)
    for pos, piece in enumerate(board):
        if piece > 11:
            continue

        format_square = (pos-21)//10*8 + (pos-21) % 10
        code ^= PIECE_HASH_KEYS[piece][format_square]

    if ep != -1:
        code ^= EP_HASH_KEYS[(ep-21)//10*8 + (ep-21) % 10]

    castle_bits = castle[0] | (castle[1] << 1) | (opp_castle[0] << 2) | (opp_castle[1] << 3)
    code ^= CASTLE_HASH_KEYS[castle_bits]

    return code


def make_readable_board(board):
    new_board = ""
    for j, i in enumerate(board):
        if (j + 1) % 10 == 0:
            new_board += "\n"
        if i == 30:
            new_board += " "
            continue
        if i == 20:
            new_board += "."
            continue
        idx = i
        piece = ""
        if idx > 5:
            idx -= 6
        if idx == 0:
            piece = "p"
        elif idx == 1:
            piece = "n"
        elif idx == 2:
            piece = "b"
        elif idx == 3:
            piece = "r"
        elif idx == 4:
            piece = "q"
        elif idx == 5:
            piece = "k"
        if i < 6:
            piece = piece.upper()
        new_board += piece

    return new_board


PIECE_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=(12, 64), dtype=np.uint64)
EP_HASH_KEYS = np.random.randint(1, 2**64 - 1, size=64, dtype=np.uint64)
CASTLE_HASH_KEYS = np.random.randint(1, 2 ** 64 - 1, size=16, dtype=np.uint64)
SIDE_HASH_KEY = np.random.randint(1, 2 ** 64 - 1, dtype=np.uint64)