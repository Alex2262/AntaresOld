"""
Antares V3 using int board representation
"""

import timeit
from copy import deepcopy
from basic_functions import *


class Search:

    def __init__(self):
        self.TRANSPOSITION_TABLE = {}
        self.WIN_TABLE = {}

        self.max_depth = 15  # seems this number is multiplied by 2
        self.max_qdepth = 1000

        self.constant_max_time = 5
        self.max_time = 5

        self.min_stop_depth = 2
        self.curr_depth = 0

        self.start_time = 0
        self.node_count = 0

        self.aspiration_window = 65  # in centi pawns

        self.searching = False

    @staticmethod
    def compile_functions(board):
        heuristic(board)
        rotate(board)
        # get_ordered_legal_moves(board, np.array([0, 0]), -1)
        moves = get_ordered_pseudo_legal_moves(board, np.array([0, 0]), -1)
        # get_ordered_legal_captures(board)
        get_ordered_pseudo_legal_captures(board)
        test_for_win(board, len(moves))
        in_check(board, 0)
        make_move(board, board[30], 30, 31, np.array([0, 0]), np.array([0, 0]), 50)
        make_capture(board, board[30], 30, 31)
        undo_move(board, board[30], 30, 31, board[50], 50)
        undo_capture(board, board[30], 30, 31, board[40])

        board_hash(board)

    @staticmethod
    def uci_to_format(uci, turn):
        square = [8 - int(uci[1]), ord(uci[0]) - 97]  # row, col  b1 == [7, 1]
        if turn == "b":
            square = [7 - square[0], 7 - square[1]]

        square = square[0] * 8 + square[1]  # [7, 1] == 7*8 + 1 == 57
        return square

    @staticmethod
    def notation_to_format(notation, turn):  # format is for AntaresPy.py and game_handler.py Notation is for self.
        num_pos = notation - 19 - 2 * ((notation - 10) // 10)
        square = [num_pos // 8, num_pos % 8]
        if turn == "b":
            square = [7 - square[0], 7 - square[1]]
        return square

    @staticmethod
    def format_to_notation(square, turn):
        if turn == "b":
            square = [7 - square[0], 7 - square[1]]
        square = square[0] * 8 + square[1]
        square = square // 8 * 10 + 21 + square % 8
        return square

    def uci_to_notation_move(self, uci, turn):  # example is b1c3 which is Nc3 And [92, 73]
        move = []
        if len(uci) == 5:
            uci = uci[:4]

        square = self.uci_to_format(uci[0] + uci[1], turn)
        move.append(square // 8 * 10 + 21 + square % 8)
        square = self.uci_to_format(uci[2] + uci[3], turn)
        move.append(square // 8 * 10 + 21 + square % 8)
        return move

    def notation_to_uci_move(self, notation, promotion, turn):
        move = []
        square = self.notation_to_format(notation[0], turn)
        move.append(chr(square[1] + 97) + str(8 - square[0]))
        square = self.notation_to_format(notation[1], turn)
        move.append(chr(square[1] + 97) + str(8 - square[0]))
        move = "".join(move)
        if promotion:
            move += "q"
        return move

    def start_iterative_engine(self, board, plr, wc, bc, ep, history):
        new_history = []
        self.max_time = self.constant_max_time
        for h_board in history:
            new_history.append(board_hash(h_board))
        self.node_count = 0
        castle = wc if plr == "w" else bc
        opp_castle = bc if plr == "w" else wc
        returned = self.iterative_deepening(board, plr, castle, opp_castle, ep)
        return returned

    def start_fixed_engine(self, board, plr, wc, bc, ep, history, alpha, beta, depth):
        self.searching = True

        new_history = []
        self.max_time = 1000

        self.start_time = timeit.default_timer()
        for h_board in history:
            new_history.append(board_hash(h_board))

        self.node_count = 0
        castle = wc if plr == "w" else bc
        opp_castle = bc if plr == "w" else wc

        returned = self.negamax(board, alpha, beta, depth, castle, opp_castle, ep)
        move = returned[0]
        score = returned[1] if plr == "w" else -returned[1]

        self.searching = False

        return move, score, self.node_count

    def iterative_deepening(self, board, plr, castle, opp_castle, ep):
        self.start_time = timeit.default_timer()
        node_sum = 0
        best_return = []

        alpha = -1000000
        beta = 1000000

        aspiration_val = 50

        running_depth = 1

        while True:
            if running_depth >= self.max_depth:
                break
            # temp_tt = copy.deepcopy(self.TRANSPOSITION_TABLE)
            self.node_count = 0
            self.curr_depth = running_depth

            returned = self.negamax(board, alpha, beta, self.curr_depth, castle, opp_castle, ep)

            node_sum += self.node_count
            promotion_flag = False

            if returned[0] == [-2, -2]:
                print(f"info depth {running_depth-1} score cp {best_return[1]} "
                      f"time {int((timeit.default_timer() - self.start_time) * 1000)} nodes {self.node_count} "
                      f"nps {int(node_sum / (timeit.default_timer() - self.start_time))} "
                      f"pv {self.notation_to_uci_move(best_return[0], promotion_flag, plr)}")
                return best_return

            if returned[1] <= alpha or returned[1] >= beta:
                # print("RETRY")
                alpha = -1000000
                beta = 1000000
                continue

            alpha = returned[1] - aspiration_val
            beta = returned[1] + aspiration_val

            best_return = deepcopy(returned)

            move = best_return[0]
            selected_piece = board[move[0]]
            new_pos = move[1]

            if selected_piece == 0:
                if 21 <= new_pos <= 28:
                    promotion_flag = True

            print(f"info depth {running_depth} score cp {best_return[1]} "
                    f"time {int((timeit.default_timer() - self.start_time) * 1000)} nodes {self.node_count} "
                    f"nps {int(node_sum / (timeit.default_timer() - self.start_time))} "
                    f"pv {self.notation_to_uci_move(best_return[0], promotion_flag, plr)}")

            running_depth += 1

        return best_return

    def negamax(self, board, alpha, beta, depth, castle, opp_castle, ep):
        hash_code = board_hash(board)

        moves = get_ordered_pseudo_legal_moves(board, castle, ep)

        situation = 1
        in_check_bool = False

        if hash_code in self.WIN_TABLE:  # kind of useless
            test_result = self.WIN_TABLE[hash_code]
            if test_result == 2:
                return [-1, -1], -100000 - depth
            elif test_result == 1:
                return [-1, -1], 0
        else:
            k_pos = 0
            for pos, piece in enumerate(board):
                if piece == 5:  # if piece is king
                    k_pos = pos
            in_check_bool = in_check(board, k_pos)

        if depth == 0:
            return [-1, -1], self.qsearch(board, alpha, beta, self.max_qdepth)

        self.node_count += 1

        if self.curr_depth > self.min_stop_depth and timeit.default_timer() - self.start_time >= self.max_time:
            return [-2, -2], 0

        alpha_org = alpha

        if hash_code in self.TRANSPOSITION_TABLE:  # basic Transposition Table implementation
            entry = self.TRANSPOSITION_TABLE[hash_code]
            if entry[3] >= depth:
                if entry[2] == 0:
                    return entry[0], entry[1]
                elif entry[2] == -1:
                    alpha = max(alpha, entry[1])
                elif entry[3] == 1:
                    beta = min(beta, entry[1])

                if alpha >= beta:
                    return entry[0], entry[1]

            if entry[0] in moves:
                moves.insert(0, moves.pop(moves.index(entry[0])))  # search previously known best move first

        best_move = []
        best_score = -1000000

        for move in moves:
            old_pos, new_pos = move[0], move[1]
            selected_piece = board[old_pos]
            occupied_square = board[new_pos]

            values = make_move(board, selected_piece, old_pos, new_pos, castle, opp_castle, ep)

            if values == -10:
                undo_move(board, selected_piece, old_pos, new_pos, occupied_square, ep)
                continue

            situation = 0
            board = rotate(board)
            returned = self.negamax(board, -beta, -alpha, depth - 1, opp_castle, castle, values)

            board = rotate(board)
            undo_move(board, selected_piece, old_pos, new_pos, occupied_square, ep)
            if returned[0] == [-2, -2]:
                return [-2, -2], 0

            return_eval = -returned[1]
            if return_eval > best_score:
                best_score = return_eval
                best_move = move

                if return_eval > alpha:
                    alpha = return_eval

                    if alpha >= beta:  # alpha-beta cutoff
                        self.TRANSPOSITION_TABLE[hash_code] = [best_move, best_score, -1, depth]
                        return best_move, beta

        if situation == 1 and not in_check_bool:
            self.WIN_TABLE[hash_code] = 1
            return [-1, -1], 0
        if situation == 1 and in_check_bool:
            self.WIN_TABLE[hash_code] = 2
            return [-1, -1], -100000 - depth

        if best_score <= alpha_org:
            flag = 1
        elif best_score >= beta:
            flag = -1
        else:
            flag = 0

        self.TRANSPOSITION_TABLE[hash_code] = [best_move, best_score, flag, depth]

        return best_move, best_score

    def qsearch(self, board, alpha, beta, depth):

        self.node_count += 1

        static_eval = heuristic(board)
        if static_eval >= beta:
            return beta

        alpha = max(alpha, static_eval)

        if depth == 0:
            return heuristic(board)

        moves = get_ordered_pseudo_legal_captures(board)

        for move in moves:
            old_pos, new_pos = move[0], move[1]
            selected_piece = board[old_pos]
            occupied_square = board[new_pos]
            value = make_capture(board, selected_piece, old_pos, new_pos)
            if value == 0:
                undo_capture(board, selected_piece, old_pos, new_pos, occupied_square)
                continue

            board = rotate(board)
            return_eval = -self.qsearch(board, -beta, -alpha, depth - 1)

            board = rotate(board)
            undo_capture(board, selected_piece, old_pos, new_pos, occupied_square)

            if return_eval >= beta:
                return beta

            alpha = max(alpha, return_eval)

        return alpha


'''
info depth 1 score cp 58 time 665 nodes 21 nps 31 pv g1f3
info depth 2 score cp 0 time 668 nodes 60 nps 121 pv g1f3
info depth 3 score cp 58 time 691 nodes 524 nps 874 pv g1f3
info depth 4 score cp 0 time 861 nodes 1366 nps 2287 pv g1f3
info depth 5 score cp 41 time 1227 nodes 8535 nps 8555 pv g1f3
info depth 6 score cp 0 time 2206 nodes 26798 nps 16908 pv g1f3
info depth 7 score cp 30 time 8725 nodes 157618 nps 22338 pv e2e4
info depth 7 score cp 30 time 15000 nodes 163663 nps 23905 pv e2e4
bestmove e2e4
'''