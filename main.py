
import sys
import copy
import threading

from main_searcher import Search
from basic_functions import make_move, rotate, make_readable_board
from board import parse_fen


def parse_go(msg, bot, turn):
    """parse 'go' uci command"""

    d = bot.max_depth
    t = bot.constant_max_time

    _, *params = msg.split()

    for p, v in zip(*2 * (iter(params),)):
        # print(p, v)
        if p == "depth":
            d = int(v)
        elif p == "movetime":
            t = int(v) / 2 / 1000
        elif p == "nodes":
            n = int(v)
        elif p == "wtime":
            if turn == "w":
                t = int(v) / 20 / 1000
        elif p == "btime":
            if turn == "b":
                t = int(v) / 20 / 1000

    bot.constant_max_time = t
    bot.max_depth = d


def main():
    """
    The main input/output loop.
    This implements a slice of the UCI protocol.
    """

    antares = Search()

    results = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    main_board = results[0]
    turn = results[1]
    w_castle = results[2]
    b_castle = results[3]
    ep = results[4]

    compile_thread = threading.Thread(target=compile_stuff, args=(antares, results))
    compile_thread.start()
    compiling = True

    # f = open('/Users/alexandertian/Documents/PycharmProjects/AntaresChess/AntaresV3/debug_file.txt', 'w')

    while True:
        msg = input().strip()
        print(f">>> {msg}", file=sys.stderr)

        # f.write(f">>> {msg}")
        # f.write("\n")

        tokens = msg.split()

        if msg == "quit":
            break

        elif msg == "uci" or msg.startswith("uciok"):
            print("id name AntaresPy")

            # f.write("< id name AntaresPy")
            # f.write("\n")

            print("id author Alexander_Tian")

            # f.write("< id author Alexander_Tian")
            # f.write("\n")

            print("uciok")

            # f.write("< uciok")
            # f.write("\n")

            continue

        elif msg == "isready":
            print("readyok")
            continue

        elif msg == "ucinewgame":
            results = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            main_board = results[0]
            turn = results[1]
            w_castle = results[2]
            b_castle = results[3]
            ep = results[4]

        elif msg.startswith("position"):
            if len(tokens) < 2:
                continue

            if tokens[1] == "startpos":
                results = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

                next_idx = 2

            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                results = parse_fen(fen)
                next_idx = 8

            else:
                continue

            main_board = results[0]
            turn = results[1]
            w_castle = results[2]
            b_castle = results[3]
            ep = results[4]

            if len(tokens) <= next_idx or tokens[next_idx] != "moves":
                continue

            for move in tokens[(next_idx + 1):]:
                formatted_move = antares.uci_to_notation_move(move, turn)
                ep = make_move(main_board, main_board[formatted_move[0]],
                               formatted_move[0], formatted_move[1],
                               w_castle, b_castle, ep)

                main_board = rotate(main_board)
                turn = "w" if turn == "b" else "b"
                # print(make_readable_board(main_board))

        if msg.startswith("go"):
            parse_go(msg, antares, turn)
            if compiling:
                compile_thread.join()
                compiling = False

            results = antares.start_iterative_engine(main_board, turn, w_castle, b_castle, ep, {})
            move = results[0]
            selected_piece = main_board[move[0]]
            new_pos = move[1]
            promotion_flag = False
            if selected_piece == 0:
                if 21 <= new_pos <= 28:
                    promotion_flag = True
            print(f"bestmove {antares.notation_to_uci_move(move, promotion_flag, turn)}")
            continue

    # f.close()
    sys.exit()


def compile_stuff(bot, res):
    test_board = copy.deepcopy(res[0])

    for i in range(1):
        bot.compile_functions(test_board)

    bot.start_fixed_engine(res[0], res[1], res[2], res[3], res[4], {}, 1000000, -1000000, 2)


if __name__ == "__main__":
    main()

