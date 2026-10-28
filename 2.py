# Tic-Tac-Toe game agent, the agent should play optimally using the minimax
# algorithm. Allow a human user to play against the AI agent. Display the result
# of the game (Win / Lose / Draw).

def show():
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])

def winner(p):
    wins = [(0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)]
    return any(all(board[i] == p for i in w) for w in wins)

def minimax(ai):
    if winner("O"): return 1
    if winner("X"): return -1
    if " " not in board: return 0

    scores = []
    for i in range(9):
        if board[i] == " ":
            board[i] = ai
            scores.append(minimax("X" if ai == "O" else "O"))
            board[i] = " "

    return max(scores) if ai == "O" else min(scores)

board = [" "] * 9

while True:
    show()
    pos = int(input("Enter position (1-9): ")) - 1

    if pos < 0 or pos > 8 or board[pos] != " ":
        print("Invalid move!")
        continue

    board[pos] = "X"

    if winner("X"):
        show()
        print("You Win!")
        break
    if " " not in board:
        show()
        print("Draw!")
        break

    best, move = -2, -1
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax("X")
            board[i] = " "
            if score > best:
                best, move = score, i

    board[move] = "O"

    if winner("O"):
        show()
        print("AI Wins! You Lose!")
        break
    if " " not in board:
        show()
        print("Draw!")
        break
