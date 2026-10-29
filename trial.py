
from collections import deque

def moves(x, y, a, b):
    return [
        ((a, y), "fill jug 1"),
        ((x, b), "fill jug 2"),
        ((0, y), "empty jug 1"),
        ((x, 0), "empty jug 2"),
        ((x - min(x, b-y), y + min(x, b-y)), "pour from 1 to 2"),
        ((x + min(y, a-x), y - min(y, a-x)), "pour from 2 to 1")
    ]

def bfs(a, b, goal):
    q = deque([((0, 0), [])])
    visited = set()

    while q:
        (x, y), path = q.popleft()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        if x == goal or y == goal:
            for action, state in path:
                print(action, "->", state)
            print("Goal reached:", (x, y))
            return

        for state, action in moves(x, y, a, b):
            if state not in visited:
                q.append((state, path + [(action, state)]))

    print("No solution")

def dfs(a, b, goal):
    stack = [((0, 0), [])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        if x == goal or y == goal:
            for action, state in path:
                print(action, "->", state)
            print("Goal reached:", (x, y))
            return

        for state, action in moves(x, y, a, b):
            if state not in visited:
                stack.append((state, path + [(action, state)]))

    print("No solution")

a = int(input("Enter Jug 1 capacity: "))
b = int(input("Enter Jug 2 capacity: "))
goal = int(input("Enter goal amount: "))

print("\nBFS:")
bfs(a, b, goal)

print("\nDFS:")
dfs(a, b, goal)
