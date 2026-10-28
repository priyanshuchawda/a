# Implement the A* search algorithm for solving the 8-puzzle problem.
# i. Use an appropriate heuristic function (e.g., misplaced tiles or Manhattan
#  distance).
# ii. Display the sequence of states leading to the goal state along with the cost.

import heapq

def heuristic(state, goal):
    distance = 0
    for i in range(9):
        if state[i] != 0:
            j = goal.index(state[i])
            distance += abs(i//3 - j//3) + abs(i%3 - j%3)
    return distance

def solve(start, goal):
    pq = [(heuristic(start, goal), 0, start, [])]
    visited = set()

    while pq:
        f, cost, state, path = heapq.heappop(pq)

        if state in visited:
            continue
        visited.add(state)
        path = path + [state]

        if state == goal:
            for s in path:
                print(s[:3])
                print(s[3:6])
                print(s[6:])
                print()
            print("Total cost:", cost)
            return

        z = state.index(0)
        for move in [-3, 3, -1, 1]:
            n = z + move
            if 0 <= n < 9 and not (
                move == -1 and z % 3 == 0 or
                move == 1 and z % 3 == 2
            ):
                new = state.copy()
                new[z], new[n] = new[n], new[z]
                if tuple(new) not in visited:
                    g = cost + 1
                    h = heuristic(new, goal)
                    heapq.heappush(
                        pq, (g + h, g, new, path)
                    )

start = [1, 2, 3,
         4, 0, 6,
         7, 5, 8]

goal = [1, 2, 3,
        4, 5, 6,
        7, 8, 0]

solve(start, goal)
