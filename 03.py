# Implement the A* search algorithm for solving the 8-puzzle problem.
# i. Use an appropriate heuristic function (e.g., misplaced tiles or Manhattan
#  distance).
# ii. Display the sequence of states leading to the goal state along with the cost.

import heapq

goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def heuristic(s):
    total_sum = 0
    for i in range(9):
        if s[i] != 0:
            if s[i] != goal[i]:
                total_sum += 1
    return total_sum

def solve(start):
    q = [(heuristic(start), 0, start, [])]
    visited = set()

    while q:
        f, cost, s, path = heapq.heappop(q)

        if s in visited:
            continue
        visited.add(s)
        path = path + [s]

        if s == goal:
            for state in path:
                print(state[:3])
                print(state[3:6])
                print(state[6:])
                print()
            print("Total cost:", cost)
            return

        z = s.index(0)

        for d in [-3, 3, -1, 1]:
            n = z + d

            if 0 <= n < 9 and not (d == -1 and z % 3 == 0) and not (d == 1 and z % 3 == 2):
                new = list(s)
                new[z], new[n] = new[n], new[z]
                new = tuple(new)

                if new not in visited:
                    g = cost + 1
                    heapq.heappush(q, (g + heuristic(new), g, new, path))

start = tuple(map(int, input("Enter 9 numbers (use 0 for blank): ").split()))
solve(start)
