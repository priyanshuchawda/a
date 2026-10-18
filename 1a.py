# Write a program to implement the,
# i. Water jug problem using BFS and DFS search strategies. Input # jug capacities
# and goal state from the user. Display the sequence of states and # actions to reach
# the goal.

from collections import deque

def moves(x,y,a,b):
	return [
		((a,y), "Fill Jug 1"),
		((x,b), "Fill Jug 2"),
		((0,y), "Empty Jug 1"),
		((x,0), "Empty Jug 2"),
		((x - min(x,b-y) , y+min(x,b-y)), "Pour 1 to 2"),
		((x+min(y,a-x) , y - min(y,a-x)) , "Pour 2 to 1")
	]

def bfs(a,b,goal):
	q = deque([((0,0) , [])])
	visited = set()

	while q:
		(x,y), path = q.popleft()

		if x == goal or y == goal:
			for action , state in path:
				print(action , "->" , state)
			print("goal reached:", (x,y))
			return

		if (x,y) in visited:
			continue
		visited.add((x,y))

		for state, action in moves (x,y,a,b):
			if state not in visited:
				q.append((state , path + [(action,state)]))
	print ("none")



def dfs(a, b, goal):
    stack = [((0, 0), [])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if x == goal or y == goal:
            for action, state in path:
                print(action, "->", state)
            print("Goal reached:", (x, y))
            return

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for state, action in moves(x, y, a, b):
            if state not in visited:
                stack.append((state, path + [(action, state)]))

    print("None")


a = int(input("Enter Jug 1 capacity: ")) #5
b = int(input("Enter Jug 2 capacity: ")) #3
goal = int(input("Enter goal amount: ")) #4


bfs(a,b,goal)
print()
print()
dfs(a,b,goal)