"""Maze task for GPU-scale curriculum x MaxRL experiments.

Mirrors the MaxRL paper's maze setup (procedurally generated mazes, tiny
transformer, binary verifier on the emitted move sequence) but with a *smooth*
difficulty dimension: the maze is always 17x17, and level l requires reaching
a goal at BFS distance d = 4*(l+1) from the fixed start (1,1).  Longer routes
are concatenations of shorter ones, so competence transfers between adjacent
levels — maze *size* as difficulty proved to have a hard generalization cliff
(9x9 stays at exactly p=0 after 7x7 SFT).

Prompt tokens:  BOS <grid: '0' open / '1' wall / 'G' goal, ROW after each row> SEP
Response:       sequence of moves in {U,D,L,R} ending with EOS
Verifier:       simulate moves from (1,1); success iff at goal when EOS is
                emitted, never stepping into walls, within the move budget.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- tokenizer
PAD, BOS, SEP, EOS = 0, 1, 2, 3
T_OPEN, T_WALL, T_ROW, T_GOAL = 4, 5, 6, 11
T_U, T_D, T_L, T_R = 7, 8, 9, 10
VOCAB_SIZE = 12
MOVE_TOKENS = [T_U, T_D, T_L, T_R]
MOVE_DELTA = {T_U: (-1, 0), T_D: (1, 0), T_L: (0, -1), T_R: (0, 1)}

SIZE = 17
LEVELS = list(range(13))                        # 13 difficulty levels
LEVEL_DIST = {l: 4 + 2 * l for l in LEVELS}     # goal distance 4,6,...,28
# (17x17 Prim mazes have max BFS depth >= 28 in ~100% of samples; spacing of
# 2 keeps adjacent levels within the policy's generalization reach)
# perfect maze => unique simple path; small slack for wander-and-return
MOVE_BUDGET = {l: LEVEL_DIST[l] + 8 for l in LEVELS}
MAX_PROMPT_LEN = 2 + SIZE * (SIZE + 1)          # BOS + rows(+ROW) + SEP
MAX_RESP_LEN = MOVE_BUDGET[LEVELS[-1]] + 1      # moves + EOS


def gen_maze(size: int, rng: random.Random) -> np.ndarray:
    """Prim's-algorithm maze, same construction as maze/generate_maze.py."""
    grid = np.ones((size, size), dtype=np.int8)
    sx, sy = 1, 1
    grid[sx, sy] = 0
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    frontier = []
    for dx, dy in directions:
        nx, ny = sx + dx, sy + dy
        if 0 < nx < size - 1 and 0 < ny < size - 1 and grid[nx, ny] == 1:
            frontier.append((nx, ny))
    while frontier:
        idx = rng.randint(0, len(frontier) - 1)
        fx, fy = frontier.pop(idx)
        neighbors = [(fx + dx, fy + dy) for dx, dy in directions
                     if 0 <= fx + dx < size and 0 <= fy + dy < size
                     and grid[fx + dx, fy + dy] == 0]
        if neighbors:
            nx, ny = rng.choice(neighbors)
            grid[(fx + nx) // 2, (fy + ny) // 2] = 0
            grid[fx, fy] = 0
            for dx, dy in directions:
                wx, wy = fx + dx, fy + dy
                if (0 < wx < size - 1 and 0 < wy < size - 1
                        and grid[wx, wy] == 1 and (wx, wy) not in frontier):
                    frontier.append((wx, wy))
    grid[1, 1] = 0
    grid[size - 2, size - 2] = 0
    return grid


def bfs_tree(grid: np.ndarray, start=(1, 1)) -> dict:
    """BFS predecessor tree from start over open cells: node -> (parent, move)."""
    prev: dict = {start: None}
    q = deque([start])
    size = grid.shape[0]
    while q:
        cur = q.popleft()
        for tok, (dx, dy) in MOVE_DELTA.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                    and grid[nxt] == 0 and nxt not in prev):
                prev[nxt] = (cur, tok)
                q.append(nxt)
    return prev


def path_moves(prev: dict, goal) -> list[int] | None:
    if goal not in prev:
        return None
    moves = []
    node = goal
    while prev[node] is not None:
        node, tok = prev[node]
        moves.append(tok)
    return moves[::-1]


def encode_prompt(grid: np.ndarray, goal) -> list[int]:
    toks = [BOS]
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            if (i, j) == goal:
                toks.append(T_GOAL)
            else:
                toks.append(T_WALL if c else T_OPEN)
        toks.append(T_ROW)
    toks.append(SEP)
    return toks


def verify(grid: np.ndarray, goal, response: list[int]) -> bool:
    """Simulate response tokens; success iff at goal when EOS is emitted."""
    size = grid.shape[0]
    pos = (1, 1)
    for tok in response:
        if tok == EOS:
            return pos == goal
        if tok not in MOVE_DELTA:
            return False
        dx, dy = MOVE_DELTA[tok]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or grid[nxt] == 1:
            return False
        pos = nxt
    return False  # ran out of budget without EOS


@dataclass
class MazeTask:
    level: int
    grid: np.ndarray
    goal: tuple
    prompt: list[int]
    solution: list[int]


def sample_task(level: int, rng: random.Random) -> MazeTask:
    """Fresh 13x13 maze with a goal at exactly BFS distance LEVEL_DIST[level]
    (nearest available distance if the maze has no cell at that exact depth)."""
    target = LEVEL_DIST[level]
    while True:
        grid = gen_maze(SIZE, rng)
        prev = bfs_tree(grid)
        # depth of every reachable cell
        depths: dict = {}
        for node in prev:
            d, n = 0, node
            while prev[n] is not None:
                n = prev[n][0]
                d += 1
            depths[node] = d
        candidates = [n for n, d in depths.items() if d == target]
        if not candidates:
            close = [(abs(d - target), n) for n, d in depths.items() if d >= 6]
            if not close:
                continue
            gap, goal = min(close)
            if gap > 2:
                continue
        else:
            goal = candidates[rng.randint(0, len(candidates) - 1)]
        sol = path_moves(prev, goal)
        return MazeTask(level, grid, goal, encode_prompt(grid, goal), sol)


def sft_example(level: int, rng: random.Random) -> tuple[list[int], list[int]]:
    task = sample_task(level, rng)
    return task.prompt, task.solution + [EOS]


def simulate_prefix(grid: np.ndarray, response: list[int]) -> tuple[int, tuple]:
    """Longest legal move prefix of a response and the cell it reaches.

    Returns (n_legal_moves, final_pos).  Stops at the first illegal move,
    non-move token, or EOS.  A rollout's legal prefix is always a valid path
    from (1,1) to final_pos — the basis for hindsight goal relabeling.
    """
    size = grid.shape[0]
    pos = (1, 1)
    n = 0
    for tok in response:
        if tok not in MOVE_DELTA:
            break
        dx, dy = MOVE_DELTA[tok]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or grid[nxt] == 1:
            break
        pos = nxt
        n += 1
    return n, pos
