"""
Autonomous Cleaning Strategy Planner
=====================================
A tkinter-based interactive AI simulation demonstrating:
- CO1: PEAS agent model, state/action/transition representation
- CO2: BFS, UCS, A*, Greedy search with live visualization
- CO3: CSP constraint validation (movement rules, obstacle avoidance)
- CO4: Utility-based decision making for uncleaned cell priority

"""

import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

# ─────────────────────────────────────────────
# CO1 ── Agent & Environment Representation
# ─────────────────────────────────────────────

ROWS, COLS = 12, 12
CELL_SIZE = 44

# Cell states
FREE      = 0
OBSTACLE  = 1
CLEANED   = 2
FRONTIER  = 3
VISITED   = 4
PATH      = 5

COLORS = {
    FREE:     "#E6F1FB",
    OBSTACLE: "#2C2C2A",
    CLEANED:  "#97C459",
    FRONTIER: "#EF9F27",
    VISITED:  "#B5D4F4",
    PATH:     "#E24B4A",
    "start":  "#185FA5",
    "goal":   "#D4537E",
    "grid":   "#C8D8E8",
    "text":   "#1a1a1a",
}

@dataclass
class AgentState:
    """CO1: Formal state representation for the cleaning agent."""
    row: int
    col: int
    cleaned: frozenset = field(default_factory=frozenset)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

@dataclass(order=True)
class SearchNode:
    """CO1: Search node with priority for priority queue."""
    priority: float
    g_cost: float = field(compare=False)
    state: AgentState = field(compare=False)
    path: list = field(default_factory=list, compare=False)
    action: str = field(default="", compare=False)

class CleaningEnvironment:
    """
    CO1: PEAS model for the Cleaning Agent
    ----------------------------------------
    Performance : Maximize coverage, minimize path cost
    Environment : 12x12 grid with free cells and obstacles
    Actuators   : Move N/S/E/W, Clean current cell
    Sensors     : Current position, obstacle detection, cleaned map
    """
    ACTIONS = [(-1,0,"North"), (1,0,"South"), (0,-1,"West"), (0,1,"East")]

    def __init__(self, rows=ROWS, cols=COLS):
        self.rows = rows
        self.cols = cols
        self.grid = [[FREE]*cols for _ in range(rows)]

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r, c):
        return self.is_valid(r, c) and self.grid[r][c] != OBSTACLE

    def neighbors(self, r, c):
        """Return valid (neighbor_r, neighbor_c, action_name) tuples."""
        result = []
        for dr, dc, name in self.ACTIONS:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                result.append((nr, nc, name))
        return result

    def transition(self, state: AgentState, action: tuple):
        """CO1: Transition model T(s, a) → s'"""
        dr, dc, name = action
        nr, nc = state.row + dr, state.col + dc
        if self.is_free(nr, nc):
            return AgentState(row=nr, col=nc)
        return None

    def step_cost(self, from_r, from_c, to_r, to_c):
        """CO1: Path cost (uniform=1 for each move)."""
        return 1

    def free_count(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols)
                   if self.grid[r][c] != OBSTACLE)

    def load_example(self):
        walls = [
            (1,3),(2,3),(3,3),(3,4),(3,5),
            (5,2),(5,3),(5,4),(5,8),(5,9),
            (7,5),(7,6),(7,7),(8,7),(9,7),
            (2,8),(2,9),(2,10),(4,7),(4,8),
            (6,1),(6,2),(8,2),(8,3),(8,4),
            (9,4),(10,4),(10,5),(10,6)
        ]
        for r, c in walls:
            if self.is_valid(r, c):
                self.grid[r][c] = OBSTACLE


# ─────────────────────────────────────────────
# CO2 ── Search Algorithms
# ─────────────────────────────────────────────

class SearchEngine:
    """
    CO2: Classical AI Search Algorithms
    Implements BFS, UCS, A*, Greedy with
    step-by-step generator for live visualization.
    """

    def __init__(self, env: CleaningEnvironment):
        self.env = env

    def manhattan(self, r, c, gr, gc):
        """CO2: Admissible & consistent heuristic h(n) = |Δrow| + |Δcol|."""
        return abs(r - gr) + abs(c - gc)

    def bfs(self, start, goal):
        """
        CO2: Breadth-First Search
        Complete: Yes | Optimal: Yes (unit cost) | Time: O(b^d) | Space: O(b^d)
        """
        frontier = deque()
        frontier.append((start[0], start[1], [(start[0], start[1])]))
        visited = {(start[0], start[1])}
        while frontier:
            r, c, path = frontier.popleft()
            yield "expand", (r, c), path, len(visited), f"BFS expand ({r},{c}) depth={len(path)-1}"
            if (r, c) == goal:
                yield "found", (r, c), path, len(visited), f"FOUND! Length={len(path)-1}"
                return
            for nr, nc, action in self.env.neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    frontier.append((nr, nc, path + [(nr, nc)]))
                    yield "frontier", (nr, nc), path, len(visited), f"  → queue ({nr},{nc}) via {action}"
        yield "fail", None, [], len(visited), "No path found"

    def ucs(self, start, goal):
        """
        CO2: Uniform Cost Search
        Complete: Yes | Optimal: Yes | Time: O(b^(1+C*/ε))
        """
        counter = 0
        pq = [(0, counter, start[0], start[1], [(start[0], start[1])])]
        visited = {}
        while pq:
            cost, _, r, c, path = heapq.heappop(pq)
            if (r, c) in visited and visited[(r,c)] <= cost:
                continue
            visited[(r,c)] = cost
            yield "expand", (r, c), path, len(visited), f"UCS expand ({r},{c}) g={cost}"
            if (r, c) == goal:
                yield "found", (r, c), path, len(visited), f"FOUND! Length={len(path)-1} cost={cost}"
                return
            for nr, nc, action in self.env.neighbors(r, c):
                new_cost = cost + self.env.step_cost(r, c, nr, nc)
                counter += 1
                heapq.heappush(pq, (new_cost, counter, nr, nc, path + [(nr, nc)]))
                yield "frontier", (nr, nc), path, len(visited), f"  → push ({nr},{nc}) via {action} cost={new_cost}"
        yield "fail", None, [], len(visited), "No path found"

    def astar(self, start, goal):
        """
        CO2: A* Search with Manhattan heuristic
        Complete: Yes | Optimal: Yes (admissible h) | Time: O(b^d) best case
        Heuristic: h(n) = Manhattan distance (admissible + consistent)
        """
        counter = 0
        gr, gc = goal
        h0 = self.manhattan(start[0], start[1], gr, gc)
        pq = [(h0, counter, 0, start[0], start[1], [(start[0], start[1])])]
        visited = {}
        while pq:
            f, _, g, r, c, path = heapq.heappop(pq)
            if (r, c) in visited and visited[(r,c)] <= g:
                continue
            visited[(r,c)] = g
            h = self.manhattan(r, c, gr, gc)
            yield "expand", (r, c), path, len(visited), f"A* expand ({r},{c}) g={g} h={h} f={g+h}"
            if (r, c) == goal:
                yield "found", (r, c), path, len(visited), f"FOUND! Length={len(path)-1} f={f}"
                return
            for nr, nc, action in self.env.neighbors(r, c):
                ng = g + self.env.step_cost(r, c, nr, nc)
                nh = self.manhattan(nr, nc, gr, gc)
                counter += 1
                heapq.heappush(pq, (ng+nh, counter, ng, nr, nc, path + [(nr, nc)]))
                yield "frontier", (nr, nc), path, len(visited), f"  → push ({nr},{nc}) via {action} g={ng} h={nh} f={ng+nh}"
        yield "fail", None, [], len(visited), "No path found"

    def greedy(self, start, goal):
        """
        CO2: Greedy Best-First Search
        Complete: No (can loop) | Optimal: No | Fast but suboptimal
        """
        counter = 0
        gr, gc = goal
        h0 = self.manhattan(start[0], start[1], gr, gc)
        pq = [(h0, counter, start[0], start[1], [(start[0], start[1])])]
        visited = {(start[0], start[1])}
        while pq:
            h, _, r, c, path = heapq.heappop(pq)
            yield "expand", (r, c), path, len(visited), f"Greedy expand ({r},{c}) h={h}"
            if (r, c) == goal:
                yield "found", (r, c), path, len(visited), f"FOUND! Length={len(path)-1}"
                return
            for nr, nc, action in self.env.neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    nh = self.manhattan(nr, nc, gr, gc)
                    counter += 1
                    heapq.heappush(pq, (nh, counter, nr, nc, path + [(nr, nc)]))
                    yield "frontier", (nr, nc), path, len(visited), f"  → push ({nr},{nc}) via {action} h={nh}"
        yield "fail", None, [], len(visited), "No path found"


# ─────────────────────────────────────────────
# CO3 ── CSP Constraint Checker
# ─────────────────────────────────────────────

class CSPValidator:
    """
    CO3: Constraint Satisfaction Problem Validator
    Constraints enforced on any generated path:
    1. No diagonal movement (4-connectivity only)
    2. No obstacle cell traversal
    3. Each step must be exactly 1 cell
    4. Start and goal must be reachable (free cells)
    """

    def __init__(self, env: CleaningEnvironment):
        self.env = env

    def validate_path(self, path, start, goal):
        violations = []
        explanations = []

        # Constraint 1: start must be free
        if not self.env.is_free(start[0], start[1]):
            violations.append(f"C1 FAIL: Start ({start[0]},{start[1]}) is an obstacle")

        # Constraint 2: goal must be free
        if not self.env.is_free(goal[0], goal[1]):
            violations.append(f"C2 FAIL: Goal ({goal[0]},{goal[1]}) is an obstacle")

        for i in range(1, len(path)):
            pr, pc = path[i-1]
            cr, cc = path[i]
            dr = abs(cr - pr)
            dc = abs(cc - pc)

            # Constraint 3: exactly 1 step (no teleport, no diagonal)
            if dr + dc == 0:
                violations.append(f"C3 FAIL: Step {i} has no movement at ({cr},{cc})")
            elif dr + dc > 1:
                violations.append(f"C3 FAIL: Step {i} diagonal move ({pr},{pc})→({cr},{cc})")
            elif dr > 1 or dc > 1:
                violations.append(f"C3 FAIL: Step {i} jump from ({pr},{pc})→({cr},{cc})")

            # Constraint 4: no obstacle traversal
            if self.env.grid[cr][cc] == OBSTACLE:
                violations.append(f"C4 FAIL: Step {i} obstacle at ({cr},{cc})")

        if not violations:
            explanations = [
                "✓ C1: Start cell is free",
                "✓ C2: Goal cell is free",
                "✓ C3: All moves are valid 4-directional steps",
                "✓ C4: No obstacle cells traversed",
            ]

        return violations, explanations

    def explain_failure(self, r, c):
        """CO3: Explain why a constraint failed at this cell."""
        if self.env.grid[r][c] == OBSTACLE:
            return f"Cell ({r},{c}) is an obstacle — movement blocked (C4)"
        if not self.env.is_valid(r, c):
            return f"Cell ({r},{c}) is out of bounds — invalid action (C3)"
        neighbors = self.env.neighbors(r, c)
        if not neighbors:
            return f"Cell ({r},{c}) is isolated — surrounded by obstacles (dead end)"
        return f"Cell ({r},{c}) is reachable with {len(neighbors)} neighbor(s)"


# ─────────────────────────────────────────────
# CO4 ── Utility-Based Decision Making
# ─────────────────────────────────────────────

class UtilityAgent:
    """
    CO4: Utility-based goal selection
    Uses a utility function to select the next uncleaned
    cell to target, balancing coverage vs. travel cost.
    """

    def utility(self, current_pos, candidate_pos, cleaned_set, env):
        """
        CO4: U(s) = Coverage_gain / Travel_cost
        Higher utility = better next target
        """
        cr, cc = current_pos
        tr, tc = candidate_pos
        distance = abs(tr - cr) + abs(tc - cc) + 1  # +1 to avoid div-by-zero
        # Reward uncleaned, penalise far targets
        coverage_gain = 1 if (tr, tc) not in cleaned_set else 0
        return coverage_gain / distance

    def select_next_goal(self, current_pos, env, cleaned_set):
        """CO4: Choose best uncleaned cell based on utility."""
        best_util = -1
        best_cell = None
        for r in range(env.rows):
            for c in range(env.cols):
                if env.grid[r][c] == FREE and (r, c) not in cleaned_set:
                    u = self.utility(current_pos, (r, c), cleaned_set, env)
                    if u > best_util:
                        best_util = u
                        best_cell = (r, c)
        return best_cell, best_util


# ─────────────────────────────────────────────
# GUI ── Tkinter Dashboard
# ─────────────────────────────────────────────

class CleaningPlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Cleaning Strategy Planner")
        self.root.configure(bg="#0f0f0f")
        self.root.resizable(True, True)

        self.env = CleaningEnvironment()
        self.engine = SearchEngine(self.env)
        self.csp = CSPValidator(self.env)
        self.utility_agent = UtilityAgent()

        self.start = (0, 0)
        self.goal = (ROWS-1, COLS-1)
        self.draw_mode = tk.StringVar(value="obstacle")
        self.algo_var = tk.StringVar(value="astar")
        self.speed_ms = tk.IntVar(value=60)

        # Search state
        self.search_gen = None
        self.anim_id = None
        self.is_running = False
        self.display_grid = [[FREE]*COLS for _ in range(ROWS)]
        self.path_cells = []
        self.nodes_expanded = 0
        self.path_len = 0
        self.start_time = 0
        self.peak_mem = 0

        self._build_ui()
        self.env.load_example()
        self._sync_display()
        self._draw()
        self._log("// Autonomous Cleaning Strategy Planner", "info")
        self._log("// Example maze loaded. Select algorithm → Run.", "info")
        self._log("// Draw: click/drag on grid to place obstacles.", "muted")

    # ── UI Construction ──────────────────────

    def _build_ui(self):
        # Top bar
        header = tk.Frame(self.root, bg="#0f0f0f", pady=10)
        header.pack(fill="x", padx=16)
        tk.Label(header, text="CLEANING STRATEGY PLANNER",
                 font=("Courier", 13, "bold"), bg="#0f0f0f", fg="#E6F1FB").pack(side="left")
        tk.Label(header, text="// autonomous agent · search · CSP · utility",
                 font=("Courier", 9), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=12)

        # Main layout
        content = tk.Frame(self.root, bg="#0f0f0f")
        content.pack(fill="both", expand=True, padx=16, pady=(0,12))

        left = tk.Frame(content, bg="#0f0f0f")
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(content, bg="#181818", width=260,
                         highlightbackground="#2C2C2A", highlightthickness=1)
        right.pack(side="right", fill="y", padx=(12,0))
        right.pack_propagate(False)

        self._build_canvas(left)
        self._build_controls(left)
        self._build_sidebar(right)

    def _build_canvas(self, parent):
        canvas_frame = tk.Frame(parent, bg="#0f0f0f")
        canvas_frame.pack()

        # Algorithm tabs
        tab_row = tk.Frame(canvas_frame, bg="#0f0f0f", pady=6)
        tab_row.pack(fill="x")
        tk.Label(tab_row, text="ALGORITHM:", font=("Courier", 9, "bold"),
                 bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=(0,8))
        for algo, label in [("bfs","BFS"), ("ucs","UCS"), ("astar","A*"), ("greedy","Greedy")]:
            tk.Radiobutton(tab_row, text=label, variable=self.algo_var, value=algo,
                           font=("Courier", 10, "bold"), bg="#0f0f0f", fg="#E6F1FB",
                           selectcolor="#185FA5", activebackground="#0f0f0f",
                           activeforeground="#fff", indicatoron=False,
                           relief="flat", padx=10, pady=4,
                           command=self._on_algo_change).pack(side="left", padx=2)

        w = COLS * CELL_SIZE + 1
        h = ROWS * CELL_SIZE + 1
        self.canvas = tk.Canvas(canvas_frame, width=w, height=h,
                                bg="#0a0a0a", highlightthickness=1,
                                highlightbackground="#2C2C2A", cursor="crosshair")
        self.canvas.pack(pady=6)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Legend
        leg_frame = tk.Frame(canvas_frame, bg="#0f0f0f")
        leg_frame.pack(fill="x", pady=(0,4))
        items = [
            ("Free", COLORS[FREE]), ("Obstacle", COLORS[OBSTACLE]),
            ("Visited", COLORS[VISITED]), ("Frontier", COLORS[FRONTIER]),
            ("Path", COLORS[PATH]), ("Cleaned", COLORS[CLEANED]),
        ]
        for label, color in items:
            dot = tk.Canvas(leg_frame, width=12, height=12, bg="#0f0f0f",
                            highlightthickness=0)
            dot.pack(side="left", padx=(6,2))
            dot.create_rectangle(0,0,12,12, fill=color, outline="")
            tk.Label(leg_frame, text=label, font=("Courier", 8),
                     bg="#0f0f0f", fg="#888780").pack(side="left", padx=(0,4))

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg="#0f0f0f", pady=6)
        ctrl.pack(fill="x")

        # Draw mode
        mode_frame = tk.Frame(ctrl, bg="#0f0f0f")
        mode_frame.pack(fill="x", pady=(0,6))
        tk.Label(mode_frame, text="DRAW:", font=("Courier", 9, "bold"),
                 bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=(0,8))
        for val, label in [("obstacle","Obstacle"), ("start","Set Start"),
                            ("goal","Set Goal"), ("erase","Erase")]:
            tk.Radiobutton(mode_frame, text=label, variable=self.draw_mode, value=val,
                           font=("Courier", 9), bg="#0f0f0f", fg="#E6F1FB",
                           selectcolor="#633806", activebackground="#0f0f0f",
                           indicatoron=False, relief="flat",
                           padx=8, pady=3).pack(side="left", padx=2)

        # Buttons
        btn_frame = tk.Frame(ctrl, bg="#0f0f0f")
        btn_frame.pack(fill="x")

        self.run_btn = tk.Button(btn_frame, text="▶  RUN",
                                 font=("Courier", 10, "bold"),
                                 bg="#185FA5", fg="white", relief="flat",
                                 padx=16, pady=5, cursor="hand2",
                                 command=self._toggle_run)
        self.run_btn.pack(side="left", padx=(0,4))

        for text, cmd in [("  Step  ", self._step_once),
                          ("↺ Reset ", self._reset_search),
                          ("  Clear ", self._clear_all),
                          ("Example ", self._load_example)]:
            tk.Button(btn_frame, text=text, font=("Courier", 9),
                      bg="#1e1e1e", fg="#E6F1FB", relief="flat",
                      padx=8, pady=5, cursor="hand2",
                      activebackground="#2C2C2A",
                      command=cmd).pack(side="left", padx=2)

        # Speed
        spd_frame = tk.Frame(ctrl, bg="#0f0f0f", pady=4)
        spd_frame.pack(fill="x")
        tk.Label(spd_frame, text="SPEED:", font=("Courier", 9, "bold"),
                 bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=(0,8))
        tk.Scale(spd_frame, from_=10, to=400, orient="horizontal",
                 variable=self.speed_ms, length=200,
                 bg="#0f0f0f", fg="#E6F1FB", troughcolor="#1e1e1e",
                 highlightthickness=0, showvalue=True,
                 font=("Courier", 8)).pack(side="left")
        tk.Label(spd_frame, text="ms delay", font=("Courier", 8),
                 bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=4)

    def _build_sidebar(self, parent):
        pad = dict(padx=12, pady=0)

        # PEAS
        tk.Label(parent, text="PEAS MODEL", font=("Courier", 9, "bold"),
                 bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(12,4))
        peas = [("P", "Max coverage, min cost"),
                ("E", "12×12 grid with obstacles"),
                ("A", "Move N/S/E/W · Clean"),
                ("S", "Search + heuristic engine")]
        for k, v in peas:
            row = tk.Frame(parent, bg="#181818")
            row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=k, font=("Courier", 9, "bold"),
                     bg="#181818", fg="#185FA5", width=2).pack(side="left")
            tk.Label(row, text=v, font=("Courier", 8),
                     bg="#181818", fg="#888780", anchor="w").pack(side="left")

        self._sep(parent)

        # Metrics
        tk.Label(parent, text="METRICS", font=("Courier", 9, "bold"),
                 bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        mg = tk.Frame(parent, bg="#181818")
        mg.pack(fill="x", padx=12)
        self.m_nodes = self._metric_card(mg, "0", "Nodes expanded", 0, 0)
        self.m_len   = self._metric_card(mg, "0", "Path length",    0, 1)
        self.m_cover = self._metric_card(mg, "0%","Coverage",       1, 0)
        self.m_time  = self._metric_card(mg, "0ms","Runtime",       1, 1)
        mg2 = tk.Frame(parent, bg="#181818")
        mg2.pack(fill="x", padx=12, pady=(4,0))
        self.m_mem   = self._metric_card(mg2, "0KB", "Peak memory",  0, 0)
        self.m_util  = self._metric_card(mg2, "-",   "Next utility", 0, 1)

        self._sep(parent)

        # Trace log
        tk.Label(parent, text="TRACE LOG", font=("Courier", 9, "bold"),
                 bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        log_frame = tk.Frame(parent, bg="#181818")
        log_frame.pack(fill="both", expand=True, padx=12)
        self.trace_text = tk.Text(log_frame, height=10, bg="#0a0a0a",
                                  fg="#888780", font=("Courier", 8),
                                  relief="flat", wrap="word",
                                  insertbackground="#fff")
        sb = tk.Scrollbar(log_frame, command=self.trace_text.yview,
                          bg="#1e1e1e", troughcolor="#0a0a0a")
        self.trace_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.trace_text.pack(fill="both", expand=True)
        self.trace_text.tag_config("info",  foreground="#185FA5")
        self.trace_text.tag_config("ok",    foreground="#3B6D11")
        self.trace_text.tag_config("warn",  foreground="#BA7517")
        self.trace_text.tag_config("error", foreground="#A32D2D")
        self.trace_text.tag_config("muted", foreground="#444441")

        self._sep(parent)

        # CSP panel
        tk.Label(parent, text="CSP CONSTRAINTS", font=("Courier", 9, "bold"),
                 bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        self.csp_text = tk.Text(parent, height=5, bg="#0a0a0a",
                                fg="#5F5E5A", font=("Courier", 8),
                                relief="flat", wrap="word", state="disabled")
        self.csp_text.pack(fill="x", padx=12, pady=(0,12))
        self._update_csp("No violations detected.")

    def _metric_card(self, parent, val, label, row, col):
        f = tk.Frame(parent, bg="#0a0a0a", padx=8, pady=6,
                     highlightbackground="#2C2C2A", highlightthickness=1)
        f.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
        parent.columnconfigure(col, weight=1)
        v = tk.Label(f, text=val, font=("Courier", 14, "bold"),
                     bg="#0a0a0a", fg="#E6F1FB")
        v.pack(anchor="w")
        tk.Label(f, text=label, font=("Courier", 7),
                 bg="#0a0a0a", fg="#5F5E5A").pack(anchor="w")
        return v

    def _sep(self, parent):
        tk.Frame(parent, height=1, bg="#2C2C2A").pack(fill="x", padx=12, pady=4)

    # ── Drawing ──────────────────────────────

    def _draw(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                state = self.display_grid[r][c]
                fill = COLORS.get(state, COLORS[FREE])
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=fill, outline="#1e1e1e", width=1)

        # Start marker
        sr, sc = self.start
        sx, sy = sc*CELL_SIZE + CELL_SIZE//2, sr*CELL_SIZE + CELL_SIZE//2
        self.canvas.create_oval(sx-9, sy-9, sx+9, sy+9,
                                fill=COLORS["start"], outline="white", width=2)
        self.canvas.create_text(sx, sy, text="S", fill="white",
                                font=("Courier", 8, "bold"))

        # Goal marker
        gr, gc_ = self.goal
        gx, gy = gc_*CELL_SIZE + CELL_SIZE//2, gr*CELL_SIZE + CELL_SIZE//2
        pts = [gx, gy-10, gx+8, gy+6, gx-8, gy+6]
        self.canvas.create_polygon(pts, fill=COLORS["goal"], outline="white", width=2)
        self.canvas.create_text(gx, gy+1, text="G", fill="white",
                                font=("Courier", 7, "bold"))

    def _sync_display(self):
        for r in range(ROWS):
            for c in range(COLS):
                if self.env.grid[r][c] == OBSTACLE:
                    self.display_grid[r][c] = OBSTACLE
                else:
                    self.display_grid[r][c] = FREE

    # ── Event Handlers ────────────────────────

    def _cell_from_event(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None, None

    def _on_click(self, event):
        r, c = self._cell_from_event(event)
        if r is None:
            return
        self._apply_draw(r, c)

    def _on_drag(self, event):
        r, c = self._cell_from_event(event)
        if r is None:
            return
        self._apply_draw(r, c)

    def _apply_draw(self, r, c):
        mode = self.draw_mode.get()
        if mode == "obstacle":
            if (r, c) != self.start and (r, c) != self.goal:
                self.env.grid[r][c] = OBSTACLE
                self.display_grid[r][c] = OBSTACLE
        elif mode == "erase":
            self.env.grid[r][c] = FREE
            self.display_grid[r][c] = FREE
        elif mode == "start":
            if self.env.grid[r][c] != OBSTACLE:
                self.start = (r, c)
        elif mode == "goal":
            if self.env.grid[r][c] != OBSTACLE:
                self.goal = (r, c)
        self._reset_search()
        self._draw()

    def _on_algo_change(self):
        self._reset_search()
        self._log(f"// Algorithm: {self.algo_var.get().upper()}", "info")

    # ── Search Control ────────────────────────

    def _get_generator(self):
        algo = self.algo_var.get()
        s, g = self.start, self.goal
        tracemalloc.start()
        self.start_time = time.perf_counter()
        if algo == "bfs":    return self.engine.bfs(s, g)
        if algo == "ucs":    return self.engine.ucs(s, g)
        if algo == "astar":  return self.engine.astar(s, g)
        if algo == "greedy": return self.engine.greedy(s, g)

    def _toggle_run(self):
        if self.is_running:
            self._stop_anim()
        else:
            self._start_run()

    def _start_run(self):
        self._reset_search(keep_env=True)
        self.search_gen = self._get_generator()
        self.is_running = True
        self.run_btn.config(text="■  STOP", bg="#A32D2D")
        self._log(f"// Running {self.algo_var.get().upper()} from {self.start} → {self.goal}", "info")
        self._schedule_step()

    def _schedule_step(self):
        if self.is_running and self.search_gen:
            self._do_step()
            if self.is_running:
                self.anim_id = self.root.after(self.speed_ms.get(), self._schedule_step)

    def _do_step(self):
        try:
            event_type, pos, path, n_visited, msg = next(self.search_gen)
            self.nodes_expanded = n_visited
            self._log(f"  {msg}", "muted" if event_type == "frontier" else "ok" if event_type == "found" else None)

            if event_type == "expand" and pos:
                r, c = pos
                if (r, c) != self.start and (r, c) != self.goal:
                    self.display_grid[r][c] = VISITED

            elif event_type == "frontier" and pos:
                r, c = pos
                if (r, c) != self.start and (r, c) != self.goal:
                    if self.display_grid[r][c] != VISITED:
                        self.display_grid[r][c] = FRONTIER

            elif event_type == "found":
                self._on_found(path)

            elif event_type == "fail":
                self._log("// No path found! Check if goal is reachable.", "warn")
                self._stop_anim()

            self._update_metrics()
            self._draw()

        except StopIteration:
            self._stop_anim()

    def _on_found(self, path):
        self.path_cells = path
        self.path_len = len(path) - 1
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mem = peak // 1024

        for r, c in path:
            if (r, c) != self.start and (r, c) != self.goal:
                self.display_grid[r][c] = PATH

        violations, explanations = self.csp.validate_path(path, self.start, self.goal)
        free = self.env.free_count()
        coverage = round(len(path) / free * 100) if free else 0

        self.m_time.config(text=f"{elapsed_ms}ms")
        self.m_cover.config(text=f"{coverage}%")
        self.m_mem.config(text=f"{self.peak_mem}KB")

        # Utility agent picks next best uncleaned cell
        cleaned_set = set(path)
        next_goal, util_val = self.utility_agent.select_next_goal(self.goal, self.env, cleaned_set)
        self.m_util.config(text=f"{round(util_val,3)}" if util_val else "-")

        self._log(f"// PATH FOUND ✓  length={self.path_len}  nodes={self.nodes_expanded}  time={elapsed_ms}ms", "ok")
        self._log(f"// Coverage: {coverage}%  |  Peak mem: {self.peak_mem}KB", "ok")
        if next_goal:
            self._log(f"// Utility agent → next target {next_goal} (u={round(util_val,3)})", "info")

        if violations:
            self._update_csp("\n".join(violations))
            for v in violations:
                self._log(f"  CSP: {v}", "error")
        else:
            self._update_csp("\n".join(explanations))

        self._stop_anim()

    def _step_once(self):
        if self.search_gen is None:
            self.search_gen = self._get_generator()
            self._reset_search(keep_env=True, keep_gen=True)
            self._log(f"// Stepping through {self.algo_var.get().upper()}...", "info")
        self._do_step()

    def _stop_anim(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.is_running = False
        self.run_btn.config(text="▶  RUN", bg="#185FA5")

    def _reset_search(self, keep_env=False, keep_gen=False):
        self._stop_anim()
        if not keep_gen:
            self.search_gen = None
        self.path_cells = []
        self.nodes_expanded = 0
        self.path_len = 0
        self._sync_display()
        self._update_metrics()
        self._update_csp("No violations detected.")
        self._draw()

    def _clear_all(self):
        self.env.grid = [[FREE]*COLS for _ in range(ROWS)]
        self.start = (0, 0)
        self.goal = (ROWS-1, COLS-1)
        self._reset_search()
        self._log("// Grid cleared.", "info")

    def _load_example(self):
        self.env.grid = [[FREE]*COLS for _ in range(ROWS)]
        self.start = (0, 0)
        self.goal = (ROWS-1, COLS-1)
        self.env.load_example()
        self._reset_search()
        self._log("// Example maze loaded. Run an algorithm.", "ok")

    # ── Metrics & Logging ─────────────────────

    def _update_metrics(self):
        free = self.env.free_count()
        coverage = round(len(self.path_cells) / free * 100) if free and self.path_cells else 0
        self.m_nodes.config(text=str(self.nodes_expanded))
        self.m_len.config(text=str(self.path_len))
        self.m_cover.config(text=f"{coverage}%")

    def _log(self, msg, tag=None):
        self.trace_text.config(state="normal")
        if tag:
            self.trace_text.insert("end", msg + "\n", tag)
        else:
            self.trace_text.insert("end", msg + "\n")
        self.trace_text.see("end")
        self.trace_text.config(state="disabled")

    def _update_csp(self, text):
        self.csp_text.config(state="normal")
        self.csp_text.delete("1.0", "end")
        self.csp_text.insert("end", text)
        self.csp_text.config(state="disabled")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = CleaningPlannerGUI(root)
    root.mainloop()
