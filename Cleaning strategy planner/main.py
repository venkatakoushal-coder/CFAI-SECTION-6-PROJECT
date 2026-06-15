import tkinter as tk
from collections import deque
import heapq, time, tracemalloc
R, C, SZ = 12, 12, 44
FREE, OBSTACLE, CLEANED, FRONTIER, VISITED, PATH = 0, 1, 2, 3, 4, 5
COLORS = {FREE: "#E6F1FB", OBSTACLE: "#2C2C2A", CLEANED: "#97C459", FRONTIER: "#EF9F27", VISITED: "#B5D4F4", PATH: "#E24B4A", "start": "#185FA5", "goal": "#D4537E", "grid": "#C8D8E8"}
class Env:
    def __init__(self):
        self.grid = [[FREE]*C for _ in range(R)]
        for r, c in [(1,3),(2,3),(3,3),(3,4),(3,5),(5,2),(5,3),(5,4),(5,8),(5,9),(7,5),(7,6),(7,7),(8,7),(9,7),(2,8),(2,9),(2,10),(4,7),(4,8),(6,1),(6,2),(8,2),(8,3),(8,4),(9,4),(10,4),(10,5),(10,6)]: self.grid[r][c] = OBSTACLE
    def is_free(self, r, c): return 0 <= r < R and 0 <= c < C and self.grid[r][c] != OBSTACLE
    def neighbors(self, r, c): return [(r+dr, c+dc, n) for dr, dc, n in [(-1,0,"North"), (1,0,"South"), (0,-1,"West"), (0,1,"East")] if self.is_free(r+dr, c+dc)]
class Search:
    def __init__(self, env): self.env = env
    def run(self, algo, start, goal):
        gr, gc = goal
        if algo == "bfs":
            q, vis = deque([(start, [start])]), {start}
            while q:
                (r, c), p = q.popleft()
                yield "expand", (r, c), p, (len(vis), []), f"Expanding node ({r},{c})"
                if (r, c) == goal: yield "found", (r, c), p, (len(vis), []), f"FOUND! Length={len(p)-1}"; return
                for nr, nc, act in self.env.neighbors(r, c):
                    if (nr, nc) not in vis: vis.add((nr, nc)); q.append(((nr, nc), p + [(nr, nc)])); yield "frontier", (nr, nc), p, (len(vis), []), f"  → queue ({nr},{nc}) via {act}"
                yield "step_frontier", (r, c), p, (len(vis), [x[0] for x in q]), ""
        else:
            cnt, pq, vis = 0, [], {}
            h0 = abs(start[0]-gr) + abs(start[1]-gc)
            if algo == "ucs": pq.append((0, 0, 0, start, [start]))
            elif algo == "astar": pq.append((h0, 0, 0, start, [start]))
            elif algo == "greedy": pq.append((h0, 0, 0, start, [start])); vis = {start}
            while pq:
                f, _, g, (r, c), p = heapq.heappop(pq)
                if algo != "greedy" and (r, c) in vis and vis[(r, c)] <= g: continue
                if algo != "greedy": vis[(r, c)] = g
                yield "expand", (r, c), p, (len(vis), []), f"Expanding node ({r},{c})"
                if (r, c) == goal: yield "found", (r, c), p, (len(vis), []), f"FOUND! Length={len(p)-1}" + (f" cost={g}" if algo=="ucs" else f" f={f}" if algo=="astar" else ""); return
                for nr, nc, act in self.env.neighbors(r, c):
                    cnt += 1; nh = abs(nr-gr) + abs(nc-gc)
                    if algo != "greedy" or (nr, nc) not in vis:
                        if algo == "greedy": vis.add((nr, nc))
                        ng, nf = ((g + 1) if algo != "greedy" else 0), (((g + 1) if algo != "greedy" else 0) + (nh if algo != "ucs" else 0))
                        heapq.heappush(pq, (nf, cnt, ng, (nr, nc), p + [(nr, nc)]))
                        yield "frontier", (nr, nc), p, (len(vis), []), f"  → push ({nr},{nc}) via {act} cost={ng}" if algo=="ucs" else f"  → push ({nr},{nc}) via {act} g={ng} h={nh} f={nf}" if algo=="astar" else f"  → push ({nr},{nc}) via {act} h={nf}"
                yield "step_frontier", (r, c), p, (len(vis), [x[3] for x in pq]), ""
        yield "fail", None, [], (len(vis), []), "No path found"
class CSP:
    def validate(self, p, s, g, grid):
        v = []
        if grid[s[0]][s[1]] == OBSTACLE: v.append(f"C1 FAIL: Start {s} is an obstacle")
        if grid[g[0]][g[1]] == OBSTACLE: v.append(f"C2 FAIL: Goal {g} is an obstacle")
        for i in range(1, len(p)):
            (pr, pc), (cr, cc) = p[i-1], p[i]; dr, dc = abs(cr - pr), abs(cc - pc)
            if dr + dc == 0: v.append(f"C3 FAIL: Step {i} has no movement at ({cr},{cc})")
            elif dr + dc > 1: v.append(f"C3 FAIL: Step {i} " + (f"jump from ({pr},{pc})→({cr},{cc})" if dr > 1 or dc > 1 else f"diagonal move ({pr},{pc})→({cr},{cc})"))
            if grid[cr][cc] == OBSTACLE: v.append(f"C4 FAIL: Step {i} obstacle at ({cr},{cc})")
        return v, ["✓ C1: Start cell is free", "✓ C2: Goal cell is free", "✓ C3: All moves are valid 4-directional steps", "✓ C4: No obstacle cells traversed"] if not v else []
class UtilityAgent:
    def select_next_goal(self, curr, grid, cleaned):
        cands = [((r, c), 1 / (abs(r - curr[0]) + abs(c - curr[1]) + 1)) for r in range(R) for c in range(C) if grid[r][c] == FREE and (r, c) not in cleaned]
        return max(cands, key=lambda x: x[1], default=(None, -1))
class CleaningPlannerGUI:
    def __init__(self, root):
        self.root = root; self.root.title("Autonomous Cleaning Strategy Planner"); self.root.configure(bg="#0f0f0f")
        self.env = Env(); self.search, self.csp, self.util = Search(self.env), CSP(), UtilityAgent()
        self.start, self.goal = (0, 0), (R-1, C-1)
        self.draw_mode, self.algo_var, self.speed_ms = tk.StringVar(value="obstacle"), tk.StringVar(value="astar"), tk.IntVar(value=300)
        self.search_gen, self.anim_id, self.is_running = None, None, False
        self.display_grid = [[FREE]*C for _ in range(R)]
        self.path_cells, self.nodes_expanded, self.path_len = [], 0, 0
        self.start_time = self.step_cnt = 0; self.build_ui(); self.reset_search()
    def build_ui(self):
        self.root.geometry("860x680")
        h = tk.Frame(self.root, bg="#0f0f0f", pady=10); h.pack(fill="x", padx=16)
        tk.Label(h, text="CLEANING STRATEGY PLANNER", font=("Courier", 13, "bold"), bg="#0f0f0f", fg="#E6F1FB").pack(side="left")
        tk.Label(h, text="// autonomous agent · search · CSP · utility", font=("Courier", 9), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=12)
        m = tk.Frame(self.root, bg="#0f0f0f"); m.pack(fill="both", expand=True, padx=16, pady=(0,12))
        left = tk.Frame(m, bg="#0f0f0f"); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(m, bg="#181818", width=260, highlightbackground="#2C2C2A", highlightthickness=1); right.pack(side="right", fill="y", padx=(12,0)); right.pack_propagate(False)
        tb = tk.Frame(left, bg="#0f0f0f", pady=4); tb.pack(fill="x")
        tr = tk.Frame(tb, bg="#0f0f0f"); tr.pack(side="left"); tk.Label(tr, text="ALGO:", font=("Courier", 9, "bold"), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=(0,4))
        for a, l in [("bfs","BFS"), ("ucs","UCS"), ("astar","A*"), ("greedy","Greedy")]: tk.Radiobutton(tr, text=l, variable=self.algo_var, value=a, font=("Courier", 9, "bold"), bg="#0f0f0f", fg="#E6F1FB", selectcolor="#185FA5", activebackground="#0f0f0f", indicatoron=False, relief="flat", padx=6, pady=2, command=self.on_algo_change).pack(side="left", padx=1)
        df = tk.Frame(tb, bg="#0f0f0f"); df.pack(side="right"); tk.Label(df, text="DRAW:", font=("Courier", 9, "bold"), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=(0,4))
        for v, l in [("obstacle","Obstacle"), ("start","Start"), ("goal","Goal"), ("erase","Erase")]: tk.Radiobutton(df, text=l, variable=self.draw_mode, value=v, font=("Courier", 9), bg="#0f0f0f", fg="#E6F1FB", selectcolor="#633806", activebackground="#0f0f0f", indicatoron=False, relief="flat", padx=6, pady=2).pack(side="left", padx=1)
        self.canvas = tk.Canvas(left, width=C*SZ+1, height=R*SZ+1, bg="#0a0a0a", highlightthickness=1, highlightbackground="#2C2C2A", cursor="crosshair")
        self.canvas.pack(pady=4); self.canvas.bind("<Button-1>", self.on_click); self.canvas.bind("<B1-Motion>", self.on_drag)
        lf = tk.Frame(left, bg="#0f0f0f"); lf.pack(fill="x", pady=2)
        for lbl, col in [("Free", COLORS[FREE]), ("Obstacle", COLORS[OBSTACLE]), ("Visited", COLORS[VISITED]), ("Frontier", COLORS[FRONTIER]), ("Path", COLORS[PATH]), ("Cleaned", COLORS[CLEANED])]:
            tk.Label(lf, text="  ", bg=col, font=("Courier", 6), relief="flat").pack(side="left", padx=(4,1)); tk.Label(lf, text=lbl, font=("Courier", 8), bg="#0f0f0f", fg="#888780").pack(side="left", padx=(0,3))
        bf = tk.Frame(left, bg="#0f0f0f", pady=4); bf.pack(fill="x")
        self.run_btn = tk.Button(bf, text="▶  RUN", font=("Courier", 10, "bold"), bg="#185FA5", fg="white", relief="flat", padx=12, pady=4, cursor="hand2", command=self.toggle_run); self.run_btn.pack(side="left", padx=2)
        for t, cmd in [("Step", self.step_once), ("Reset", self.reset_search), ("Clear", self.clear_all), ("Example", self.load_example)]: tk.Button(bf, text=t, font=("Courier", 9), bg="#1e1e1e", fg="#E6F1FB", relief="flat", padx=6, pady=4, cursor="hand2", activebackground="#2C2C2A", command=cmd).pack(side="left", padx=2)
        sf = tk.Frame(bf, bg="#0f0f0f"); sf.pack(side="right"); tk.Label(sf, text="SPD:", font=("Courier", 8, "bold"), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=2)
        tk.Scale(sf, from_=10, to=400, orient="horizontal", variable=self.speed_ms, length=100, bg="#0f0f0f", fg="#E6F1FB", troughcolor="#1e1e1e", highlightthickness=0, showvalue=False, font=("Courier", 8)).pack(side="left")
        tk.Label(sf, text="ms", font=("Courier", 8), bg="#0f0f0f", fg="#5F5E5A").pack(side="left", padx=2)
        tk.Label(right, text="PEAS MODEL", font=("Courier", 9, "bold"), bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(12,4))
        for k, v in [("P", "Max coverage, min cost"), ("E", f"{R}×{C} grid with obstacles"), ("A", "Move N/S/E/W · Clean"), ("S", "Search + heuristic engine")]: rf = tk.Frame(right, bg="#181818"); rf.pack(fill="x", padx=12, pady=1); tk.Label(rf, text=k, font=("Courier", 9, "bold"), bg="#181818", fg="#185FA5", width=2).pack(side="left"); tk.Label(rf, text=v, font=("Courier", 8), bg="#181818", fg="#888780", anchor="w").pack(side="left")
        def sep(): tk.Frame(right, height=1, bg="#2C2C2A").pack(fill="x", padx=12, pady=4)
        sep()
        tk.Label(right, text="METRICS", font=("Courier", 9, "bold"), bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        self.metrics, mg1, mg2 = {}, tk.Frame(right, bg="#181818"), tk.Frame(right, bg="#181818")
        mg1.pack(fill="x", padx=12); mg2.pack(fill="x", padx=12, pady=(4,0))
        for name, p, r, c in [("Nodes expanded", mg1, 0, 0), ("Path length", mg1, 0, 1), ("Coverage", mg1, 1, 0), ("Runtime", mg1, 1, 1), ("Peak memory", mg2, 0, 0), ("Next utility", mg2, 0, 1)]:
            f = tk.Frame(p, bg="#0a0a0a", padx=8, pady=6, highlightbackground="#2C2C2A", highlightthickness=1); f.grid(row=r, column=c, padx=3, pady=3, sticky="ew"); p.columnconfigure(c, weight=1); v = tk.Label(f, text="-", font=("Courier", 14, "bold"), bg="#0a0a0a", fg="#E6F1FB"); v.pack(anchor="w")
            tk.Label(f, text=name, font=("Courier", 7), bg="#0a0a0a", fg="#5F5E5A").pack(anchor="w"); self.metrics[name] = v
        sep(); tk.Label(right, text="TRACE LOG", font=("Courier", 9, "bold"), bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        lf = tk.Frame(right, bg="#181818"); lf.pack(fill="both", expand=True, padx=12); self.trace = tk.Text(lf, height=10, bg="#0a0a0a", fg="#888780", font=("Courier", 8), relief="flat", wrap="word")
        sb = tk.Scrollbar(lf, command=self.trace.yview, bg="#1e1e1e", troughcolor="#0a0a0a"); self.trace.config(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); self.trace.pack(fill="both", expand=True)
        for tag, col in [("info", "#185FA5"), ("ok", "#3B6D11"), ("warn", "#BA7517"), ("error", "#A32D2D"), ("muted", "#444441")]: self.trace.tag_config(tag, foreground=col)
        sep(); tk.Label(right, text="CSP CONSTRAINTS", font=("Courier", 9, "bold"), bg="#181818", fg="#5F5E5A").pack(anchor="w", padx=12, pady=(8,4))
        self.csp_text = tk.Text(right, height=5, bg="#0a0a0a", fg="#5F5E5A", font=("Courier", 8), relief="flat", wrap="word", state="disabled"); self.csp_text.pack(fill="x", padx=12, pady=(0,12))
    def draw(self):
        self.canvas.delete("all")
        for r in range(R):
            for c in range(C): self.canvas.create_rectangle(c*SZ, r*SZ, (c+1)*SZ, (r+1)*SZ, fill=COLORS.get(self.display_grid[r][c], COLORS[FREE]), outline="#1e1e1e", width=1)
        for (r, c), ch, col, is_circle in [(self.start, "S", "start", True), (self.goal, "G", "goal", False)]:
            x, y = c*SZ + SZ//2, r*SZ + SZ//2
            if is_circle: self.canvas.create_oval(x-9, y-9, x+9, y+9, fill=COLORS[col], outline="white", width=2); self.canvas.create_text(x, y, text=ch, fill="white", font=("Courier", 8, "bold"))
            else: self.canvas.create_polygon([x, y-10, x+8, y+6, x-8, y+6], fill=COLORS[col], outline="white", width=2); self.canvas.create_text(x, y+1, text=ch, fill="white", font=("Courier", 7, "bold"))
    def get_cell(self, e): return (e.y // SZ, e.x // SZ) if (0 <= e.y // SZ < R and 0 <= e.x // SZ < C) else (None, None)
    def on_click(self, e): r, c = self.get_cell(e); self.apply_draw(r, c) if r is not None else None
    def on_drag(self, e): self.on_click(e)
    def apply_draw(self, r, c):
        m = self.draw_mode.get()
        if m == "obstacle" and (r, c) not in (self.start, self.goal): self.env.grid[r][c] = self.display_grid[r][c] = OBSTACLE
        elif m == "erase": self.env.grid[r][c] = self.display_grid[r][c] = FREE
        elif m in ("start", "goal") and self.env.grid[r][c] != OBSTACLE: setattr(self, m, (r, c))
        self.reset_search(keep_env=True)
    def on_algo_change(self): self.reset_search(); self.log(f"// Algorithm: {self.algo_var.get().upper()}", "info")
    def reset_search(self, keep_env=False, keep_gen=False):
        self.stop_anim(); self.path_cells, self.nodes_expanded, self.path_len, self.step_cnt = [], 0, 0, 0
        if not keep_gen: self.search_gen = None
        for r in range(R):
            for c in range(C): self.display_grid[r][c] = OBSTACLE if self.env.grid[r][c] == OBSTACLE else FREE
        self.update_metrics(); self.update_csp("No violations detected."); self.draw()
    def toggle_run(self):
        if self.is_running: self.stop_anim()
        else:
            self.reset_search(keep_env=True); tracemalloc.start(); self.start_time = time.perf_counter()
            self.search_gen = self.search.run(self.algo_var.get(), self.start, self.goal); self.is_running = True
            self.run_btn.config(text="■  STOP", bg="#A32D2D"); self.log(f"// Running {self.algo_var.get().upper()} from {self.start} → {self.goal}", "info"); self.schedule_step()
    def schedule_step(self):
        if self.is_running and self.search_gen:
            self.do_step()
            if self.is_running: self.anim_id = self.root.after(self.speed_ms.get(), self.schedule_step)
    def do_step(self):
        try:
            evt, pos, path, (n_vis, frontier), msg = next(self.search_gen)
            self.nodes_expanded = n_vis
            if evt == "expand":
                self.step_cnt += 1
                self.log(f"Step {self.step_cnt}: Expanding node {pos}")
            elif evt == "step_frontier":
                self.log(f"Frontier: {frontier}\n", "muted")
            else:
                self.log(f"  {msg}", "muted" if evt == "frontier" else "ok" if evt == "found" else None)
            if evt in ("expand", "frontier") and pos and pos != self.start and pos != self.goal: self.display_grid[pos[0]][pos[1]] = VISITED if evt == "expand" else FRONTIER
            elif evt == "found": self.on_found(path)
            elif evt == "fail": self.log("// No path found! Check if goal is reachable.", "warn"); self.stop_anim()
            self.update_metrics(); self.draw()
        except StopIteration: self.stop_anim()
    def stop_anim(self):
        if self.anim_id: self.root.after_cancel(self.anim_id); self.anim_id = None
        self.is_running = False; self.run_btn.config(text="▶  RUN", bg="#185FA5")
    def step_once(self):
        if not self.search_gen:
            self.reset_search(keep_env=True, keep_gen=True); tracemalloc.start(); self.start_time = time.perf_counter()
            self.search_gen = self.search.run(self.algo_var.get(), self.start, self.goal); self.log(f"// Stepping through {self.algo_var.get().upper()}...", "info")
        self.do_step()
    def on_found(self, path):
        self.path_cells, self.path_len = path, len(path) - 1; elapsed = round((time.perf_counter() - self.start_time) * 1000, 1)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        for r, c in path:
            if (r, c) not in (self.start, self.goal): self.display_grid[r][c] = PATH
        viols, exps = self.csp.validate(path, self.start, self.goal, self.env.grid); free = sum(1 for r in range(R) for c in range(C) if self.env.grid[r][c] != OBSTACLE)
        cov = round(len(path) / free * 100) if free else 0
        self.metrics["Runtime"].config(text=f"{elapsed}ms"); self.metrics["Coverage"].config(text=f"{cov}%"); self.metrics["Peak memory"].config(text=f"{peak // 1024}KB")
        next_goal, util_val = self.util.select_next_goal(self.goal, self.env.grid, set(path)); self.metrics["Next utility"].config(text=f"{round(util_val, 3)}" if util_val else "-")
        dirs = [{(-1,0): "North", (1,0): "South", (0,-1): "West", (0,1): "East"}[(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])] for i in range(1, len(path))]
        for msg, tag in [(f"// PATH FOUND ✓  length={self.path_len}  nodes={self.nodes_expanded}  time={elapsed}ms", "ok"), (f"// Path Nodes: {' -> '.join(map(str, path))}", "ok"), (f"// Path Directions: {' -> '.join(dirs)}", "ok"), (f"// Coverage: {cov}%  |  Peak mem: {peak // 1024}KB", "ok")]: self.log(msg, tag)
        if next_goal: self.log(f"// Utility agent → next target {next_goal} (u={round(util_val,3)})", "info")
        self.update_csp("\n".join(viols if viols else exps))
        for v in viols: self.log(f"  CSP: {v}", "error")
        self.stop_anim()
    def clear_all(self): self.env.grid = [[FREE]*C for _ in range(R)]; self.start, self.goal = (0, 0), (R-1, C-1); self.reset_search(); self.log("// Grid cleared.", "info")
    def load_example(self): self.env = Env(); self.start, self.goal = (0, 0), (R-1, C-1); self.reset_search(); self.log("// Example maze loaded. Run an algorithm.", "ok")
    def log(self, msg, tag=None): self.trace.config(state="normal"); self.trace.insert("end", msg + "\n", tag); self.trace.see("end"); self.trace.config(state="disabled"); self.trace.update()
    def update_csp(self, text): self.csp_text.config(state="normal"); self.csp_text.delete("1.0", "end"); self.csp_text.insert("end", text); self.csp_text.config(state="disabled")
    def update_metrics(self):
        free = sum(1 for r in range(R) for c in range(C) if self.env.grid[r][c] != OBSTACLE)
        self.metrics["Nodes expanded"].config(text=str(self.nodes_expanded)); self.metrics["Path length"].config(text=str(self.path_len)); self.metrics["Coverage"].config(text=f"{round(len(self.path_cells)/free*100) if free and self.path_cells else 0}%")
if __name__ == "__main__":
    root = tk.Tk(); app = CleaningPlannerGUI(root); root.mainloop()
