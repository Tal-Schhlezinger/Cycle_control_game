"""Tkinter UI for Cycle Control — analytical, not pretty.

Usage:
    python ui.py           # radius-3 default
    python ui.py 2         # radius-2 board
"""

from __future__ import annotations

import math
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cycle_control.engine import MoveEngine, MoveError
from cycle_control.persistence import load_from_file, save_to_file
from cycle_control.rules import RulesConfig
from cycle_control.scoring import scoring_nodes
from cycle_control.state import NodeState, Player, TurnPhase
from cycle_control.topology import BoardTopology

CELL_SIZE = 32


def triangle_to_pixels(q: int, r: int, o: int, cell_size: float, ox: float, oy: float):
    """Return the 3 pixel vertices of triangle (q, r, o).

    Axial basis: e_q = (1, 0), e_r = (0.5, -sqrt(3)/2) (y flipped for screen).
    """
    s32 = math.sqrt(3) / 2

    def vx(a: int, b: int) -> float:
        return ox + cell_size * (a + 0.5 * b)

    def vy(a: int, b: int) -> float:
        return oy - cell_size * (s32 * b)

    if o == 0:
        verts = [(q, r), (q + 1, r), (q, r + 1)]
    else:
        verts = [(q, r + 1), (q + 1, r + 1), (q + 1, r)]
    return [(vx(a, b), vy(a, b)) for a, b in verts]


def _point_in_triangle(px: float, py: float, verts) -> bool:
    (x1, y1), (x2, y2), (x3, y3) = verts
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if denom == 0:
        return False
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    c = 1 - a - b
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1


class CycleControlUI:
    def __init__(self, root: tk.Tk, radius: int = 3):
        self.root = root
        self.root.title(f"Cycle Control  (R={radius})")

        self.rules = RulesConfig(board_radius=radius)
        self.topology = BoardTopology(radius)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

        self.sandbox_mode = tk.BooleanVar(value=False)
        self.highlight_scoring = tk.BooleanVar(value=True)
        self.show_coords = tk.BooleanVar(value=False)

        # Experimental balance modes (checkbox states; not applied until
        # "Apply & Restart" is pressed).
        self.neutrality_var = tk.BooleanVar(value=self.rules.neutrality_rule)
        self.strict_adjacency_var = tk.BooleanVar(value=self.rules.strict_adjacency_rule)
        self.mirror_adjacency_var = tk.BooleanVar(value=self.rules.mirror_adjacency)
        self.partial_credit_on_var = tk.BooleanVar(value=self.rules.partial_credit_k > 0)
        self.partial_credit_k_var = tk.IntVar(
            value=self.rules.partial_credit_k if self.rules.partial_credit_k > 0 else 3
        )

        self._build_ui()
        self._redraw()

    # ---------- layout ----------

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=6)
        root_frame.pack(fill="both", expand=True)

        canvas_w, canvas_h = 640, 640
        self.canvas = tk.Canvas(root_frame, width=canvas_w, height=canvas_h,
                                bg="white", highlightthickness=1,
                                highlightbackground="#888")
        self.canvas.grid(row=0, column=0, rowspan=20, sticky="nw")
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        side = ttk.Frame(root_frame, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="nw")

        def lbl(text, row, font=("TkDefaultFont", 10)):
            v = tk.StringVar(value=text)
            ttk.Label(side, textvariable=v, font=font, anchor="w",
                      justify="left").grid(row=row, column=0, sticky="w", pady=1)
            return v

        self.status_var = lbl("", 0, ("TkDefaultFont", 12, "bold"))
        self.score_var = lbl("", 1, ("TkDefaultFont", 11, "bold"))
        self.phase_var = lbl("", 2)
        self.turn_var = lbl("", 3)
        self.pass_var = lbl("", 4)
        self.supply_var = lbl("", 5)
        self.history_var = lbl("", 6)

        ttk.Separator(side, orient="horizontal").grid(row=7, column=0, sticky="we", pady=6)

        ttk.Button(side, text="Pass", command=self._on_pass).grid(row=8, column=0, sticky="we", pady=1)
        ttk.Button(side, text="Undo", command=self._on_undo).grid(row=9, column=0, sticky="we", pady=1)
        ttk.Button(side, text="Redo", command=self._on_redo).grid(row=10, column=0, sticky="we", pady=1)
        ttk.Button(side, text="Restart", command=self._on_restart).grid(row=11, column=0, sticky="we", pady=1)
        ttk.Button(side, text="Save...", command=self._on_save).grid(row=12, column=0, sticky="we", pady=1)
        ttk.Button(side, text="Load...", command=self._on_load).grid(row=13, column=0, sticky="we", pady=1)

        ttk.Separator(side, orient="horizontal").grid(row=14, column=0, sticky="we", pady=6)

        ttk.Checkbutton(side, text="Sandbox mode", variable=self.sandbox_mode,
                        command=self._redraw).grid(row=15, column=0, sticky="w")
        ttk.Checkbutton(side, text="Highlight scoring", variable=self.highlight_scoring,
                        command=self._redraw).grid(row=16, column=0, sticky="w")
        ttk.Checkbutton(side, text="Show coords", variable=self.show_coords,
                        command=self._redraw).grid(row=17, column=0, sticky="w")

        ttk.Separator(side, orient="horizontal").grid(row=18, column=0, sticky="we", pady=6)
        ttk.Label(side, text="Balance modes", font=("TkDefaultFont", 10, "bold")
                  ).grid(row=19, column=0, sticky="w")

        ttk.Checkbutton(side, text="Neutrality rule",
                        variable=self.neutrality_var
                        ).grid(row=20, column=0, sticky="w")
        ttk.Checkbutton(side, text="Strict adjacency",
                        variable=self.strict_adjacency_var
                        ).grid(row=21, column=0, sticky="w")
        ttk.Checkbutton(side, text="Mirror-vertex adjacency",
                        variable=self.mirror_adjacency_var
                        ).grid(row=22, column=0, sticky="w")

        pc_frame = ttk.Frame(side)
        pc_frame.grid(row=23, column=0, sticky="w")
        ttk.Checkbutton(pc_frame, text="Partial credit  K=",
                        variable=self.partial_credit_on_var
                        ).grid(row=0, column=0, sticky="w")
        ttk.Spinbox(pc_frame, from_=2, to=99, width=3,
                    textvariable=self.partial_credit_k_var
                    ).grid(row=0, column=1, padx=(0, 0))

        ttk.Button(side, text="Apply & Restart",
                   command=self._on_apply_modes
                   ).grid(row=24, column=0, sticky="we", pady=(4, 0))

        ttk.Separator(side, orient="horizontal").grid(row=25, column=0, sticky="we", pady=6)

        help_text = (
            "Normal:  left-click = place\n"
            "Sandbox: left = cycle B/W/empty\n"
            "         right = force empty\n"
            "\n"
            "Red   = Black stone\n"
            "Blue  = White stone\n"
            "White dot = Black in cycle\n"
            "Black dot = White in cycle\n"
        )
        ttk.Label(side, text=help_text, justify="left", foreground="#555"
                  ).grid(row=26, column=0, sticky="w", pady=2)

    # ---------- rendering ----------

    def _redraw(self):
        self.canvas.delete("all")
        cw = int(self.canvas["width"])
        ch = int(self.canvas["height"])
        ox, oy = cw / 2, ch / 2

        # Center board: adjust origin based on board extents.
        nodes = list(self.topology.iterate_nodes())
        all_pix = []
        for n in nodes:
            all_pix.extend(triangle_to_pixels(n[0], n[1], n[2], CELL_SIZE, ox, oy))
        if all_pix:
            xs = [p[0] for p in all_pix]
            ys = [p[1] for p in all_pix]
            dx = ox - (min(xs) + max(xs)) / 2
            dy = oy - (min(ys) + max(ys)) / 2
            ox += dx
            oy += dy

        k = self.rules.partial_credit_k
        scoring_b = scoring_nodes(self.topology, self.state.board, Player.BLACK, k)
        scoring_w = scoring_nodes(self.topology, self.state.board, Player.WHITE, k)

        for node in nodes:
            verts = triangle_to_pixels(node[0], node[1], node[2], CELL_SIZE, ox, oy)
            st = self.state.board.get(node, NodeState.EMPTY)
            if st == NodeState.BLACK:
                fill = "#c0392b"   # red
                text_color = "#fff"
            elif st == NodeState.WHITE:
                fill = "#2471a3"   # blue
                text_color = "#fff"
            else:
                if self.sandbox_mode.get():
                    fill = "#f4f0d8"
                else:
                    fill = "#eaeaea"
                text_color = "#666"

            self.canvas.create_polygon(
                *[c for v in verts for c in v],
                fill=fill, outline="#333", width=1,
            )
            cx = sum(v[0] for v in verts) / 3
            cy = sum(v[1] for v in verts) / 3

            if self.highlight_scoring.get():
                if node in scoring_b:
                    self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                            fill="white", outline="")
                elif node in scoring_w:
                    self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                            fill="black", outline="")
            if self.show_coords.get():
                self.canvas.create_text(cx, cy, text=f"{node[0]},{node[1]},{node[2]}",
                                        fill=text_color, font=("TkSmallCaptionFont", 7))

        self._update_status(len(scoring_b), len(scoring_w))

    def _update_status(self, b_score: int, w_score: int):
        if self.state.game_over:
            if self.state.winner == "draw":
                self.status_var.set("GAME OVER — DRAW")
            elif self.state.winner is not None:
                winner = self.state.winner.value.upper()  # type: ignore[union-attr]
                self.status_var.set(f"GAME OVER — {winner} WINS")
            else:
                self.status_var.set("GAME OVER")
        else:
            self.status_var.set(f"Active: {self.state.active_player.value.upper()}")
        self.score_var.set(f"Score  B={b_score}  W={w_score}")
        self.phase_var.set(f"Phase: {self.state.turn_phase.value}")
        self.turn_var.set(f"Turn: {self.state.current_turn}")
        self.pass_var.set(f"Pass counter: {self.state.consecutive_pass_count}")
        if self.rules.supply_enabled():
            b = self.state.stones_remaining.get(Player.BLACK, 0)
            w = self.state.stones_remaining.get(Player.WHITE, 0)
            self.supply_var.set(f"Supply  B={b}  W={w}")
        else:
            self.supply_var.set("Supply: unlimited")
        self.history_var.set(f"Moves: {self.state.move_count()}")

    # ---------- hit-testing ----------

    def _find_clicked_node(self, x: float, y: float):
        cw = int(self.canvas["width"])
        ch = int(self.canvas["height"])
        ox, oy = cw / 2, ch / 2
        nodes = list(self.topology.iterate_nodes())
        all_pix = []
        for n in nodes:
            all_pix.extend(triangle_to_pixels(n[0], n[1], n[2], CELL_SIZE, ox, oy))
        if all_pix:
            xs = [p[0] for p in all_pix]
            ys = [p[1] for p in all_pix]
            dx = ox - (min(xs) + max(xs)) / 2
            dy = oy - (min(ys) + max(ys)) / 2
            ox += dx
            oy += dy
        for n in nodes:
            verts = triangle_to_pixels(n[0], n[1], n[2], CELL_SIZE, ox, oy)
            if _point_in_triangle(x, y, verts):
                return n
        return None

    # ---------- event handlers ----------

    def _on_left_click(self, event):
        node = self._find_clicked_node(event.x, event.y)
        if node is None:
            return
        if self.sandbox_mode.get():
            current = self.state.board.get(node, NodeState.EMPTY)
            try:
                if current == NodeState.EMPTY:
                    self.engine.sandbox_place(self.state, node, NodeState.BLACK)
                elif current == NodeState.BLACK:
                    self.engine.sandbox_place(self.state, node, NodeState.WHITE)
                else:
                    self.engine.sandbox_remove(self.state, node)
            except MoveError as e:
                messagebox.showwarning("Sandbox error", str(e))
                return
            self._redraw()
        else:
            try:
                self.engine.apply_placement(self.state, node)
            except MoveError as e:
                messagebox.showwarning("Illegal move", str(e))
                return
            self._redraw()

    def _on_right_click(self, event):
        if not self.sandbox_mode.get():
            return
        node = self._find_clicked_node(event.x, event.y)
        if node is None:
            return
        try:
            self.engine.sandbox_remove(self.state, node)
        except MoveError as e:
            messagebox.showwarning("Sandbox error", str(e))
            return
        self._redraw()

    def _on_pass(self):
        try:
            self.engine.apply_pass(self.state)
        except MoveError as e:
            messagebox.showwarning("Cannot pass", str(e))
            return
        self._redraw()

    def _on_undo(self):
        try:
            self.engine.undo(self.state)
        except MoveError as e:
            messagebox.showwarning("Undo failed", str(e))
            return
        self._redraw()

    def _on_redo(self):
        try:
            self.engine.redo(self.state)
        except MoveError as e:
            messagebox.showwarning("Redo failed", str(e))
            return
        self._redraw()

    def _on_apply_modes(self):
        """Rebuild engine with currently-selected balance modes and restart."""
        if self.state.move_history and not messagebox.askyesno(
            "Apply modes",
            "Applying modes will restart the current game. Continue?",
        ):
            return
        k = int(self.partial_credit_k_var.get()) if self.partial_credit_on_var.get() else 0
        try:
            new_rules = RulesConfig(
                board_radius=self.rules.board_radius,
                stones_per_player=self.rules.stones_per_player,
                pass_enabled=self.rules.pass_enabled,
                end_on_consecutive_passes=self.rules.end_on_consecutive_passes,
                end_on_all_stones_placed=self.rules.end_on_all_stones_placed,
                end_on_board_full=self.rules.end_on_board_full,
                neutrality_rule=self.neutrality_var.get(),
                strict_adjacency_rule=self.strict_adjacency_var.get(),
                mirror_adjacency=self.mirror_adjacency_var.get(),
                partial_credit_k=k,
            )
            new_topology = BoardTopology(
                new_rules.board_radius,
                mirror_adjacency=new_rules.mirror_adjacency,
            )
        except ValueError as e:
            messagebox.showerror("Invalid rules", str(e))
            return
        self.rules = new_rules
        self.topology = new_topology
        self.engine = MoveEngine(new_rules, new_topology)
        self.state = self.engine.initial_state()
        self._redraw()

    def _sync_mode_vars_from_rules(self):
        """After loading a save, reflect the loaded rules in the checkboxes."""
        self.neutrality_var.set(self.rules.neutrality_rule)
        self.strict_adjacency_var.set(self.rules.strict_adjacency_rule)
        self.mirror_adjacency_var.set(self.rules.mirror_adjacency)
        self.partial_credit_on_var.set(self.rules.partial_credit_k > 0)
        if self.rules.partial_credit_k > 0:
            self.partial_credit_k_var.set(self.rules.partial_credit_k)

    def _on_restart(self):
        if not messagebox.askyesno("Restart", "Discard current game and restart?"):
            return
        self.state = self.engine.initial_state()
        self._redraw()

    def _on_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            save_to_file(path, self.state, self.rules)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _on_load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            rules, state = load_from_file(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.rules = rules
        self.topology = BoardTopology(
            rules.board_radius, mirror_adjacency=rules.mirror_adjacency,
        )
        self.engine = MoveEngine(rules, self.topology)
        self.state = state
        self._sync_mode_vars_from_rules()
        self.root.title(f"Cycle Control  (R={rules.board_radius})")
        self._redraw()


def main():
    radius = 3
    if len(sys.argv) > 1:
        try:
            radius = int(sys.argv[1])
        except ValueError:
            print(f"ignoring non-integer radius argument: {sys.argv[1]!r}")
    root = tk.Tk()
    CycleControlUI(root, radius=radius)
    root.mainloop()


if __name__ == "__main__":
    main()
