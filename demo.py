#!/usr/bin/env python3
"""
===============================================================================
 SIMULATION OF DATA LINK LAYER PROTOCOLS
===============================================================================
 Protocols implemented
   (a) Stop-and-Wait ARQ            : W_send = 1, W_recv = 1
   (b) Sliding Window (generic)     : W_send = N, W_recv = 1, pipelined
   (c) Go-Back-N ARQ                : W_send = N, W_recv = 1, cumulative ACK
   (d) Selective Repeat ARQ         : W_send = N, W_recv = N, individual ACK

 Every protocol runs both WITHOUT errors (set all error sliders to 0) and
 WITH errors (random loss/corruption via sliders, or manual injection by
 clicking a frame in flight).

 The GUI draws a space-time (ladder) diagram: time flows downward, the sender
 is the left vertical line, the receiver is the right vertical line, and each
 frame is an arrow travelling across the channel.

 Requires only the Python standard library.  Run:  python3 dll_protocol_simulator.py
===============================================================================
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict

# ---------------------------------------------------------------------------
# Protocol identifiers
# ---------------------------------------------------------------------------
STOP_WAIT = "Stop-and-Wait"
SLIDING   = "Sliding Window (generic)"
GO_BACK_N = "Go-Back-N"
SEL_REPEAT = "Selective Repeat"

PROTOCOLS = [STOP_WAIT, SLIDING, GO_BACK_N, SEL_REPEAT]

# Protocols that use a single timer for the oldest unacknowledged frame and
# cumulative acknowledgements.  Selective Repeat is the odd one out.
CUMULATIVE = {STOP_WAIT, SLIDING, GO_BACK_N}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    protocol: str = STOP_WAIT
    window: int = 4          # N (ignored for Stop-and-Wait, which forces 1)
    total_frames: int = 10   # how many frames the network layer wants sent
    prop_delay: int = 6      # one-way propagation delay, in ticks
    timeout: int = 20        # retransmission timer, in ticks
    p_data_loss: float = 0.0
    p_ack_loss: float = 0.0
    p_corrupt: float = 0.0   # data frame arrives but fails the CRC check
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# A frame (or ACK) travelling across the channel
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    fid: int
    kind: str                       # 'data' or 'ack'
    seq: int                        # ABSOLUTE sequence number (see note below)
    depart: int                     # tick it left the sender/receiver
    arrive: int                     # tick it would reach the other end
    lost: bool = False
    corrupt: bool = False
    retx: bool = False              # is this a retransmission?
    die_frac: Optional[float] = None  # 0..1 point along the link where it dies
    note: str = ""

    def in_flight(self, t: int) -> bool:
        return self.depart <= t <= self.arrive


# ---------------------------------------------------------------------------
# The simulator core  (pure logic - no GUI code in here)
# ---------------------------------------------------------------------------
#
# NOTE ON SEQUENCE NUMBERS
# ------------------------
# Internally every frame is tracked by an ABSOLUTE index 0,1,2,3,...  This
# avoids all the painful modulo-arithmetic bugs in window comparisons.
# What the *protocol* would actually put in the header is  absolute % M,
# where M is the sequence-number space, and that is what the GUI displays.
#   Stop-and-Wait  : M = 2          (the classic 1-bit sequence number)
#   Sliding / GBN  : M = N + 1
#   Selective Rep. : M = 2N
# ---------------------------------------------------------------------------
class Simulator:

    MAX_TICKS = 4000     # safety stop, in case the error rate makes it endless

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    # ---------------- setup ----------------
    def reset(self):
        c = self.cfg
        self.rng = random.Random(c.seed)
        self.t = 0
        self.fid_counter = 0
        self.frames: List[Frame] = []

        # --- sender state ---
        self.base = 0             # oldest unacknowledged frame
        self.next_seq = 0         # next frame to be sent for the first time
        self.acked = set()        # Selective Repeat: individually ACKed frames
        self.timer: Optional[int] = None      # single timer (cumulative protos)
        self.timers: Dict[int, int] = {}      # per-frame timers (Sel. Repeat)
        self.retx_queue: deque = deque()      # frames waiting to be resent

        # --- receiver state ---
        self.rcv_expected = 0     # R_base: next in-order frame wanted
        self.rcv_buffer = set()   # Selective Repeat: out-of-order frames held

        # --- bookkeeping ---
        self.delivered: List[int] = []
        self.timeout_events: List[tuple] = []
        self.log_lines: List[tuple] = []
        self.finished = False
        self.arm_drop_data = False
        self.arm_drop_ack = False

        self.stats = dict(data_tx=0, retx=0, ack_tx=0,
                          data_lost=0, ack_lost=0, corrupted=0, timeouts=0)

        self.log("Simulation reset.  Protocol = %s,  N = %d,  M = %d"
                 % (c.protocol, self.W, self.M), "hdr")

    # ---------------- derived parameters ----------------
    @property
    def W(self) -> int:
        """Sender window size."""
        return 1 if self.cfg.protocol == STOP_WAIT else self.cfg.window

    @property
    def RW(self) -> int:
        """Receiver window size."""
        return self.W if self.cfg.protocol == SEL_REPEAT else 1

    @property
    def M(self) -> int:
        """Sequence-number space."""
        p = self.cfg.protocol
        if p == STOP_WAIT:
            return 2
        if p == SEL_REPEAT:
            return 2 * self.W
        return self.W + 1

    def s(self, abs_seq: int) -> int:
        """Absolute index -> the number that would sit in the frame header."""
        return abs_seq % self.M

    # ---------------- logging ----------------
    def log(self, msg: str, tag: str = ""):
        self.log_lines.append((self.t, msg, tag))

    # ---------------- channel ----------------
    def _new_frame(self, kind, seq, lost, corrupt, retx, reason):
        self.fid_counter += 1
        f = Frame(fid=self.fid_counter, kind=kind, seq=seq,
                  depart=self.t, arrive=self.t + self.cfg.prop_delay,
                  lost=lost, corrupt=corrupt, retx=retx,
                  die_frac=0.5 if lost else None, note=reason)
        self.frames.append(f)
        return f

    def send_data(self, seq: int, retx: bool):
        c = self.cfg
        lost, reason = False, ""
        if self.arm_drop_data:
            lost, reason = True, "manually dropped"
            self.arm_drop_data = False
        elif self.rng.random() < c.p_data_loss:
            lost, reason = True, "lost in channel"
        corrupt = (not lost) and (self.rng.random() < c.p_corrupt)

        self._new_frame("data", seq, lost, corrupt, retx, reason)
        self.stats["data_tx"] += 1
        if retx:
            self.stats["retx"] += 1
        if lost:
            self.stats["data_lost"] += 1

        kind = "RETRANSMIT" if retx else "send"
        extra = ""
        if lost:
            extra = "   ->  X %s" % reason
        elif corrupt:
            extra = "   ->  will arrive CORRUPTED"
        self.log("S: %s DATA seq=%d (frame #%d)%s"
                 % (kind, self.s(seq), seq, extra),
                 "retx" if retx else "data")

    def send_ack(self, value: int, label: str):
        c = self.cfg
        lost, reason = False, ""
        if self.arm_drop_ack:
            lost, reason = True, "manually dropped"
            self.arm_drop_ack = False
        elif self.rng.random() < c.p_ack_loss:
            lost, reason = True, "lost in channel"

        self._new_frame("ack", value, lost, False, False, reason)
        self.stats["ack_tx"] += 1
        if lost:
            self.stats["ack_lost"] += 1
        self.log("R: send %s%s" % (label, ("   ->  X " + reason) if lost else ""),
                 "ack")

    # ---------------- receiver ----------------
    def on_data_arrive(self, f: Frame):
        p = self.cfg.protocol

        if f.corrupt:
            self.stats["corrupted"] += 1
            self.log("R: DATA seq=%d fails CRC  ->  discarded" % self.s(f.seq),
                     "err")
            if p in CUMULATIVE:
                # Re-send the last cumulative ACK so the sender is not left
                # completely in the dark (a "duplicate ACK").
                self.send_ack(self.rcv_expected,
                              "ACK %d (duplicate, re-ack)" % self.s(self.rcv_expected))
            return

        if p in CUMULATIVE:
            if f.seq == self.rcv_expected:
                self.rcv_expected += 1
                self.delivered.append(f.seq)
                self.log("R: DATA seq=%d accepted  ->  delivered to network layer"
                         % self.s(f.seq), "ok")
            elif f.seq < self.rcv_expected:
                self.log("R: DATA seq=%d is a DUPLICATE (already delivered)  ->  "
                         "discarded, not passed up twice" % self.s(f.seq), "err")
            else:
                self.log("R: DATA seq=%d out of order (expected %d)  ->  discarded"
                         % (self.s(f.seq), self.s(self.rcv_expected)), "err")
            self.send_ack(self.rcv_expected,
                          "ACK %d  (= next frame expected)" % self.s(self.rcv_expected))

        else:  # Selective Repeat
            rbase = self.rcv_expected
            if rbase <= f.seq < rbase + self.RW:
                if f.seq in self.rcv_buffer or f.seq < rbase:
                    self.log("R: DATA seq=%d is a duplicate  ->  re-ACK"
                             % self.s(f.seq), "err")
                else:
                    self.rcv_buffer.add(f.seq)
                    if f.seq == rbase:
                        self.log("R: DATA seq=%d accepted (in order)"
                                 % self.s(f.seq), "ok")
                    else:
                        self.log("R: DATA seq=%d accepted OUT OF ORDER  ->  buffered"
                                 % self.s(f.seq), "ok")
                    # slide the receiver window over every contiguous frame
                    while self.rcv_expected in self.rcv_buffer:
                        self.rcv_buffer.discard(self.rcv_expected)
                        self.delivered.append(self.rcv_expected)
                        self.rcv_expected += 1
                self.send_ack(f.seq, "ACK %d  (individual)" % self.s(f.seq))

            elif rbase - self.RW <= f.seq < rbase:
                self.log("R: DATA seq=%d already delivered  ->  re-ACK"
                         % self.s(f.seq), "err")
                self.send_ack(f.seq, "ACK %d  (duplicate)" % self.s(f.seq))
            else:
                self.log("R: DATA seq=%d outside receiver window  ->  ignored"
                         % self.s(f.seq), "err")

    # ---------------- sender: handling ACKs ----------------
    def on_ack_arrive(self, f: Frame):
        if self.cfg.protocol in CUMULATIVE:
            if f.seq > self.base:
                old = self.base
                self.base = f.seq
                self.log("S: ACK %d received  ->  window slides, base %d -> %d"
                         % (self.s(f.seq), old, self.base), "ok")
                # restart or cancel the single timer
                self.timer = None if self.base >= self.next_seq \
                    else self.t + self.cfg.timeout
                # anything already ACKed no longer needs retransmitting
                self.retx_queue = deque(x for x in self.retx_queue if x >= self.base)
            else:
                self.log("S: ACK %d is old/duplicate  ->  ignored" % self.s(f.seq),
                         "err")
        else:
            if f.seq in self.acked:
                self.log("S: ACK %d duplicate  ->  ignored" % self.s(f.seq), "err")
                return
            self.acked.add(f.seq)
            self.timers.pop(f.seq, None)
            self.retx_queue = deque(x for x in self.retx_queue if x != f.seq)
            if f.seq == self.base:
                old = self.base
                while self.base in self.acked:
                    self.base += 1
                self.log("S: ACK %d received  ->  window slides, base %d -> %d"
                         % (self.s(f.seq), old, self.base), "ok")
            else:
                self.log("S: ACK %d received  ->  frame marked ACKed "
                         "(window does not slide yet)" % self.s(f.seq), "ok")

    # ---------------- sender: timers ----------------
    def check_timeouts(self):
        c = self.cfg
        if c.protocol in CUMULATIVE:
            if self.timer is not None and self.t >= self.timer:
                self.stats["timeouts"] += 1
                self.timeout_events.append((self.t, self.base))
                if c.protocol == STOP_WAIT:
                    self.log("*** TIMEOUT for frame %d  ->  resend it"
                             % self.s(self.base), "to")
                    self._queue_retx([self.base])
                else:
                    span = list(range(self.base, self.next_seq))
                    self.log("*** TIMEOUT for frame %d  ->  GO BACK N: resend "
                             "frames %s" % (self.s(self.base),
                                            [self.s(x) for x in span]), "to")
                    self._queue_retx(span)
                self.timer = self.t + c.timeout       # restart
        else:
            for seq in sorted(self.timers):
                if self.t >= self.timers[seq] and seq not in self.acked:
                    self.stats["timeouts"] += 1
                    self.timeout_events.append((self.t, seq))
                    self.log("*** TIMEOUT for frame %d  ->  resend ONLY that frame"
                             % self.s(seq), "to")
                    self._queue_retx([seq])
                    self.timers[seq] = self.t + c.timeout   # restart

    def _queue_retx(self, seqs):
        for s in seqs:
            if s not in self.retx_queue:
                self.retx_queue.append(s)

    # ---------------- sender: transmit one frame per tick ----------------
    def try_send(self):
        # Transmission time is modelled as 1 tick, so at most one frame can
        # leave the sender per tick.  This is what produces the staircase
        # pattern you see in pipelined protocols.
        if self.retx_queue:
            seq = self.retx_queue.popleft()
            if self.cfg.protocol == SEL_REPEAT and seq in self.acked:
                return
            self.send_data(seq, retx=True)
            if self.cfg.protocol == SEL_REPEAT:
                self.timers[seq] = self.t + self.cfg.timeout
            return

        if self.next_seq < self.base + self.W and self.next_seq < self.cfg.total_frames:
            seq = self.next_seq
            self.send_data(seq, retx=False)
            if self.cfg.protocol in CUMULATIVE:
                if self.timer is None:
                    self.timer = self.t + self.cfg.timeout
            else:
                self.timers[seq] = self.t + self.cfg.timeout
            self.next_seq += 1

    # ---------------- one tick of simulated time ----------------
    def step(self):
        if self.finished:
            return
        self.t += 1

        # 1. deliver anything that arrives at this instant
        for f in self.frames:
            if f.arrive == self.t and not f.lost:
                if f.kind == "data":
                    self.on_data_arrive(f)
                else:
                    self.on_ack_arrive(f)

        # 2. expire timers
        self.check_timeouts()

        # 3. transmit
        self.try_send()

        # 4. are we done?
        if len(self.delivered) >= self.cfg.total_frames:
            if not any(f.in_flight(self.t) for f in self.frames):
                self.finished = True
                self.log("=== All %d frames delivered in order.  Total time = %d "
                         "ticks ===" % (self.cfg.total_frames, self.t), "hdr")
        if self.t >= self.MAX_TICKS:
            self.finished = True
            self.log("=== Stopped: tick limit reached (error rate too high) ===",
                     "hdr")

    # ---------------- metrics for the report ----------------
    def metrics(self):
        st = self.stats
        total = self.cfg.total_frames
        eff = (total / st["data_tx"] * 100) if st["data_tx"] else 0.0
        thr = (len(self.delivered) / self.t) if self.t else 0.0
        return dict(time=self.t,
                    delivered=len(self.delivered),
                    data_tx=st["data_tx"],
                    retx=st["retx"],
                    ack_tx=st["ack_tx"],
                    lost=st["data_lost"] + st["ack_lost"],
                    corrupted=st["corrupted"],
                    timeouts=st["timeouts"],
                    efficiency=eff,
                    throughput=thr)


# ===========================================================================
#  GUI
# ===========================================================================
def build_gui():
    import tkinter as tk
    from tkinter import ttk, font as tkfont

    # ---- palette -----------------------------------------------------------
    BG      = "#0f172a"
    PANEL   = "#1e293b"
    INK     = "#e2e8f0"
    MUTED   = "#94a3b8"
    C_DATA  = "#38bdf8"
    C_RETX  = "#fb923c"
    C_ACK   = "#4ade80"
    C_LOST  = "#f87171"
    C_TO    = "#fbbf24"
    GRID    = "#334155"

    SX, RX = 120, 500            # x of sender / receiver timelines
    TOP    = 34                  # y of tick 0
    PPT    = 7                   # pixels per tick

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Data Link Layer Protocol Simulator")
            root.configure(bg=BG)

            self.cfg = Config()
            self.sim = Simulator(self.cfg)
            self.running = False
            self.job = None
            self.scroll = 0
            self.mono = tkfont.Font(family="Courier", size=9)

            style = ttk.Style()
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("TCombobox", fieldbackground=PANEL, background=PANEL)

            self._build_controls()
            self._build_canvas()
            self._build_side()
            self.on_protocol_change()
            self.redraw()

        # ---------------- left control panel ----------------
        def _build_controls(self):
            p = tk.Frame(self.root, bg=PANEL, padx=10, pady=10)
            p.grid(row=0, column=0, sticky="ns")

            def head(txt):
                tk.Label(p, text=txt, bg=PANEL, fg=C_DATA,
                         font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(6, 1))

            def spin(label, frm, to, init):
                tk.Label(p, text=label, bg=PANEL, fg=INK,
                         font=("Helvetica", 8)).pack(anchor="w")
                v = tk.IntVar(value=init)
                s = tk.Spinbox(p, from_=frm, to=to, textvariable=v, width=8,
                               bg=BG, fg=INK, buttonbackground=PANEL,
                               insertbackground=INK, relief="flat",
                               command=self.on_config_change)
                s.pack(anchor="w", pady=(0, 2))
                # commit a TYPED value too - not just arrow clicks / Enter.
                s.bind("<Return>", lambda e: self.on_config_change())
                s.bind("<FocusOut>", lambda e: self.on_config_change())
                v.trace_add("write", lambda *a: self.on_config_change())
                return v, s

            def slider(label, init):
                self.lbl = tk.Label(p, text=label, bg=PANEL, fg=INK,
                                    font=("Helvetica", 8))
                self.lbl.pack(anchor="w")
                v = tk.IntVar(value=init)
                tk.Scale(p, from_=0, to=100, orient="horizontal", variable=v,
                         bg=PANEL, fg=INK, troughcolor=BG, highlightthickness=0,
                         length=170, relief="flat", sliderlength=16, width=11,
                         command=lambda e: self.on_config_change()).pack(anchor="w")
                return v

            tk.Label(p, text="PROTOCOL", bg=PANEL, fg=C_DATA,
                     font=("Helvetica", 10, "bold")).pack(anchor="w")
            self.proto = tk.StringVar(value=STOP_WAIT)
            cb = ttk.Combobox(p, textvariable=self.proto, values=PROTOCOLS,
                              state="readonly", width=22)
            cb.pack(anchor="w", pady=(2, 6))
            cb.bind("<<ComboboxSelected>>", lambda e: self.on_protocol_change())

            head("LINK PARAMETERS")
            self.v_win,  self.sp_win = spin("Window size  N", 1, 8, 4)
            self.v_tot,  _ = spin("Frames to send", 3, 18, 10)
            self.v_prop, _ = spin("Propagation delay (ticks)", 2, 20, 6)
            self.v_to,   _ = spin("Timeout (ticks)", 5, 80, 20)

            head("ERROR INJECTION")
            self.v_dl = slider("Data frame loss  %", 0)
            self.v_al = slider("ACK loss  %", 0)
            self.v_cr = slider("Data corruption  %", 0)

            tk.Button(p, text="Drop next DATA frame", command=self.drop_data,
                      bg="#7f1d1d", fg="white", relief="flat",
                      font=("Helvetica", 8, "bold")).pack(fill="x", pady=(4, 1))
            tk.Button(p, text="Drop next ACK", command=self.drop_ack,
                      bg="#7f1d1d", fg="white", relief="flat",
                      font=("Helvetica", 8, "bold")).pack(fill="x", pady=1)
            tk.Label(p, text="...or click any arrow in flight\nto kill it mid-channel",
                     bg=PANEL, fg=MUTED, font=("Helvetica", 7),
                     justify="left").pack(anchor="w")

            head("PLAYBACK")
            tk.Label(p, text="ms per tick", bg=PANEL, fg=INK,
                     font=("Helvetica", 8)).pack(anchor="w")
            self.v_speed = tk.IntVar(value=80)
            tk.Scale(p, from_=10, to=250, orient="horizontal",
                     variable=self.v_speed, bg=PANEL, fg=INK, troughcolor=BG,
                     highlightthickness=0, length=170, relief="flat",
                     sliderlength=16, width=11).pack(anchor="w")

            row = tk.Frame(p, bg=PANEL)
            row.pack(fill="x", pady=4)
            self.btn_play = tk.Button(row, text="\u25b6  Play", width=8,
                                      command=self.toggle, bg="#166534",
                                      fg="white", relief="flat",
                                      font=("Helvetica", 9, "bold"))
            self.btn_play.pack(side="left", padx=2)
            tk.Button(row, text="Step", width=5, command=self.one_step,
                      bg="#1e40af", fg="white", relief="flat").pack(side="left", padx=2)
            tk.Button(row, text="Reset", width=5, command=self.reset,
                      bg="#475569", fg="white", relief="flat").pack(side="left", padx=2)

            head("PRESETS")
            g = tk.Frame(p, bg=PANEL)
            g.pack(fill="x")
            presets = [("No errors", 0, 0, 0), ("Data loss 20%", 20, 0, 0),
                       ("ACK loss 25%", 0, 25, 0), ("Noisy 15%", 15, 15, 15)]
            for i, (txt, d, a, cr) in enumerate(presets):
                tk.Button(g, text=txt, font=("Helvetica", 8),
                          command=lambda d=d, a=a, cr=cr: self.preset(d, a, cr),
                          bg=BG, fg=INK, relief="flat", width=11).grid(
                              row=i // 2, column=i % 2, padx=1, pady=1, sticky="ew")

        # ---------------- centre: windows + ladder ----------------
        def _build_canvas(self):
            mid = tk.Frame(self.root, bg=BG)
            mid.grid(row=0, column=1, sticky="nsew")
            self.win_cv = tk.Canvas(mid, width=640, height=118, bg=BG,
                                    highlightthickness=0)
            self.win_cv.pack(fill="x")
            self.cv = tk.Canvas(mid, width=640, height=520, bg=BG,
                                highlightthickness=0)
            self.cv.pack(fill="both", expand=True)
            self.cv.bind("<Button-1>", self.on_click)

        # ---------------- right: log + stats ----------------
        def _build_side(self):
            s = tk.Frame(self.root, bg=PANEL, padx=8, pady=8)
            s.grid(row=0, column=2, sticky="ns")

            tk.Label(s, text="EVENT LOG", bg=PANEL, fg=C_DATA,
                     font=("Helvetica", 9, "bold")).pack(anchor="w")
            wrap = tk.Frame(s, bg=PANEL)
            wrap.pack(fill="both", expand=True)
            sb = tk.Scrollbar(wrap)
            sb.pack(side="right", fill="y")
            self.log = tk.Text(wrap, width=46, height=22, bg=BG, fg=INK,
                               font=self.mono, relief="flat", wrap="word",
                               yscrollcommand=sb.set)
            self.log.pack(side="left", fill="both", expand=True)
            sb.config(command=self.log.yview)
            for tag, col in (("data", C_DATA), ("retx", C_RETX), ("ack", C_ACK),
                             ("ok", "#86efac"), ("err", C_LOST), ("to", C_TO),
                             ("hdr", "#c4b5fd")):
                self.log.tag_config(tag, foreground=col)

            tk.Label(s, text="STATISTICS", bg=PANEL, fg=C_DATA,
                     font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(6, 1))
            self.stat = tk.Label(s, text="", bg=BG, fg=INK, font=self.mono,
                                 justify="left", anchor="w", padx=6, pady=6)
            self.stat.pack(fill="x")

            tk.Label(s, text="LEGEND", bg=PANEL, fg=C_DATA,
                     font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(6, 1))
            leg = ("blue \u2192 data   orange \u2192 retransmit\n"
                   "green \u2190 ACK    red \u2715 lost   TO = timeout\n"
                   "sender strip: grey unsent / yellow sent,\n"
                   "  unacked / green acked / orange resend\n"
                   "recv strip: green delivered / blue buffered\n"
                   "  white outline = expected frame")
            tk.Label(s, text=leg, bg=BG, fg=MUTED, font=("Courier", 8),
                     justify="left", anchor="w", padx=6, pady=6).pack(fill="x")

        # ---------------- config plumbing ----------------
        def on_protocol_change(self):
            p = self.proto.get()
            self.sp_win.config(state="disabled" if p == STOP_WAIT else "normal")
            self.on_config_change()

        def on_config_change(self):
            self.pause()
            c = self.cfg
            try:
                # while the user is mid-edit the box may be empty or invalid;
                # ignore those keystrokes and keep the previous setting.
                win, tot = max(1, self.v_win.get()), self.v_tot.get()
                prop, tmo = self.v_prop.get(), self.v_to.get()
            except tk.TclError:
                return
            if tot < 1 or prop < 1 or tmo < 1:
                return
            c.protocol = self.proto.get()
            c.window = win
            c.total_frames = tot
            c.prop_delay = prop
            c.timeout = tmo
            c.p_data_loss = self.v_dl.get() / 100.0
            c.p_ack_loss = self.v_al.get() / 100.0
            c.p_corrupt = self.v_cr.get() / 100.0
            self.reset()

        def preset(self, d, a, c):
            self.v_dl.set(d); self.v_al.set(a); self.v_cr.set(c)
            self.on_config_change()

        def reset(self):
            self.pause()
            self.sim = Simulator(self.cfg)
            self.scroll = 0
            self.log.delete("1.0", "end")
            self.redraw()

        # ---------------- playback ----------------
        def toggle(self):
            self.running = not self.running
            self.btn_play.config(text="\u23f8  Pause" if self.running else "\u25b6  Play",
                                 bg="#92400e" if self.running else "#166534")
            if self.running:
                self.loop()

        def pause(self):
            self.running = False
            if hasattr(self, "btn_play"):
                self.btn_play.config(text="\u25b6  Play", bg="#166534")
            if self.job:
                self.root.after_cancel(self.job)
                self.job = None

        def loop(self):
            if not self.running:
                return
            self.one_step()
            if self.sim.finished:
                self.pause()
                return
            self.job = self.root.after(self.v_speed.get(), self.loop)

        def one_step(self):
            n = len(self.sim.log_lines)
            self.sim.step()
            for tick, msg, tag in self.sim.log_lines[n:]:
                self.log.insert("end", "t=%3d  %s\n" % (tick, msg), tag)
            self.log.see("end")
            self.redraw()

        def drop_data(self):
            self.sim.arm_drop_data = True
            self.sim.log("(armed: next DATA frame will be destroyed)", "to")
            self.log.insert("end", "       (armed: next DATA frame will be destroyed)\n", "to")
            self.log.see("end")

        def drop_ack(self):
            self.sim.arm_drop_ack = True
            self.sim.log("(armed: next ACK will be destroyed)", "to")
            self.log.insert("end", "       (armed: next ACK will be destroyed)\n", "to")
            self.log.see("end")

        # ---------------- click to kill a frame in flight ----------------
        def on_click(self, ev):
            sim = self.sim
            best, bestd = None, 1e9
            for f in sim.frames:
                if f.lost or not f.in_flight(sim.t):
                    continue
                x, y = self.point_of(f, sim.t)
                d = (x - ev.x) ** 2 + (y - ev.y) ** 2
                if d < bestd:
                    best, bestd = f, d
            if best and bestd < 400:          # within 20 px
                prog = (sim.t - best.depart) / max(1, best.arrive - best.depart)
                best.lost = True
                best.die_frac = max(0.05, min(1.0, prog))
                best.note = "killed by user"
                if best.kind == "data":
                    sim.stats["data_lost"] += 1
                else:
                    sim.stats["ack_lost"] += 1
                msg = "!! %s seq=%d destroyed in the channel by user" % (
                    "DATA" if best.kind == "data" else "ACK", sim.s(best.seq))
                sim.log(msg, "err")
                self.log.insert("end", "t=%3d  %s\n" % (sim.t, msg), "err")
                self.log.see("end")
                self.redraw()

        # ---------------- geometry ----------------
        def y(self, tick):
            return TOP + (tick - self.scroll) * PPT

        def point_of(self, f, t):
            x1, x2 = (SX, RX) if f.kind == "data" else (RX, SX)
            span = max(1, f.arrive - f.depart)
            p = min(1.0, max(0.0, (t - f.depart) / span))
            if f.lost and f.die_frac is not None:
                p = min(p, f.die_frac)
            return x1 + (x2 - x1) * p, self.y(f.depart) + (self.y(f.arrive) - self.y(f.depart)) * p

        # ---------------- drawing ----------------
        def redraw(self):
            self.draw_windows()
            self.draw_ladder()
            self.draw_stats()

        def draw_windows(self):
            c, sim = self.win_cv, self.sim
            c.delete("all")
            n = sim.cfg.total_frames
            bw = min(30, int(600 / max(1, n)))
            x0 = 20

            c.create_text(x0, 10, text="SENDER  window [%d .. %d)   N=%d   seq space M=%d"
                          % (sim.base, sim.base + sim.W, sim.W, sim.M),
                          anchor="w", fill=MUTED, font=("Helvetica", 8))
            for i in range(n):
                x, y = x0 + i * bw, 20
                if sim.cfg.protocol == SEL_REPEAT and i in sim.acked:
                    col = "#166534"
                elif i < sim.base:
                    col = "#166534"
                elif i in sim.retx_queue:
                    col = "#9a3412"
                elif i < sim.next_seq:
                    col = "#854d0e"
                else:
                    col = "#334155"
                c.create_rectangle(x, y, x + bw - 3, y + 22, fill=col, outline=GRID)
                c.create_text(x + (bw - 3) / 2, y + 11, text=str(sim.s(i)),
                              fill=INK, font=("Courier", 8, "bold"))
            wx1 = x0 + min(n - 1, sim.base) * bw - 3
            wx2 = x0 + min(n, max(sim.base + sim.W, sim.base + 1)) * bw - 3
            if sim.base < n:
                c.create_rectangle(wx1, 16, wx2, 46, outline=C_DATA, width=2)

            c.create_text(x0, 62, text="RECEIVER  expecting %d   receiver window = %d"
                          % (sim.s(sim.rcv_expected), sim.RW),
                          anchor="w", fill=MUTED, font=("Helvetica", 8))
            for i in range(n):
                x, y = x0 + i * bw, 72
                if i < sim.rcv_expected:
                    col = "#166534"
                elif i in sim.rcv_buffer:
                    col = "#1e40af"
                else:
                    col = "#334155"
                out = INK if i == sim.rcv_expected else GRID
                c.create_rectangle(x, y, x + bw - 3, y + 22, fill=col,
                                   outline=out, width=2 if out == INK else 1)
                c.create_text(x + (bw - 3) / 2, y + 11, text=str(sim.s(i)),
                              fill=INK, font=("Courier", 8, "bold"))
            if sim.cfg.protocol == SEL_REPEAT:
                rx1 = x0 + sim.rcv_expected * bw - 3
                rx2 = x0 + min(n, sim.rcv_expected + sim.RW) * bw - 3
                c.create_rectangle(rx1, 68, rx2, 98, outline="#60a5fa", width=2)

        def draw_ladder(self):
            c, sim = self.cv, self.sim
            c.delete("all")
            H = int(c.winfo_height() or 520)
            visible = (H - TOP) / PPT
            if sim.t - self.scroll > visible * 0.72:
                self.scroll = sim.t - visible * 0.72

            c.create_line(SX, TOP - 12, SX, H, fill=GRID, width=2)
            c.create_line(RX, TOP - 12, RX, H, fill=GRID, width=2)

            first = int(self.scroll) - int(self.scroll) % 5
            for tick in range(max(0, first), int(self.scroll + visible) + 5, 5):
                yy = self.y(tick)
                if TOP - 6 <= yy <= H:
                    c.create_line(SX - 6, yy, RX + 6, yy, fill="#1e293b")
                    c.create_text(40, yy, text="t=%d" % tick, fill=MUTED,
                                  font=("Courier", 7), anchor="w")

            for tk_, seq in sim.timeout_events:
                yy = self.y(tk_)
                if TOP - 6 <= yy <= H:
                    c.create_line(SX - 30, yy, SX, yy, fill=C_TO, width=2,
                                  dash=(3, 2))
                    c.create_text(SX - 34, yy, text="TO %d" % sim.s(seq),
                                  fill=C_TO, font=("Courier", 7, "bold"),
                                  anchor="e")

            for f in sim.frames:
                if f.depart > sim.t:
                    continue
                y1 = self.y(f.depart)
                if y1 > H + 40:
                    continue
                x1 = SX if f.kind == "data" else RX
                cx, cy = self.point_of(f, sim.t)
                if cy < TOP - 40:
                    continue
                col = C_LOST if f.lost else (C_RETX if f.retx else
                                             (C_DATA if f.kind == "data" else C_ACK))
                done = (sim.t >= f.arrive) and not f.lost
                c.create_line(x1, y1, cx, cy, fill=col, width=2,
                              dash=(4, 3) if f.lost else None,
                              arrow="last" if done else None,
                              arrowshape=(9, 11, 4))
                if f.lost and (sim.t - f.depart) / max(1, f.arrive - f.depart) >= f.die_frac:
                    c.create_text(cx, cy, text="\u2715", fill=C_LOST,
                                  font=("Helvetica", 13, "bold"))
                if f.kind == "data":
                    txt = "D%d%s" % (sim.s(f.seq), "*" if f.retx else "")
                    off = 8 + 26 * (f.depart % 2)   # stagger: no label collisions
                    c.create_text(x1 + off, y1 - 7, text=txt, fill=col,
                                  anchor="w", font=("Courier", 8, "bold"))
                    if f.corrupt and not f.lost:
                        c.create_text(cx - 8, cy - 8, text="CRC!", fill=C_TO,
                                      anchor="e", font=("Courier", 7, "bold"))
                else:
                    off = 8 + 26 * (f.depart % 2)
                    c.create_text(x1 - off, y1 - 7, text="A%d" % sim.s(f.seq),
                                  fill=col, anchor="e",
                                  font=("Courier", 8, "bold"))

            # header band drawn last, on an opaque strip, so no arrow bleeds into it
            c.create_rectangle(0, 0, 640, TOP - 13, fill=BG, outline="")
            c.create_text(SX, 12, text="SENDER", fill=INK,
                          font=("Helvetica", 9, "bold"))
            c.create_text(RX, 12, text="RECEIVER", fill=INK,
                          font=("Helvetica", 9, "bold"))
            c.create_text(SX + (RX - SX) / 2, 12,
                          text="time \u2193      %s" % sim.cfg.protocol,
                          fill=MUTED, font=("Helvetica", 8))

            if sim.finished:
                c.create_text(SX + (RX - SX) / 2, H - 16,
                              text="SIMULATION COMPLETE",
                              fill="#86efac", font=("Helvetica", 11, "bold"))

        def draw_stats(self):
            m = self.sim.metrics()
            txt = ("time elapsed      : %4d ticks\n"
                   "frames delivered  : %4d / %d\n"
                   "data frames sent  : %4d   (retx %d)\n"
                   "ACKs sent         : %4d\n"
                   "lost in channel   : %4d\n"
                   "corrupted (CRC)   : %4d\n"
                   "timeouts fired    : %4d\n"
                   "efficiency        : %5.1f %%\n"
                   "throughput        : %5.3f frames/tick"
                   % (m["time"], m["delivered"], self.cfg.total_frames,
                      m["data_tx"], m["retx"], m["ack_tx"], m["lost"],
                      m["corrupted"], m["timeouts"], m["efficiency"],
                      m["throughput"]))
            self.stat.config(text=txt)

    root = tk.Tk()
    root.geometry("1250x800")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    build_gui()
