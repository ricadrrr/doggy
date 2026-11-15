#!/usr/bin/env python3
"""Live Go2 speed tracking: joystick command vs MuJoCo body-frame velocity.

Samples history at a fixed rate (not per DDS packet) into a rolling window.

Requires env ``unitree-measure`` (unitree_sdk2py + matplotlib).

  conda activate unitree-measure
  python scripts/deploy/measure_mujoco_speed.py --network lo --domain 0 --vmax 2.5
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque


def _import_sdk():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            LowState_,
            SportModeState_,
            WirelessController_,
        )
    except ImportError as e:
        print(
            "Missing unitree_sdk2py. Activate unitree-measure and install:\n"
            "  export CYCLONEDDS_HOME=~/cyclonedds/install\n"
            "  cd ~/unitree_sdk2_python && pip install -e .",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, SportModeState_, WirelessController_


def scale_axis(axis: float, lim_min: float, lim_max: float) -> float:
    """Same mapping as deploy observations.h velocity_commands."""
    axis = max(-1.0, min(1.0, float(axis)))
    if axis >= 0.0:
        return axis * lim_max
    return axis * (-lim_min)


def world_vel_to_body_x(quat_wxyz, vx: float, vy: float, vz: float) -> float:
    w, x, y, z = quat_wxyz
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    return r00 * vx + r10 * vy + r20 * vz


class SpeedMeter:
    """DDS callbacks update latest_*; UI thread samples history at fixed rate."""

    def __init__(self, lim_min: float, lim_max: float, window_s: float, sample_hz: float):
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.target = lim_max
        self.window_s = window_s
        self.sample_dt = 1.0 / max(sample_hz, 1.0)
        n = max(32, int(window_s * sample_hz) + 2)
        self.t = deque(maxlen=n)
        self.cmd = deque(maxlen=n)
        self.act = deque(maxlen=n)
        self.err = deque(maxlen=n)

        self._lock = threading.Lock()
        self.pkt_n = 0
        self.cmd_ly = 0.0
        self.latest_v = (0.0, 0.0, 0.0)
        self.latest_horiz = 0.0
        self.latest_speed3 = 0.0
        self.latest_body_x = 0.0
        self.latest_cmd = 0.0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.has_imu = False
        self.has_vel = False

        self.max_horiz = 0.0
        self.max_vx = 0.0
        self.max_speed3 = 0.0
        self.max_body_x = 0.0
        self.max_body_x_abs = 0.0
        self.max_abs_err = 0.0
        self.t_above_90 = 0.0
        self.t_above_target = 0.0
        self._last_pkt_t = None
        self.t0 = time.time()
        self._last_sample_t = 0.0

    def on_lowstate(self, msg):
        q = msg.imu_state.quaternion
        with self._lock:
            self.quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            self.has_imu = True

    def on_wireless(self, msg):
        with self._lock:
            self.cmd_ly = float(msg.ly)

    def on_highstate(self, msg):
        vx = float(msg.velocity[0])
        vy = float(msg.velocity[1])
        vz = float(msg.velocity[2])
        horiz = math.hypot(vx, vy)
        speed3 = math.sqrt(vx * vx + vy * vy + vz * vz)
        now = time.time()

        with self._lock:
            cmd = scale_axis(self.cmd_ly, self.lim_min, self.lim_max)
            if self.has_imu:
                body_x = world_vel_to_body_x(self.quat, vx, vy, vz)
            else:
                body_x = math.copysign(horiz, cmd) if abs(cmd) > 1e-3 else horiz

            dt = 0.0 if self._last_pkt_t is None else max(0.0, now - self._last_pkt_t)
            self._last_pkt_t = now

            self.pkt_n += 1
            self.has_vel = True
            self.latest_v = (vx, vy, vz)
            self.latest_horiz = horiz
            self.latest_speed3 = speed3
            self.latest_body_x = body_x
            self.latest_cmd = cmd
            self.max_horiz = max(self.max_horiz, horiz)
            self.max_vx = max(self.max_vx, vx)
            self.max_speed3 = max(self.max_speed3, speed3)
            self.max_body_x = max(self.max_body_x, body_x)
            self.max_body_x_abs = max(self.max_body_x_abs, abs(body_x))
            self.max_abs_err = max(self.max_abs_err, abs(cmd - body_x))
            if dt > 0.0:
                if abs(body_x) >= 0.9 * self.target:
                    self.t_above_90 += dt
                if abs(body_x) >= self.target:
                    self.t_above_target += dt

    def catch_up_samples(self) -> int:
        with self._lock:
            if not self.has_vel:
                return 0
            cmd = self.latest_cmd
            act = self.latest_body_x

        now = time.time()
        if self._last_sample_t <= 0.0:
            self._last_sample_t = now - self.sample_dt

        n = 0
        max_catch = max(1, int(self.window_s / self.sample_dt))
        while self._last_sample_t + self.sample_dt <= now and n < max_catch:
            self._last_sample_t += self.sample_dt
            t_rel = self._last_sample_t - self.t0
            self.t.append(t_rel)
            self.cmd.append(cmd)
            self.act.append(act)
            self.err.append(cmd - act)
            n += 1
        return n

    def snapshot(self):
        with self._lock:
            vx, vy, vz = self.latest_v
            return {
                "pkt_n": self.pkt_n,
                "cmd": self.latest_cmd,
                "act": self.latest_body_x,
                "err": self.latest_cmd - self.latest_body_x,
                "horiz": self.latest_horiz,
                "speed3": self.latest_speed3,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "ly": self.cmd_ly,
                "has_imu": self.has_imu,
                "has_vel": self.has_vel,
                "max_body_x_abs": self.max_body_x_abs,
                "max_body_x": self.max_body_x,
                "max_horiz": self.max_horiz,
                "max_vx": self.max_vx,
                "max_speed3": self.max_speed3,
                "max_abs_err": self.max_abs_err,
                "t_above_90": self.t_above_90,
                "t_above_target": self.t_above_target,
            }

    def history(self):
        return list(self.t), list(self.cmd), list(self.act), list(self.err)

    def reached(self) -> bool:
        return self.max_body_x_abs >= self.target

    def pct_of_target(self) -> float:
        if self.target <= 0:
            return 0.0
        return 100.0 * self.max_body_x_abs / self.target


def _fmt_metrics(s: dict, meter: SpeedMeter, args) -> str:
    ok = "YES" if meter.reached() else "NO"
    imu = "OK" if s["has_imu"] else "WAIT"
    return (
        f"{'METRIC':<18}{'NOW':>9}{'PEAK':>9}\n"
        f"{'-'*36}\n"
        f"{'cmd vx':<18}{s['cmd']:>+9.3f}{'':>9}\n"
        f"{'body vx':<18}{s['act']:>+9.3f}{s['max_body_x']:>+9.3f}\n"
        f"{'tracking err':<18}{s['err']:>+9.3f}{s['max_abs_err']:>9.3f}\n"
        f"{'|v_xy| world':<18}{s['horiz']:>9.3f}{s['max_horiz']:>9.3f}\n"
        f"{'vx world':<18}{s['vx']:>+9.3f}{s['max_vx']:>+9.3f}\n"
        f"{'vy world':<18}{s['vy']:>+9.3f}{'':>9}\n"
        f"{'vz world':<18}{s['vz']:>+9.3f}{'':>9}\n"
        f"{'|v| 3D':<18}{s['speed3']:>9.3f}{s['max_speed3']:>9.3f}\n"
        f"{'stick ly':<18}{s['ly']:>+9.2f}{'':>9}\n"
        f"{'-'*36}\n"
        f"{'target':<18}{args.vmax:>9.2f} m/s\n"
        f"{'range':<18}[{args.vmin:g}, {args.vmax:g}]\n"
        f"{'|body vx|/tgt':<18}{meter.pct_of_target():>8.1f}%\n"
        f"{'reached target':<18}{ok:>9}\n"
        f"{'t >= 90% tgt':<18}{s['t_above_90']:>8.2f}s\n"
        f"{'t >= target':<18}{s['t_above_target']:>8.2f}s\n"
        f"{'packets':<18}{s['pkt_n']:>9}\n"
        f"{'IMU':<18}{imu:>9}\n"
        f"{'window':<18}{args.window:>8.1f}s\n"
    )


def run_live_plot(meter: SpeedMeter, args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    # Clean dark dashboard
    bg = "#111318"
    panel = "#1a1d24"
    grid_c = "#2a2f3a"
    text_c = "#e6e8ee"
    muted = "#9aa3b2"
    cmd_c = "#4ea1ff"
    act_c = "#ff5c5c"
    err_c = "#3ddc97"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": panel,
            "figure.facecolor": bg,
            "text.color": text_c,
            "axes.labelcolor": text_c,
            "axes.edgecolor": grid_c,
            "xtick.color": muted,
            "ytick.color": muted,
            "grid.color": grid_c,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "legend.facecolor": "#222632",
            "legend.edgecolor": grid_c,
            "legend.fontsize": 9,
        }
    )

    fig = plt.figure(figsize=(13.5, 7.2), dpi=110)
    fig.canvas.manager.set_window_title("Go2 Speed Tracking")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3.2, 1.15], height_ratios=[2.3, 1.0], hspace=0.08, wspace=0.08, left=0.06, right=0.98, top=0.93, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_m = fig.add_subplot(gs[:, 1])
    ax_m.set_facecolor(panel)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for spine in ax_m.spines.values():
        spine.set_color(grid_c)

    y_lo = min(args.vmin, -0.2) - 0.35
    y_hi = max(args.vmax, 0.2) + 0.35

    (line_cmd,) = ax.plot([], [], color=cmd_c, lw=2.0, label="command vx", solid_capstyle="round")
    (line_act,) = ax.plot([], [], color=act_c, lw=2.0, label="actual body-vx", solid_capstyle="round")
    ax.axhline(args.vmax, color=cmd_c, ls="--", lw=0.9, alpha=0.35)
    ax.axhline(args.vmin, color=cmd_c, ls="--", lw=0.9, alpha=0.35)
    ax.axhline(0.0, color=muted, lw=0.7, alpha=0.5)
    ax.set_ylabel("speed (m/s)")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0.0, args.window)
    ax.grid(True, alpha=0.85)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_title("Command vs Actual", fontsize=12, color=text_c, pad=8)
    plt.setp(ax.get_xticklabels(), visible=False)

    (line_err,) = ax_e.plot([], [], color=err_c, lw=1.8, label="error = cmd - actual")
    ax_e.axhline(0.0, color=muted, lw=0.8, alpha=0.6)
    ax_e.set_xlabel("time (s)")
    ax_e.set_ylabel("error (m/s)")
    ax_e.set_ylim(-(args.vmax + 0.4), args.vmax + 0.4)
    ax_e.set_xlim(0.0, args.window)
    ax_e.grid(True, alpha=0.85)
    ax_e.legend(loc="upper left", framealpha=0.95)

    metrics_txt = ax_m.text(
        0.05,
        0.98,
        "waiting for DDS...",
        transform=ax_m.transAxes,
        ha="left",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=9.2,
        color=text_c,
        linespacing=1.35,
    )
    ax_m.set_title("Metrics", fontsize=12, color=text_c, pad=8)

    t0 = time.time()
    warned = False

    def update(_frame):
        nonlocal warned
        meter.catch_up_samples()
        snap = meter.snapshot()
        ts, cs, ac, er = meter.history()

        if snap["pkt_n"] == 0 and (not warned) and (time.time() - t0 > 5.0):
            print("[WARN] No sportmodestate for 5s. Check mujoco / go2_ctrl / domain / lo.", flush=True)
            warned = True

        metrics_txt.set_text(_fmt_metrics(snap, meter, args))

        if len(ts) >= 2:
            t_end = ts[-1]
            if t_end < args.window:
                t_start, t_end_lim = 0.0, args.window
            else:
                t_start, t_end_lim = t_end - args.window, t_end

            line_cmd.set_data(ts, cs)
            line_act.set_data(ts, ac)
            line_err.set_data(ts, er)
            ax.set_xlim(t_start, t_end_lim)
            ax_e.set_xlim(t_start, t_end_lim)

        if args.duration > 0 and (time.time() - t0) >= args.duration:
            plt.close(fig)

        return line_cmd, line_act, line_err, metrics_txt

    interval_ms = max(20, int(1000.0 / max(args.print_hz, 1.0)))
    print(
        f"[INFO] DDS domain={args.domain} iface={args.network}\n"
        f"[INFO] blue=command  red=actual  green=error  | window={args.window:g}s @ {args.sample_hz:g} Hz\n"
        f"[INFO] Close the window or Ctrl+C to finish.",
        flush=True,
    )

    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig._ani = ani

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user", flush=True)
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def run_table(meter: SpeedMeter, args) -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    console = Console()
    period = 1.0 / max(args.print_hz, 1.0)
    t0 = time.time()

    def build():
        meter.catch_up_samples()
        s = meter.snapshot()
        table = Table(title="Go2 MuJoCo Speed", header_style="bold")
        table.add_column("metric", style="cyan")
        table.add_column("now", justify="right")
        table.add_column("peak", justify="right")
        reached = Text("YES", style="bold green") if meter.reached() else Text("NO", style="bold red")
        table.add_row("cmd vx", f"{s['cmd']:+.3f}", "—")
        table.add_row("body vx", f"{s['act']:+.3f}", f"{s['max_body_x']:+.3f}")
        table.add_row("tracking err", f"{s['err']:+.3f}", f"{s['max_abs_err']:.3f}")
        table.add_row("|v_xy| world", f"{s['horiz']:.3f}", f"{s['max_horiz']:.3f}")
        table.add_row("vx world", f"{s['vx']:+.3f}", f"{s['max_vx']:+.3f}")
        table.add_row("vy world", f"{s['vy']:+.3f}", "—")
        table.add_row("vz world", f"{s['vz']:+.3f}", "—")
        table.add_row("|v| 3D", f"{s['speed3']:.3f}", f"{s['max_speed3']:.3f}")
        table.add_row("stick ly", f"{s['ly']:+.2f}", "—")
        table.add_row(f"reached {args.vmax:g}?", reached, f"{meter.pct_of_target():.1f}%")
        table.add_row("t >= 90% tgt", f"{s['t_above_90']:.2f}s", "—")
        table.add_row("t >= target", f"{s['t_above_target']:.2f}s", "—")
        table.add_row("packets", str(s["pkt_n"]), "IMU OK" if s["has_imu"] else "IMU WAIT")
        return table

    with Live(build(), console=console, refresh_per_second=max(args.print_hz, 1.0)) as live:
        try:
            while True:
                time.sleep(period)
                live.update(build())
                if args.duration > 0 and (time.time() - t0) >= args.duration:
                    break
        except KeyboardInterrupt:
            console.print("\n[INFO] stopped by user")


def print_summary(meter: SpeedMeter, args) -> None:
    s = meter.snapshot()
    print("\n======== SUMMARY ========", flush=True)
    print(f"packets              : {s['pkt_n']}", flush=True)
    print(f"peak |body-vx|       : {s['max_body_x_abs']:.4f} m/s", flush=True)
    print(f"peak body-vx (+)     : {s['max_body_x']:.4f} m/s", flush=True)
    print(f"peak |v_xy| world    : {s['max_horiz']:.4f} m/s", flush=True)
    print(f"peak vx world        : {s['max_vx']:.4f} m/s", flush=True)
    print(f"peak |v| 3D          : {s['max_speed3']:.4f} m/s", flush=True)
    print(f"peak |tracking err|  : {s['max_abs_err']:.4f} m/s", flush=True)
    print(f"time >= 0.9*{args.vmax:g}  : {s['t_above_90']:.2f} s", flush=True)
    print(f"time >= {args.vmax:g}      : {s['t_above_target']:.2f} s", flush=True)
    if meter.reached():
        print(f"RESULT: reached {args.vmax:g} m/s (body-forward peak).", flush=True)
    else:
        print(
            f"RESULT: did NOT reach {args.vmax:g} m/s "
            f"(peak {s['max_body_x_abs']:.3f}; {meter.pct_of_target():.1f}% of target).",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Live cmd vs actual Go2 speed (MuJoCo DDS).")
    parser.add_argument("--network", default="lo")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--target", type=float, default=None, help="Alias of --vmax")
    parser.add_argument("--vmax", type=float, default=2.5, help="Forward command max")
    parser.add_argument("--vmin", type=float, default=-1.5, help="Backward command min")
    parser.add_argument("--window", type=float, default=20.0, help="Visible time window (s)")
    parser.add_argument("--sample-hz", type=float, default=50.0, help="History sample rate")
    parser.add_argument("--print-hz", type=float, default=20.0, help="UI refresh rate")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--table", action="store_true", help="Terminal table only")
    args = parser.parse_args()
    if args.target is not None:
        args.vmax = args.target

    ChannelFactoryInitialize, ChannelSubscriber, LowState_, SportModeState_, WirelessController_ = _import_sdk()
    ChannelFactoryInitialize(args.domain, args.network)

    meter = SpeedMeter(
        lim_min=args.vmin,
        lim_max=args.vmax,
        window_s=args.window,
        sample_hz=args.sample_hz,
    )

    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(meter.on_highstate, 20)
    ChannelSubscriber("rt/wirelesscontroller", WirelessController_).Init(meter.on_wireless, 20)
    ChannelSubscriber("rt/lowstate", LowState_).Init(meter.on_lowstate, 20)

    if args.table:
        run_table(meter, args)
    else:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("Missing matplotlib: pip install matplotlib", file=sys.stderr)
            raise SystemExit(1)
        run_live_plot(meter, args)

    print_summary(meter, args)


if __name__ == "__main__":
    main()
