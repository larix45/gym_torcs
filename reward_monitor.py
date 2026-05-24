#!/usr/bin/env python3
"""
Realtime reward monitor: tails a training report file and plots rewards per step.
Usage:
    python3 reward_monitor.py training_sequence_report.txt

Parses lines containing 'REWARD' and extracts numeric values.
"""
import sys
import time
import re
import os
import argparse
import traceback

try:
    import matplotlib
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        for backend in ('TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg'):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    import matplotlib.pyplot as plt
    if not matplotlib.is_interactive() and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('No graphical display available for reward monitor.')
except Exception as e:
    print('matplotlib is required for the monitor: ', e)
    sys.exit(1)

LINE_RE = re.compile(r'REWARD[^:]*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)')


def tail_and_plot(path, poll=0.1):
    rewards = []
    try:
        f = open(path, 'r')
    except Exception as e:
        print('Could not open', path, e)
        return

    # Seek to end
    f.seek(0, 2)

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-b')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Reward')
    ax.set_title('Realtime Reward per Step')
    fig.show()
    fig.canvas.draw()
    plt.pause(0.1)

    try:
        while True:
            where = f.tell()
            line_raw = f.readline()
            if not line_raw:
                time.sleep(poll)
                f.seek(where)
            else:
                m = LINE_RE.search(line_raw)
                if m:
                    try:
                        r = float(m.group(1))
                    except Exception:
                        continue
                    rewards.append(r)
                    if len(rewards) > 2000:
                        rewards = rewards[-2000:]
                    xs = list(range(len(rewards)))
                    line.set_data(xs, rewards)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    plt.pause(0.01)
    except KeyboardInterrupt:
        print('\nExiting monitor')
    finally:
        f.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', default='training_sequence_report.txt', help='Path to training report (default: training_sequence_report.txt)')
    p.add_argument('--poll', type=float, default=0.1, help='Polling interval in seconds')
    args = p.parse_args()
    tail_and_plot(args.file, poll=args.poll)
