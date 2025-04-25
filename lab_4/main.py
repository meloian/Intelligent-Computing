import os
from pathlib import Path

from .config import LUX_RANGE, HOUR_RANGE
from .utils import load_csv, save_results_csv
from .fuzzy.controller import Controller
from .viz_utils import plot_mf, plot_results

ctl = Controller()

def compute_record(rec):
    # compute fuzzy outputs and round brightness
    b, col = ctl.compute(rec["lux"], rec["hour"], rec["presence"], rec["eco"])
    result = {
        "lux": rec["lux"],
        "hour": rec["hour"],
        "presence": rec["presence"],
        "eco": rec["eco"],
        "brightness": round(b, 1),
        "color_temp": col
    }
    return result

def ask_float(prompt, vmin, vmax):
    while True:
        val = input(f"{prompt} [{vmin}-{vmax}]: ")
        try:
            num = float(val)
            if vmin <= num <= vmax:
                return num
            print(f"  value out of range {vmin}-{vmax}")
        except:
            print("  enter a number")

def ask_bool(prompt):
    ans = input(f"{prompt} (y/n): ").strip().lower()
    if ans in ("y", "yes", "1", "true"):
        return True
    return False

def fmt_bool(x):
    return "y" if x else "n"

def print_table(records):
    if not records:
        return
    headers = ["#", "lux", "hour", "presence", "eco", "brightness", "color_temp"]
    # compute column widths
    widths = {}
    for h in headers:
        widths[h] = len(h)
    for i, r in enumerate(records, 1):
        values = [
            str(i),
            str(r["lux"]),
            str(r["hour"]),
            fmt_bool(r["presence"]),
            fmt_bool(r["eco"]),
            str(r["brightness"]),
            str(r["color_temp"])
        ]
        for h, v in zip(headers, values):
            if len(v) > widths[h]:
                widths[h] = len(v)
    # header row
    line = ""
    for h in headers:
        line += h.ljust(widths[h] + 2)
    print("\n" + line.rstrip())
    # separator
    sep = ""
    for h in headers:
        sep += ("-" * widths[h]).ljust(widths[h] + 2)
    print(sep.rstrip())
    # data rows
    for i, r in enumerate(records, 1):
        vals = [
            str(i),
            str(r["lux"]),
            str(r["hour"]),
            fmt_bool(r["presence"]),
            fmt_bool(r["eco"]),
            str(r["brightness"]),
            str(r["color_temp"])
        ]
        row = ""
        for h, v in zip(headers, vals):
            row += v.ljust(widths[h] + 2)
        print(row.rstrip())

def main():
    print("=== lighting control system ===")
    print("1 - manual input, 2 - batch CSV\n")
    mode = input("choose mode (1/2): ").strip()

    if mode == "1":
        lux = ask_float("external lux", *LUX_RANGE)
        hour = ask_float("hour of day", *HOUR_RANGE)
        presence = ask_bool("presence in room")
        eco = ask_bool("eco mode")
        rec = {"lux": lux, "hour": hour, "presence": presence, "eco": eco}
        res = compute_record(rec)
        print("\n=== result ===")
        print_table([res])

    elif mode == "2":
        base = Path(__file__).parent
        csv_in = base / "data" / "sample_input.csv"
        if not csv_in.exists():
            print(f"file not found: {csv_in}")
            return
        rows = load_csv(csv_in)
        results = []
        for r in rows:
            results.append(compute_record(r))
        print(f"\n=== results for {csv_in.name} ===")
        print_table(results)
        if ask_bool("save results to CSV"):
            out = csv_in.with_name("sample_output.csv")
            save_results_csv(out, results)
        if ask_bool("show lux->brightness plot"):
            plot_results(results)

    else:
        print("unknown mode")
        return

    if ask_bool("show membership function plots"):
        for var in ("lux", "hour", "brightness"):
            plot_mf(var)

if __name__ == "__main__":
    main()