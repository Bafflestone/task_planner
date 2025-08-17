# pandas + matplotlib Gantt from daily task allocations
# -----------------------------------------------------
# Requires: pandas, numpy, matplotlib
# Usage:
#   - Put your data in allocations.csv (same 4 columns as your example)
#   - Run this file. It will save:
#       gantt_daily.png
#       gantt_merged_task_colored.png
#
# Tunables:
WORK_START_HH_MM = (9, 0)     # 09:00
WORK_HOURS = 8                # 8-hour day
ROUND_FLOATS = 4              # tidy fp artifacts in day_fraction
MERGE_THROUGH_WEEKENDS = False  # False = only Mon–Fri are consecutive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import time

def create_resource_gantt(allocations_csv):

    # -----------------------------------------------------
    # 1) Load CSV
    # -----------------------------------------------------
    # Option A: from file
    df = pd.read_csv(allocations_csv, parse_dates=["date"])

    # (If you prefer an in-script CSV, replace the line above with:)
    # import io
    # csv_text = """<paste your CSV here>"""
    # df = pd.read_csv(io.StringIO(csv_text), parse_dates=["date"])

    df["day_fraction"] = df["day_fraction"].astype(float).round(ROUND_FLOATS)

    # Preserve the original input order inside each person/day
    df["row_order"] = np.arange(len(df))
    df = df.sort_values(["person_id", "date", "row_order"]).reset_index(drop=True)

    # -----------------------------------------------------
    # 2) Build concrete start/end times (09:00–17:00 by default)
    # -----------------------------------------------------
    def build_schedule(group: pd.DataFrame) -> pd.DataFrame:
        work_start = time(*WORK_START_HH_MM)
        cum = group["day_fraction"].shift(fill_value=0).cumsum()
        start_dt = pd.to_datetime(group["date"].dt.date.astype(str) + f" {work_start.strftime('%H:%M')}")
        start_dt = start_dt + pd.to_timedelta((cum * WORK_HOURS).values, unit="h")
        end_dt = start_dt + pd.to_timedelta((group["day_fraction"] * WORK_HOURS).values, unit="h")
        out = group.copy()
        out["start"] = start_dt
        out["end"] = end_dt
        return out

    schedule = (
        df.groupby(["person_id", "date"], group_keys=False)
        .apply(build_schedule)
        .reset_index(drop=True)
    )

    # Optional: tidy tables you can export/inspect
    tidy_schedule = (
        schedule[["person_id", "date", "task_id", "day_fraction", "start", "end"]]
        .sort_values(["person_id", "date", "start"])
        .reset_index(drop=True)
    )

    # Per-person per-day summary string (e.g., "T009 (09:00–12:48), T004 (12:48–17:00)")
    def _fmt_slot(row):
        return f"{row['task_id']} ({row['start'].strftime('%H:%M')}–{row['end'].strftime('%H:%M')})"

    daily_summary = (
        tidy_schedule.assign(slot=lambda d: d.apply(_fmt_slot, axis=1))
                    .groupby(["person_id", "date"])["slot"]
                    .apply(lambda s: ", ".join(s))
                    .reset_index(name="assignments")
                    .sort_values(["person_id", "date"])
    )

    # (Uncomment to save CSVs)
    # tidy_schedule.to_csv("tidy_schedule.csv", index=False)
    # daily_summary.to_csv("daily_summary.csv", index=False)

    # -----------------------------------------------------
    # 3) Helper: merge back-to-back working days for same person+task
    # -----------------------------------------------------
    def merge_back_to_back(schedule: pd.DataFrame, through_weekends: bool = False) -> pd.DataFrame:
        """
        Merge runs where the same person_id + task_id occur on consecutive days.
        - If through_weekends=False, "consecutive" means next *business* day (Mon–Fri) using numpy.busday_offset.
        - If through_weekends=True, consecutive means calendar next day.
        Returns rows: person_id, task_id, start, end (one row per run).
        """
        # Collapse to one row per (person, task, date) with earliest start and latest end on that date
        daily = (
            schedule.groupby(["person_id", "task_id", "date"], as_index=False)
                    .agg(start=("start", "min"), end=("end", "max"))
                    .sort_values(["person_id", "task_id", "date"])
                    .reset_index(drop=True)
        )

        dates_d = daily["date"].dt.date
        date_ns = daily["date"].dt.normalize().to_numpy(dtype="datetime64[ns]")
        d64 = date_ns.astype("datetime64[D]")
        prev = np.insert(d64[:-1], 0, np.datetime64("NaT", "D"))

        if through_weekends:
            # "consecutive" = prev + 1 calendar day
            def is_next(prev_d, cur_d):
                if np.isnat(prev_d) or np.isnat(cur_d):
                    return False
                return cur_d == (prev_d + np.timedelta64(1, "D"))
        else:
            # "consecutive business day" = busday_offset(prev, +1)
            def is_next(prev_d, cur_d):
                if np.isnat(prev_d) or np.isnat(cur_d):
                    return False
                return cur_d == np.busday_offset(prev_d, 1, roll="forward")

        consec = np.array([is_next(prev[i], d64[i]) for i in range(len(d64))])

        same_group = (daily[["person_id", "task_id"]].shift() == daily[["person_id", "task_id"]]).all(axis=1).to_numpy()
        same_and_consec = same_group & consec
        new_run = ~same_and_consec
        run_id = np.cumsum(new_run)

        merged = (
            daily.assign(run_id=run_id)
                .groupby(["person_id", "task_id", "run_id"], as_index=False)
                .agg(start=("start", "min"), end=("end", "max"))
                .drop(columns="run_id")
        )
        return merged

    merged = merge_back_to_back(schedule, through_weekends=MERGE_THROUGH_WEEKENDS)

    # -----------------------------------------------------
    # 4) Plotters
    # -----------------------------------------------------
    def plot_gantt(schedule_like: pd.DataFrame, title: str, outfile: str,
                color_by_task: bool = False):
        """
        schedule_like must have columns: person_id, task_id, start, end
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        persons = sorted(schedule_like["person_id"].unique())
        ymap = {p: i for i, p in enumerate(persons)}
        height = 0.8

        if color_by_task:
            tasks = sorted(schedule_like["task_id"].unique())
            cmap = plt.cm.get_cmap("tab20", len(tasks))
            task_color = {t: cmap(i) for i, t in enumerate(tasks)}
        else:
            task_color = None

        for _, row in schedule_like.iterrows():
            width = mdates.date2num(row["end"]) - mdates.date2num(row["start"])
            left = mdates.date2num(row["start"])
            kw = dict(edgecolor="black", align="center", height=height)
            if color_by_task:
                kw["color"] = task_color[row["task_id"]]
            ax.barh(ymap[row["person_id"]], width, left=left, **kw)

            # Annotate with task id
            cx = left + width / 2.0
            if color_by_task:
                col = task_color[row["task_id"]]
                txt_color = "white" if sum(col[:3]) < 1.5 else "black"
            else:
                txt_color = "black"
            ax.text(cx, ymap[row["person_id"]], row["task_id"],
                    va="center", ha="center", fontsize=8, color=txt_color)

        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(list(ymap.keys()))
        ax.set_xlabel("Date & time")
        ax.set_title(title)

        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        fig.autofmt_xdate()
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        if color_by_task:
            tasks = sorted(schedule_like["task_id"].unique())
            handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.get_cmap("tab20", len(tasks))(i)) for i in range(len(tasks))]
            ax.legend(handles, tasks, title="Tasks", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------
    # 5) Make the charts
    # -----------------------------------------------------

    # Merged runs across back-to-back working days, task-colored
    plot_gantt(
        schedule_like=merged,
        title="Gantt (merged back-to-back working days; task-colored)",
        outfile="data/gantt_merged_task_colored.png",
        color_by_task=True
    )

    print("Saved: gantt_daily.png, gantt_merged_task_colored.png")
    # (Optional) print/touch tables:
    # print(tidy_schedule.head(12).to_string(index=False))
    # print(daily_summary.head(12).to_string(index=False))
