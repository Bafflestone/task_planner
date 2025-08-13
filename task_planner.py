# Create a Gantt-style CSV and a plot from your scheduler.
# This cell:
# 1) Re-creates your example data and runs the scheduler
# 2) Builds a simple task-level Gantt CSV (one row per task: start, end)
# 3) Saves a PNG Gantt chart
# 4) Shows a quick preview of the Gantt CSV

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# Recreate your scheduler (same as in your script)
# -----------------------------

def parse_date(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    return pd.to_datetime(s).date()

def daterange(start: date, end: date):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def workdays_between(start: date, end: date):
    return [d for d in daterange(start, end) if d.weekday() < 5]  # Mon-Fri

def expand_skills(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return set()
    return set([x.strip().lower() for x in str(s).split("|") if x.strip()])

def load_inputs(tasks_csv, people_csv):
    tasks = pd.read_csv(tasks_csv, dtype=str)
    people = pd.read_csv(people_csv, dtype=str)
    # coerce numeric
    tasks["estimated_days"] = tasks["estimated_days"].astype(float)
    tasks["priority"] = tasks["priority"].astype(float) if "priority" in tasks.columns else 3.0
    # parse dates
    tasks["deadline"] = tasks["deadline"].apply(parse_date)
    tasks["earliest_start"] = tasks["earliest_start"].apply(parse_date) if "earliest_start" in tasks.columns else None
    if "depends_on" not in tasks.columns:
        tasks["depends_on"] = ""
    # people
    people["fte"] = people["fte"].astype(float) if "fte" in people.columns else 1.0
    if "unavailable_dates" not in people.columns: people["unavailable_dates"] = ""
    people["unavailable_dates"] = people["unavailable_dates"].fillna("")
    if "end_date" not in people.columns: people["end_date"] = ""
    people["end_date"] = people["end_date"].apply(parse_date)
    return tasks, people

def build_calendar(people: pd.DataFrame, start_date: date, end_date: date):
    calendars = {}
    for _, r in people.iterrows():
        pid = r["person_id"]
        fte = float(r.get("fte", 1.0))
        end_override = r.get("end_date")
        end_override = parse_date(end_override) if isinstance(end_override, str) else end_override
        skills = expand_skills(r.get("skills",""))
        unavail = set()
        if r.get("unavailable_dates"):
            for s in str(r["unavailable_dates"]).split("|"):
                s = s.strip()
                if s:
                    unavail.add(parse_date(s))
        days = workdays_between(start_date, end_override if end_override and end_override < end_date else end_date)
        cal = {}
        for d in days:
            cal[d] = fte if d not in unavail else 0.0
        calendars[pid] = {"skills": skills, "fte": fte, "calendar": cal, "name": r.get("person_name", pid)}
    return calendars

def topo_sort(tasks_df):
    deps = {r.task_id: set([x for x in str(r.depends_on).split("|") if x and x in set(tasks_df.task_id)]) for _, r in tasks_df.iterrows()}
    incoming = {t: set(deps[t]) for t in deps}
    no_incoming = [t for t in incoming if not incoming[t]]
    order = []
    while no_incoming:
        t = no_incoming.pop()
        order.append(t)
        for tt in incoming:
            if t in incoming[tt]:
                incoming[tt].remove(t)
                if not incoming[tt]:
                    no_incoming.append(tt)
    remaining = [t for t in tasks_df.task_id if t not in order]
    return order + remaining

def schedule(tasks_csv, people_csv):
    tasks, people = load_inputs(tasks_csv, people_csv)
    today = date.today()
    min_start = min([parse_date(es) or today for es in tasks["earliest_start"]]) if "earliest_start" in tasks.columns else today
    max_deadline = max(tasks["deadline"])
    calendars = build_calendar(people, min_start, max_deadline)
    task_skills = {r.task_id: expand_skills(r.required_skills) for _, r in tasks.iterrows()}
    topo = topo_sort(tasks)
    priority_map = {r.task_id: r.priority for _, r in tasks.iterrows()}
    deadline_map = {r.task_id: r.deadline for _, r in tasks.iterrows()}
    earliest_start_map = {r.task_id: (r.earliest_start if isinstance(r.earliest_start, date) else parse_date(r.earliest_start)) for _, r in tasks.iterrows()}
    est_days = {r.task_id: float(r.estimated_days) for _, r in tasks.iterrows()}
    order = sorted(topo, key=lambda t: (priority_map.get(t, 3), deadline_map.get(t, date.max)))
    per_day_assignments = []  # rows: date, task_id, person_id, slice (0-1 day)
    task_remaining = {t: est_days[t] for t in order}
    person_skills = {pid: set(data["skills"]) for pid, data in calendars.items()}
    all_days = workdays_between(min_start, max_deadline)
    completed = set()
    completed.add("T0")
    # deps_dict = {r.task_id: set([x for x in str(r.depends_on).split("|") if x]) for _, r in tasks.iterrows()
    #              }
    deps_dict = {}
    for _, r in tasks.iterrows():
        deps_dict[r.task_id] = set()
        print(r.depends_on)
        if r.depends_on:
            deps_dict[r.task_id].add("T0")
            continue
        for x in str(r.depends_on).split("|"):
            deps_dict[r.task_id].add(x)
    print(order)
    print(task_remaining)
    print(earliest_start_map)
    print(min_start)
    print(earliest_start_map.get("T1"))
    print(deadline_map)
    print(deps_dict)
    for d in all_days:
        print(d)
        capacity = {pid: calendars[pid]["calendar"].get(d, 0.0) for pid in calendars}
        eligible = [t for t in order
                    if task_remaining[t] > 0
                    and (earliest_start_map.get(t) or min_start) <= d
                    and d <= deadline_map.get(t, date.max)
                    and deps_dict.get(t, set()).issubset(completed)]
        print(f"eligible {eligible}")
        eligible = sorted(eligible, key=lambda t: (priority_map.get(t, 3), deadline_map.get(t)))
        for t in eligible:
            need = task_remaining[t]
            if need <= 0: 
                continue
            req_skills = task_skills.get(t, set())
            candidates = [pid for pid in capacity if capacity[pid] > 0 and (person_skills[pid] & req_skills)]
            candidates = sorted(candidates, key=lambda pid: calendars[pid]["fte"], reverse=True)
            for pid in candidates:
                if need <= 0:
                    break
                take = min(capacity[pid], need)
                if take > 0:
                    per_day_assignments.append([d.isoformat(), t, pid, float(take)])
                    capacity[pid] -= take
                    need -= take
                    task_remaining[t] -= take
            if task_remaining[t] <= 0:
                completed.add(t)
    print(per_day_assignments)
    per_day_df = pd.DataFrame(per_day_assignments, columns=["date","task_id","person_id","day_fraction"])
    task_status = []
    for t in order:
        allocated = per_day_df.loc[per_day_df.task_id==t, "day_fraction"].sum() if not per_day_df.empty else 0.0
        remaining = max(0.0, est_days[t] - allocated)
        task_status.append([t, allocated, est_days[t], remaining, deadline_map[t].isoformat(), priority_map[t]])
    task_status_df = pd.DataFrame(task_status, columns=["task_id","allocated_days","estimated_days","days_short_by_deadline","deadline","priority"])
    gaps = []
    for _, r in tasks.iterrows():
        t = r.task_id
        short = task_status_df.loc[task_status_df.task_id==t, "days_short_by_deadline"].values
        short = float(short[0]) if len(short) else 0.0
        if short > 0:
            skills = expand_skills(r.required_skills)
            if not skills:
                skills = {"(unspecified)"}
            per_skill = short / len(skills)
            dline = r.deadline
            week = (dline - timedelta(days=dline.weekday()))  # Monday of deadline week
            for s in skills:
                gaps.append([s, week.isoformat(), per_skill, t, dline.isoformat()])
    gaps_df = pd.DataFrame(gaps, columns=["skill","deadline_week","days_short","task_id","task_deadline"])
    if not per_day_df.empty:
        alloc_matrix = per_day_df.groupby(["task_id","person_id"])["day_fraction"].sum().unstack(fill_value=0).reset_index()
    else:
        alloc_matrix = pd.DataFrame()
    out_alloc_per_day = "data/allocation_per_day.csv"
    out_alloc_matrix = "data/allocation_matrix.csv"
    out_task_status = "data/task_status.csv"
    out_skill_gaps = "data/skill_gaps.csv"
    os.makedirs("data", exist_ok=True)
    per_day_df.to_csv(out_alloc_per_day, index=False)
    alloc_matrix.to_csv(out_alloc_matrix, index=False)
    task_status_df.to_csv(out_task_status, index=False)
    gaps_df.to_csv(out_skill_gaps, index=False)
    return {
        "out_alloc_per_day": out_alloc_per_day,
        "out_alloc_matrix": out_alloc_matrix,
        "out_task_status": out_task_status,
        "out_skill_gaps": out_skill_gaps,
        "per_day_df": per_day_df,
        "alloc_matrix": alloc_matrix,
        "task_status_df": task_status_df,
        "gaps_df": gaps_df
    }

def plan_tasks(tasks_path, people_path):

    # -----------------------------
    # Run scheduler
    # -----------------------------
    results = schedule(tasks_path, people_path)
    per_day_df = results["per_day_df"]
    task_status_df = results["task_status_df"]

    # Bring in task names for nicer labels
    tasks_df = pd.read_csv(tasks_path, dtype=str)

    # -----------------------------
    # Build a simple task-level Gantt CSV
    #   One row per task with start (first allocated day) and end (last allocated day)
    # -----------------------------
    if per_day_df.empty:
        gantt_tasks = pd.DataFrame(columns=["task_id","task_name","start_date","end_date","allocated_days"])
    else:
        tmp = per_day_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        agg = tmp.groupby("task_id").agg(
            start_date=("date","min"),
            end_date=("date","max"),
            allocated_days=("day_fraction","sum")
        ).reset_index()
        gantt_tasks = agg.merge(tasks_df[["task_id","task_name"]], on="task_id", how="left")

    # Save the Gantt CSV
    gantt_csv_path = "data/gantt_tasks.csv"
    gantt_tasks.to_csv(gantt_csv_path, index=False)

    # -----------------------------
    # Plot a basic Gantt chart (matplotlib only, one figure, no explicit colors)
    # -----------------------------
    if not gantt_tasks.empty:
        # Sort by end_date (or deadline) for nicer stacking
        gantt_tasks_sorted = gantt_tasks.sort_values(by=["end_date","start_date"]).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Convert dates for plotting
        starts = pd.to_datetime(gantt_tasks_sorted["start_date"])
        ends = pd.to_datetime(gantt_tasks_sorted["end_date"])
        durations = (ends - starts).dt.days + 1  # inclusive of both ends

        # y positions
        y_pos = range(len(gantt_tasks_sorted))

        # Draw bars
        ax.barh(y=y_pos, width=durations, left=mdates.date2num(starts))

        # Y-axis labels as task names
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(gantt_tasks_sorted["task_name"].fillna(gantt_tasks_sorted["task_id"]))

        # X-axis format as dates
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Date")
        ax.set_ylabel("Tasks")
        ax.set_title("Project Gantt (Task-Level)")

        plt.tight_layout()
        gantt_png_path = "data/gantt.png"
        plt.savefig(gantt_png_path, dpi=160)
    else:
        gantt_png_path = None


if __name__ == "__main__":

    # -----------------------------
    # Prepare example templates (same content as your script)
    # -----------------------------
    os.makedirs("data", exist_ok=True)

    tasks_template = pd.DataFrame([
        ["T1","Design landing page","design|ux",5,"2025-09-05","2025-08-18",2,""],
        ["T2","Implement landing page","frontend",8,"2025-09-12","2025-08-25",2,"T1"],
        ["T3","Set up backend API","graph",10,"2025-09-19","2025-08-18",1,""],
        ["T4","Write marketing copy","copywriting",4,"2025-09-03","2025-08-18",3,""],
        ["T5","Analytics instrumentation","data|frontend",6,"2025-09-17","2025-08-25",3,"T2|T3"],
        ["T6","QA & bug bash","qa",7,"2025-09-24","2025-09-08",2,"T2|T3"]
    ], columns=["task_id","task_name","required_skills","estimated_days","deadline","earliest_start","priority","depends_on"])
    people_template = pd.DataFrame([
        ["P1","Ava","design|ux|frontend",1.0,"2025-08-25",""],
        ["P2","Ben","backend|devops|data",1.0,"",""],
        ["P3","Chloe","frontend|qa",0.8,"",""],
        ["P4","Diego","copywriting|design",0.6,"2025-09-02|2025-09-03",""]
    ], columns=["person_id","person_name","skills","fte","unavailable_dates","end_date"])

    tasks_path = "data/tasks_template.csv"
    people_path = "data/people_template.csv"

    tasks_template.to_csv(tasks_path, index=False)
    people_template.to_csv(people_path, index=False)

    plan_tasks(tasks_path, people_path)

