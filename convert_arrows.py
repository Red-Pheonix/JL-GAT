import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def build_arrows_for_series(times: pd.Series, phases: pd.Series, node_id: str, phase_offset: int = 1):
    # keep only rows where phase changes (and always include the first)
    changed = phases.ne(phases.shift(1))
    t = times[changed].astype(int).to_list()
    p = phases[changed].astype(int).to_list()

    arrows = []
    for (t0, p0), (t1, p1) in zip(zip(t, p), zip(t[1:], p[1:])):
        arrows.append({
            "node_id": node_id,
            "from_phase": int(p0) + phase_offset,
            "from_time": int(t0),
            "to_phase": int(p1) + phase_offset,
            "to_time": int(t1),
        })
    return arrows


def convert_all_intersections(
    source: Union[str, Path],
    *,
    time_col: str = "timestep",
    mode: Optional[str] = None,
    epoch: Optional[int] = None,
    meta_cols: Optional[List[str]] = None,
    phase_offset: int = 1,
) -> Dict:
    df = pd.read_csv(source)

    # filter if requested
    if mode is not None and "mode" in df.columns:
        df = df[df["mode"] == mode]
    if epoch is not None and "epoch" in df.columns:
        df = df[df["epoch"].astype(int) == int(epoch)]

    # sort by time (important after filtering)
    df = df.sort_values(time_col, kind="stable")

    # autodetect intersection columns
    if meta_cols is None:
        meta_cols = [c for c in ["mode", "epoch", time_col] if c in df.columns]
    intersection_cols = [c for c in df.columns if c not in meta_cols]

    times = df[time_col]
    all_arrows = []
    for col in intersection_cols:
        all_arrows.extend(build_arrows_for_series(times, df[col], node_id=str(col), phase_offset=phase_offset))

    return {"single_arrows": all_arrows}


def convert_file_all(
    in_path: str,
    out_path: str = "single_arrows_all.json",
    *,
    time_col: str = "timestep",
    mode: Optional[str] = None,
    epoch: Optional[int] = None,
    phase_offset: int = 1,
) -> None:
    data = convert_all_intersections(
        in_path,
        time_col=time_col,
        mode=mode,
        epoch=epoch,
        phase_offset=phase_offset,
    )
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(data['single_arrows'])} arrows)")
    

convert_file_all("ACTION.log", "single_arrows.json", mode="train", epoch=4)