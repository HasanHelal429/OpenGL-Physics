"""Build a self-contained Plotly figure JSON from a headless self-consistent
run: the two charge trajectories with an animated marker, alongside the
energy-budget curves (KE, interaction PE, radiated, total). Written for the
al-folio project page (fetched and rendered client-side).

Usage:
    python inspiral_plotly.py <results_dir> --out figure.json
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    dg = np.genfromtxt(os.path.join(args.results_dir, "diagnostics.csv"),
                       delimiter=",", names=True)
    s = slice(None, None, args.stride)
    t = dg["t"][s]
    x0, y0 = dg["x0"][s], dg["y0"][s]
    x1, y1 = dg["x1"][s], dg["y1"][s]
    ke, pe, er = dg["KE"][s], dg["PE_interaction"][s], dg["E_radiated"][s]
    tot = ke + pe + er
    e0 = tot[0]

    lim = 1.05 * max(np.abs(np.r_[x0, y0, x1, y1]).max(), 1.0)
    ax_traj = dict(range=[-lim, lim], zeroline=False, showgrid=True,
                   gridcolor="rgba(128,128,128,0.15)", constrain="domain")

    base = [
        go.Scatter(x=x0, y=y0, mode="lines", line=dict(color="#d98f00", width=1),
                   name="+q path", xaxis="x", yaxis="y"),
        go.Scatter(x=x1, y=y1, mode="lines", line=dict(color="#3a7bd5", width=1),
                   name="-q path", xaxis="x", yaxis="y"),
        go.Scatter(x=[x0[0]], y=[y0[0]], mode="markers",
                   marker=dict(color="#d98f00", size=13), name="+q",
                   xaxis="x", yaxis="y"),
        go.Scatter(x=[x1[0]], y=[y1[0]], mode="markers",
                   marker=dict(color="#3a7bd5", size=13), name="-q",
                   xaxis="x", yaxis="y"),
        go.Scatter(x=t, y=ke - ke[0], name="Δ kinetic", xaxis="x2", yaxis="y2",
                   line=dict(color="#2ca02c")),
        go.Scatter(x=t, y=pe - pe[0], name="Δ interaction PE", xaxis="x2",
                   yaxis="y2", line=dict(color="#9467bd")),
        go.Scatter(x=t, y=er, name="radiated", xaxis="x2", yaxis="y2",
                   line=dict(color="#d62728")),
        go.Scatter(x=t, y=tot - e0, name="budget error", xaxis="x2", yaxis="y2",
                   line=dict(color="#111", width=2)),
        go.Scatter(x=[t[0], t[0]], y=[min(pe - pe[0]) * 1.1, max(er) * 1.1],
                   mode="lines", line=dict(color="rgba(0,0,0,0.35)", dash="dot"),
                   showlegend=False, xaxis="x2", yaxis="y2", name="now"),
    ]

    frames = []
    for k in range(len(t)):
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=[x0[k]], y=[y0[k]]),
            go.Scatter(x=[x1[k]], y=[y1[k]]),
            go.Scatter(x=[t[k], t[k]], y=[min(pe - pe[0]) * 1.1, max(er) * 1.1]),
        ], traces=[2, 3, 8]))

    layout = go.Layout(
        title="Two-body radiative inspiral — self-consistent retarded fields",
        showlegend=True, height=520, margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(domain=[0, 0.46], title="x", **ax_traj),
        yaxis=dict(domain=[0, 1], title="y", scaleanchor="x", **ax_traj),
        xaxis2=dict(domain=[0.58, 1], title="t", anchor="y2"),
        yaxis2=dict(domain=[0, 1], title="energy − E(0)", anchor="x2"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.08,
                          xanchor="left", direction="left", buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=40, redraw=False),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=False))]),
        ])],
        sliders=[dict(active=0, x=0.12, len=0.84, y=-0.06, pad=dict(t=0),
                      steps=[dict(method="animate", label="",
                                  args=[[str(k)], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))])
                             for k in range(0, len(t), max(1, len(t) // 60))])],
    )

    fig = go.Figure(data=base, frames=frames, layout=layout)
    with open(args.out, "w") as f:
        json.dump(json.loads(fig.to_json()), f)
    print("wrote", args.out, f"({len(t)} frames)")


if __name__ == "__main__":
    main()
