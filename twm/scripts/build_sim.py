"""Build the interactive sensor simulator: a static page, no server.

Freezes ONE frame of the release -- three camera views and both tactile
streams -- and lets you drive either sensor with the keyboard while the
projected sensor frame follows. The photos do not change: only the overlay
moves. That is the honest thing to ship today, because nothing here predicts
what the cameras or the gels would actually see; a model goes in later, and
the page says so on its face rather than letting a frozen tactile image sit
beside a moved overlay implying it followed.

The action is the same one the probe test set uses, so what you drive here is
what a model would be asked to follow:

  * translation is a delta in WORLD millimetres;
  * rotation is a world-frame delta applied ABOUT THE GEL CENTRE, not about
    the marker cluster the pose 7-vec actually describes. The gel sits 65.7 mm
    away, so pivoting on the cluster swings the gel 30-50 mm across the table
    while looking like a clean spin -- that was a real bug in the probe
    generator, and scripts/test_sim.py asserts against it.

    python scripts/build_sim.py [--seed N] [--out DIR]
    python scripts/test_sim.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

from react_paths import force_meta, out_root, release_root     # noqa: E402

import react_toolbox as T                                      # noqa: E402
from react_toolbox.frames import require_up_axis               # noqa: E402

VIEWS = ("left", "middle", "right")
STREAMS = tuple(f"view_{v}" for v in VIEWS) + ("tactile_left", "tactile_right")
MARGIN_PX = 70.0          # room to drive before the sensor leaves the frame
STEP_MM = 5.0
STEP_DEG = 3.0
TASK = "motherboard"


def _episodes():
    """Episodes with published videos for every stream this page needs."""
    out = []
    for d in sorted((release_root(TASK) / "videos").iterdir()):
        if not d.is_dir():
            continue
        for e in sorted(d.iterdir()):
            if all((e / f"{s}.mp4").is_file() for s in STREAMS):
                out.append((d.name, e.name))
    return out


def _pick(rng, cal, splits):
    """A start frame that is held out, tracked, and comfortably in view.

    In view for BOTH sensors in ALL THREE views, with a margin -- otherwise
    the first keypress drives the thing you are looking at off the edge.
    """
    eps = _episodes()
    rng.shuffle(eps)
    for date, ep in eps:
        key = f"{date}/{ep}"
        iv = (splits["episodes"].get(key) or {}).get("test") or []
        if not iv:
            continue
        f = force_meta(TASK) / date / f"{ep}.parquet"
        if not f.exists():
            continue
        t = pq.read_table(str(f), columns=["sensor_left_pose",
                                           "sensor_right_pose"]).to_pydict()
        P = {s: np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
             for s in ("left", "right")}
        rows = [r for lo, hi in iv for r in range(lo, hi + 1)]
        rng.shuffle(rows)
        for r in rows[:400]:
            if r >= len(P["left"]):
                continue
            ok = True
            for s in ("left", "right"):
                p = P[s][r]
                if not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 0.5:
                    ok = False
                    break
                for v in VIEWS:
                    uv = T.project_gel_to_pixel(p, cal[f"gel_{s}"],
                                                cal["cams"][v])
                    if uv is None or not (MARGIN_PX <= uv[0] <= 640 - MARGIN_PX
                                          and MARGIN_PX <= uv[1] <= 480 - MARGIN_PX):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return date, ep, int(r), {s: P[s][r] for s in ("left", "right")}
    raise SystemExit("no held-out frame has both sensors comfortably in view "
                     "in all three cameras")


def _grab(date, ep, row, out):
    """The five frozen streams at that row, as JPEGs."""
    got = {}
    for s in STREAMS:
        f = release_root(TASK) / "videos" / date / ep / f"{s}.mp4"
        cap = cv2.VideoCapture(str(f))
        cap.set(cv2.CAP_PROP_POS_FRAMES, row)
        ok, im = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"cannot read {f} at row {row}")
        cv2.imwrite(str(out / f"{s}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        got[s] = (int(im.shape[1]), int(im.shape[0]))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(out_root("sim")))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    src_cal = release_root(TASK) / "calibration"
    shutil.copytree(src_cal, out / "calibration")
    cal = T.load_calibration(out)
    require_up_axis(cal, where=str(src_cal))

    splits = json.loads((release_root(TASK) / "splits.json").read_text())
    rng = np.random.default_rng(a.seed)
    date, ep, row, poses = _pick(rng, cal, splits)
    sizes = _grab(date, ep, row, out)

    cams = {}
    for v in VIEWS:
        c = cal["cams"][v]
        M = np.asarray(c["T_mocap_to_cam"], float)
        cams[v] = {"T": [float(x) for x in M[:3, :4].reshape(-1)],
                   "fx": c["intrinsics"]["fx"], "fy": c["intrinsics"]["fy"],
                   "ppx": c["intrinsics"]["ppx"], "ppy": c["intrinsics"]["ppy"]}

    data = {
        "date": date, "episode": ep, "key": f"{date}/{ep}", "row": int(row),
        "streams": list(STREAMS),
        "size": {k: list(v) for k, v in sizes.items()},
        "cams": cams,
        "gel": {s: [float(x) for x in cal[f"gel_{s}"]] for s in ("left", "right")},
        "pose": {s: [float(x) for x in poses[s]] for s in ("left", "right")},
        "step_mm": STEP_MM, "step_deg": STEP_DEG,
        "up_axis": cal["up_axis"],
        "note": ("the photos are frozen; only the projected sensor frame "
                 "moves. No model is wired in yet."),
    }
    blob = json.dumps(data, indent=1)
    (out / "sim.json").write_text(blob)
    assert "__DATA__" in _HTML, "the HTML lost its data placeholder"
    (out / "index.html").write_text(_HTML.replace("__DATA__", blob))
    print(f"{date}/{ep} row {row} -> {out}/index.html")
    print(f"  5 streams, both sensors in view with >= {MARGIN_PX:.0f} px margin")
    return 0


# An EXPLICIT html/head/body. Without it the browser still builds both, so
# everything looks right locally -- but Hugging Face injects a <script> into
# every static Space, and with no declared head the parser dropped it into the
# body, where the deployed page printed `window.huggingface={variables:...}`
# as a line of text above the title. Only the live URL showed it.
_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>React — interactive sensor simulator</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0e1116;color:#e8eaed;
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 18px 6px}
h1{font-size:17px;margin:0 0 4px;font-weight:600}
.sub{color:#9aa3ad;font-size:12.5px;max-width:76ch}
main{padding:10px 18px 28px}
.row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px}
.tile{position:relative;flex:0 0 auto;border-radius:6px;overflow:hidden;
      background:#000;line-height:0}
.tile img{display:block;width:100%;height:auto}
/* width AND height, both, in CSS. The HTML width/height attributes are
   presentational hints: setting only the CSS width let the img and its
   canvas resolve to different heights and every marker sat ~90 px off
   while every number in the page stayed correct. */
.tile canvas{position:absolute;inset:0;width:100%;height:100%}
.cap{position:absolute;left:0;top:0;padding:2px 6px;font-size:11px;
     background:rgba(0,0,0,.55);color:#cfd6de;line-height:1.4;z-index:2}
.frozen{position:absolute;right:0;bottom:0;padding:2px 6px;font-size:11px;
        background:rgba(120,60,0,.72);color:#ffd9a8;line-height:1.4;z-index:2}
.panel{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;
       background:#151a21;border:1px solid #232a33;border-radius:8px;
       padding:12px 16px;margin-top:4px}
.panel h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;
          color:#8d97a2;margin:0 0 6px;font-weight:600}
kbd{background:#222933;border:1px solid #333c47;border-bottom-width:2px;
    border-radius:4px;padding:0 5px;font:12px ui-monospace,monospace;color:#dfe5ec}
table{border-collapse:collapse;font-size:12.5px}
td{padding:1px 10px 1px 0;vertical-align:top}
.num{font:12.5px ui-monospace,monospace;color:#bfe6ff}
.sel{font-weight:600}
.warn{color:#ffb86b}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
      background:#1d2630;border:1px solid #2c3844;margin-right:6px}
</style>
</head>
<body>
<header>
  <h1>Interactive sensor simulator</h1>
  <p class="sub" data-tactile-note>Drive either GelSight with the keyboard and
  watch its projected frame move. <b>The photographs are frozen</b> — the three
  camera views and both tactile images are a single recorded instant and do not
  respond to the action. Only the overlay moves. No model is wired in yet, so
  nothing here is a prediction of what the sensor would see.</p>
</header>
<main>
  <div class="row" id="views"></div>
  <div class="row" id="tactile"></div>
  <div class="panel">
    <div>
      <h2>Driving</h2>
      <table>
        <tr><td><kbd>Tab</kbd></td><td>switch sensor</td></tr>
        <tr><td><kbd>←</kbd><kbd>→</kbd></td><td>world x</td></tr>
        <tr><td><kbd>↑</kbd><kbd>↓</kbd></td><td>world y</td></tr>
        <tr><td><kbd>[</kbd><kbd>]</kbd></td><td>world z (up)</td></tr>
        <tr><td><kbd>A</kbd><kbd>D</kbd></td><td>rotate about x</td></tr>
        <tr><td><kbd>W</kbd><kbd>S</kbd></td><td>rotate about y</td></tr>
        <tr><td><kbd>Q</kbd><kbd>E</kbd></td><td>rotate about z</td></tr>
        <tr><td><kbd>R</kbd></td><td>reset</td></tr>
      </table>
    </div>
    <div>
      <h2>Action so far</h2>
      <table>
        <tr><td>driving</td><td class="num sel" id="side">—</td></tr>
        <tr><td>translation</td><td class="num" id="tr">0, 0, 0 mm</td></tr>
        <tr><td>rotation</td><td class="num" id="rt">0.0&deg;</td></tr>
        <tr><td>gel centre</td><td class="num" id="gc">—</td></tr>
      </table>
    </div>
    <div>
      <h2>Frame</h2>
      <table>
        <tr><td>episode</td><td class="num" id="ep">—</td></tr>
        <tr><td>row</td><td class="num" id="rw">—</td></tr>
        <tr><td>world</td><td class="num" id="ua">—</td></tr>
      </table>
      <p class="sub" style="margin:8px 0 0;max-width:34ch">
      Rotation pivots on the <b>gel centre</b>, not on the marker cluster the
      pose vector describes — the cluster is 65.7&nbsp;mm away and moves.</p>
    </div>
  </div>
</main>
<script>
const AXC = ["#ff5a5a", "#5aff8f", "#5aa8ff"];      // x, y, z
const COL = {left: "#ffd23f", right: "#4fc3f7"};
let D = null, S = null;

/* ---- small matrix helpers, scalar-LAST quaternions (scipy from_quat) ---- */
function quatMat(q){
  const [x,y,z,w]=q, n=Math.hypot(x,y,z,w)||1;
  const X=x/n,Y=y/n,Z=z/n,W=w/n;
  return [[1-2*(Y*Y+Z*Z), 2*(X*Y-Z*W),   2*(X*Z+Y*W)],
          [2*(X*Y+Z*W),   1-2*(X*X+Z*Z), 2*(Y*Z-X*W)],
          [2*(X*Z-Y*W),   2*(Y*Z+X*W),   1-2*(X*X+Y*Y)]];
}
function matQuat(M){                       // -> x, y, z, w
  const t=M[0][0]+M[1][1]+M[2][2];
  let x,y,z,w;
  if(t>0){const s=Math.sqrt(t+1)*2; w=.25*s;
    x=(M[2][1]-M[1][2])/s; y=(M[0][2]-M[2][0])/s; z=(M[1][0]-M[0][1])/s;}
  else if(M[0][0]>M[1][1]&&M[0][0]>M[2][2]){
    const s=Math.sqrt(1+M[0][0]-M[1][1]-M[2][2])*2;
    w=(M[2][1]-M[1][2])/s; x=.25*s; y=(M[0][1]+M[1][0])/s; z=(M[0][2]+M[2][0])/s;}
  else if(M[1][1]>M[2][2]){
    const s=Math.sqrt(1+M[1][1]-M[0][0]-M[2][2])*2;
    w=(M[0][2]-M[2][0])/s; x=(M[0][1]+M[1][0])/s; y=.25*s; z=(M[1][2]+M[2][1])/s;}
  else{const s=Math.sqrt(1+M[2][2]-M[0][0]-M[1][1])*2;
    w=(M[1][0]-M[0][1])/s; x=(M[0][2]+M[2][0])/s; y=(M[1][2]+M[2][1])/s; z=.25*s;}
  return [x,y,z,w];
}
function mv(M,v){return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
                         M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
                         M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];}
function mm(A,B){return A.map((r,i)=>[0,1,2].map(j=>
  A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j]));}
function axisRot(k, deg){                  // rotation about world axis k
  const t=deg*Math.PI/180, c=Math.cos(t), s=Math.sin(t);
  if(k===0) return [[1,0,0],[0,c,-s],[0,s,c]];
  if(k===1) return [[c,0,s],[0,1,0],[-s,0,c]];
  return [[c,-s,0],[s,c,0],[0,0,1]];
}
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
function project(Xw, cam){
  const T=cam.T;
  const c=[T[0]*Xw[0]+T[1]*Xw[1]+T[2]*Xw[2]+T[3],
           T[4]*Xw[0]+T[5]*Xw[1]+T[6]*Xw[2]+T[7],
           T[8]*Xw[0]+T[9]*Xw[1]+T[10]*Xw[2]+T[11]];
  if(c[2]<=1) return null;
  return [cam.fx*c[0]/c[2]+cam.ppx, cam.fy*c[1]/c[2]+cam.ppy];
}

/* ---- the pose a side is currently at ---------------------------------- */
/* Composition matches react_toolbox.synth_actions.make_rotation_set:
   the gel centre is the pivot, the delta pre-multiplies (world frame), and
   the published origin is back-solved from the pivot. Rotating the origin
   instead would drag the gel 30-50 mm sideways. */
function poseOf(side){
  const p = D.pose[side], g = D.gel[side];
  const R0 = quatMat(p.slice(3,7));
  const pivot = add([p[0]*1000, p[1]*1000, p[2]*1000], mv(R0, g));
  const st = S.d[side];
  const R = mm(st.R, R0);
  const gel = add(pivot, st.t);
  return {R, gel, origin: sub(gel, mv(R, g)), quat: matQuat(R)};
}
function pointsFor(side, view){
  const cam = D.cams[view], g = D.gel[side], P = poseOf(side);
  const tips = [];
  for(let k=0;k<3;k++){const e=[0,0,0]; e[k]=60;
    tips.push(project(add(P.origin, mv(P.R, add(g,e))), cam));}
  return {centre: project(P.gel, cam), origin: project(P.origin, cam), tips};
}

/* ---- drawing ----------------------------------------------------------- */
function gizmo(g, view, w){
  if(view!=="middle") return;
  const T=D.cams[view].T, ox=88, oy=84, L=42;
  g.save();
  g.globalAlpha=.55; g.fillStyle="#12161f";
  g.beginPath(); g.arc(ox,oy,L+11,0,7); g.fill(); g.globalAlpha=1;
  g.strokeStyle="#46505f"; g.lineWidth=1.5; g.beginPath();
  g.arc(ox,oy,L+11,0,7); g.stroke();
  const R=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]];
  for(let k=0;k<3;k++){
    const e=[0,0,0]; e[k]=1; const d=mv(R,e);
    const inp=Math.hypot(d[0],d[1]);
    g.strokeStyle=AXC[k]; g.fillStyle=AXC[k]; g.lineWidth=3;
    if(inp>0.12){
      g.beginPath(); g.moveTo(ox,oy); g.lineTo(ox+d[0]*L, oy+d[1]*L); g.stroke();
      g.font="600 15px ui-monospace,monospace";
      g.fillText("xyz"[k], ox+d[0]*(L+14)-4, oy+d[1]*(L+14)+5);
    }
    if(Math.abs(d[2])>0.55){                  // out of plane: dot / cross
      g.beginPath(); g.arc(ox,oy,5,0,7); g.stroke();
      if(d[2]<0){g.beginPath(); g.arc(ox,oy,2,0,7); g.fill();}
      else{g.beginPath(); g.moveTo(ox-3,oy-3); g.lineTo(ox+3,oy+3);
           g.moveTo(ox-3,oy+3); g.lineTo(ox+3,oy-3); g.stroke();}
      g.font="600 15px ui-monospace,monospace";
      g.fillText("xyz"[k], ox+13, oy-11);
    }
  }
  g.fillStyle="#dfe6ee"; g.font="600 13px ui-sans-serif";
  g.fillText("world ("+D.up_axis+"-up)", ox-48, oy+L+30);
  g.restore();
}
function drawView(cv, view){
  const g=cv.getContext("2d");
  g.clearRect(0,0,cv.width,cv.height);
  for(const side of ["left","right"]){
    const r=pointsFor(side,view);
    if(!r.centre) continue;
    if(r.origin){                                  // stem to the marker cluster
      g.strokeStyle="rgba(200,200,200,.55)"; g.lineWidth=1;
      g.beginPath(); g.moveTo(r.origin[0],r.origin[1]);
      g.lineTo(r.centre[0],r.centre[1]); g.stroke();
      g.fillStyle="rgba(220,220,220,.75)";
      g.beginPath(); g.arc(r.origin[0],r.origin[1],2.5,0,7); g.fill();
    }
    for(let k=0;k<3;k++){const t=r.tips[k]; if(!t) continue;
      g.strokeStyle=AXC[k]; g.lineWidth=(side===S.side)?3.2:1.6;
      g.beginPath(); g.moveTo(r.centre[0],r.centre[1]);
      g.lineTo(t[0],t[1]); g.stroke();}
    g.fillStyle=COL[side];
    g.beginPath(); g.arc(r.centre[0],r.centre[1],5,0,7); g.fill();
    g.strokeStyle=(side===S.side)?"#fff":"rgba(255,255,255,.45)";
    g.lineWidth=(side===S.side)?2:1; g.stroke();
    g.fillStyle="#fff"; g.font="600 15px ui-sans-serif";
    g.fillText(side[0].toUpperCase()+(side===S.side?" *":""),
               r.centre[0]+8, r.centre[1]-8);
  }
  gizmo(g, view, cv.width);
}
function redraw(){
  for(const v of ["left","middle","right"]){
    const cv=document.querySelector(`[data-stream="view_${v}"] canvas`);
    if(cv) drawView(cv, v);
  }
  const st=S.d[S.side], P=poseOf(S.side);
  const tr=Math.acos(Math.min(1,Math.max(-1,(st.R[0][0]+st.R[1][1]+st.R[2][2]-1)/2)));
  document.getElementById("side").textContent=S.side;
  document.getElementById("tr").textContent =
    st.t.map(v=>v.toFixed(1)).join(", ")+" mm";
  document.getElementById("rt").textContent=(tr*180/Math.PI).toFixed(1)+"°";
  document.getElementById("gc").textContent =
    P.gel.map(v=>v.toFixed(1)).join(", ")+" mm";
}

/* ---- input ------------------------------------------------------------- */
const TRANS={ArrowRight:[0,+1],ArrowLeft:[0,-1],ArrowUp:[1,+1],ArrowDown:[1,-1],
             BracketRight:[2,+1],BracketLeft:[2,-1]};
const ROT={KeyD:[0,+1],KeyA:[0,-1],KeyW:[1,+1],KeyS:[1,-1],
           KeyE:[2,+1],KeyQ:[2,-1]};
function apply(code){
  if(code==="Tab"){S.side = S.side==="right" ? "left" : "right"; redraw(); return true;}
  if(code==="KeyR"){reset(); redraw(); return true;}
  const st=S.d[S.side];
  if(TRANS[code]){const [k,s]=TRANS[code]; st.t[k]+=s*D.step_mm; redraw(); return true;}
  if(ROT[code]){const [k,s]=ROT[code];
    st.R = mm(axisRot(k, s*D.step_deg), st.R); redraw(); return true;}
  return false;
}
function reset(){
  S.d={left:{t:[0,0,0],R:[[1,0,0],[0,1,0],[0,0,1]]},
       right:{t:[0,0,0],R:[[1,0,0],[0,1,0],[0,0,1]]}};
}

/* ---- build ------------------------------------------------------------- */
function tile(stream, frozen){
  const [w,h]=D.size[stream];
  const d=document.createElement("div");
  d.className="tile"; d.dataset.stream=stream;
  // The cameras are what you drive; the tactile pair is frozen. Giving them
  // equal area put the most weight on the one thing that does NOT respond.
  d.style.width=(frozen?230:Math.min(w,470))+"px";
  d.innerHTML=`<img src="${stream}.jpg" width="${w}" height="${h}" alt="${stream}">`
    +`<canvas width="${w}" height="${h}"></canvas>`
    +`<span class="cap">${stream.replace("_"," ")}</span>`
    +(frozen?`<span class="frozen" data-frozen>frozen — not simulated</span>`:``);
  return d;
}
/* The data is INLINED, not fetched. A page that fetch()es its own JSON is
   blank when opened from file:// -- Chrome blocks the request as
   cross-origin -- so it would only ever work on the Space. sim.json is still
   written beside it for tools and tests, and test_sim.py asserts the two
   copies are identical so they cannot drift. */
(function(d){
  D=d; S={side:"right", d:null}; reset();
  const vs=document.getElementById("views"), ts=document.getElementById("tactile");
  for(const v of ["left","middle","right"]) vs.appendChild(tile("view_"+v,false));
  for(const s of ["left","right"]) ts.appendChild(tile("tactile_"+s,true));
  document.getElementById("ep").textContent=D.key;
  document.getElementById("rw").textContent=D.row;
  document.getElementById("ua").textContent=D.up_axis+"-up";
  redraw();
  addEventListener("keydown", e=>{ if(apply(e.code)) e.preventDefault(); });
  /* Test hooks. __sim() reports the state the page is actually drawing from,
     so a test compares the PAGE against the library rather than against the
     page's own arithmetic. */
  window.__key = code => apply(code);
  window.__sim = () => {
    const P=poseOf(S.side), O=poseOf(S.side==="right"?"left":"right");
    const pts={};
    for(const v of ["left","middle","right"]){
      pts[v]={};
      for(const s of ["left","right"]) pts[v][s]=pointsFor(s,v);
    }
    return {side:S.side, gel_mm:P.gel, origin_mm:P.origin, quat:P.quat,
            other_gel_mm:O.gel, t_mm:S.d[S.side].t, points:pts};
  };
})(__DATA__);
</script>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
