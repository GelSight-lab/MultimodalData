"""Interactive calibration inspector: 5 timestamps x 3 views, overlay in the browser.

WHY IN THE BROWSER

A static contact sheet can only re-show what you already saw. The question
here is not "is it off" but "off by what, and in which frame" — and that is a
question about a hypothesis you want to vary. So the projection runs in
JavaScript from the same numbers the library uses: pose 7-vec, T_mocap_to_cam,
intrinsics, gel centre in the rigid frame. Drag a candidate offset and all
fifteen tiles move at once, which is the only way to tell a WORLD-frame error
(shifts every tile the same way in 3D) from a GEL-offset error (rotates with
each sensor's own pose).

The JS is a second implementation of the projection, so it can disagree with
the library. `scripts/test_calib_inspector.py` reads the projections back out
of the running page and compares them against `project_gel_frame`.

    python scripts/build_calib_inspector.py --batch a
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                                     # noqa: E402
import h5py                                                    # noqa: E402
import hdf5plugin                                              # noqa: E402,F401
import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

from react_toolbox.calibration import load_calibration          # noqa: E402
from twm.calib_epoch import calib_dir                           # noqa: E402

REL = Path("/media/yxma/Disk1/twm/release_force/motherboard/meta")
H5R = Path("/media/yxma/Disk1/twm/data/motherboard")
CAM_H5 = {"left": 1, "middle": 2, "right": 0}
VIEWS = ("left", "middle", "right")

# Two disjoint sets of five. Batch A is shown first; B is held back so the
# offset you settle on gets checked against frames it was not tuned on —
# picking the number and validating it on the same five would prove nothing.
BATCHES = {
    "a": [("episode_000", 657), ("episode_000", 180), ("episode_000", 520),
          ("episode_000", 900), ("episode_000", 1290)],
    "b": [("episode_001", 400), ("episode_002", 300), ("episode_002", 950),
          ("episode_003", 600), ("episode_004", 750)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="a", choices=sorted(BATCHES))
    ap.add_argument("--date", default="2026-05-19")
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/calib_inspector")
    args = ap.parse_args()

    out = Path(args.out) / args.batch
    (out / "img").mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)
    raw = {s: json.loads((calib_dir("motherboard") / f"T_gel_to_rigid_{s}.json").read_text())
           for s in ("left", "right")}

    data = {
        "date": args.date, "batch": args.batch,
        "cams": {v: {"T": np.asarray(cal["cams"][v]["T_mocap_to_cam"], float)[:3].ravel().tolist(),
                     "fx": cal["cams"][v]["intrinsics"]["fx"],
                     "fy": cal["cams"][v]["intrinsics"]["fy"],
                     "ppx": cal["cams"][v]["intrinsics"]["ppx"],
                     "ppy": cal["cams"][v]["intrinsics"]["ppy"],
                     "rmse_mm": cal["cams"][v]["rmse"]} for v in VIEWS},
        "gel": {s: np.asarray(cal[f"gel_{s}"], float).tolist() for s in ("left", "right")},
        "refball": {s: raw[s]["refball_center_in_rigid_mm"] for s in ("left", "right")},
        "gel_axis": {s: raw[s]["gel_axis_in_rigid"] for s in ("left", "right")},
        "frames": [],
    }

    for ep, row in BATCHES[args.batch]:
        p = REL / args.date / f"{ep}.parquet"
        t = pq.read_table(p).to_pydict()
        trim = int(np.asarray(t["source_h5_frame"])[0])
        rec = {"episode": ep, "row": row, "h5_frame": trim + row,
               "t_s": round(row / 30.0, 2), "img": {},
               "pose": {s: [float(x) for x in
                            np.asarray([y for y in t[f"sensor_{s}_pose"]], float)[row]]
                        for s in ("left", "right")},
               "force": {s: round(float(np.asarray(t[f"force_{s}_normal_n"], float)[row]), 2)
                         for s in ("left", "right")}}
        with h5py.File(str(H5R / args.date / f"{ep}.h5"), "r") as f:
            for v in VIEWS:
                img = f[f"realsense/cam{CAM_H5[v]}/color"][trim + row][..., ::-1]
                name = f"img/{ep}_{row}_{v}.jpg"
                cv2.imwrite(str(out / name), img[..., ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                rec["img"][v] = name
                rec["w"], rec["h"] = int(img.shape[1]), int(img.shape[0])
        data["frames"].append(rec)
        print(f"  {ep} row {row:5d}  h5 {trim+row:5d}  "
              f"F L{rec['force']['left']:5.2f} R{rec['force']['right']:5.2f}", flush=True)

    (out / "data.json").write_text(json.dumps(data))
    (out / "index.html").write_text(_page(data))
    print(f"\n{len(data['frames'])} timestamps x {len(VIEWS)} views -> {out}/index.html")
    return 0


def _page(d: dict) -> str:
    n = len(d["frames"])
    rm = " / ".join(f"{v} {d['cams'][v]['rmse_mm']:.1f}" for v in VIEWS)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibration inspector &mdash; __DATE__ batch __BATCH__</title><style>
:root{--bg:#0b1020;--fg:#e8eefb;--dim:#8ea0c2;--line:#1e2a45;--card:#111a2e;--accent:#ffc46b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 'IBM Plex Sans',system-ui,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:16px;color:var(--accent);margin:26px 0 8px}
p{color:var(--dim);max-width:74ch}
.panel{position:sticky;top:0;z-index:9;background:rgba(11,16,32,.96);
border-bottom:1px solid var(--line);padding:12px 0;backdrop-filter:blur(6px)}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.grp{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 12px}
.grp b{font-size:12px;color:var(--accent);display:block;margin-bottom:4px}
label{font-size:13px;color:var(--dim);margin-right:10px;white-space:nowrap}
input[type=range]{vertical-align:middle;width:120px}
.val{display:inline-block;min-width:44px;font-variant-numeric:tabular-nums;color:var(--fg)}
button{background:#1b2embed;background:#1b2a46;color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:6px 12px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
table{border-collapse:collapse;font-size:13px;margin:8px 0 0}
th,td{border-bottom:1px solid var(--line);padding:4px 12px 4px 0;text-align:left;
font-variant-numeric:tabular-nums}th{color:var(--dim);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:8px 0 22px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px}
figcaption{color:var(--dim);font-size:12px;margin-top:5px}
.tile{position:relative;line-height:0;cursor:zoom-in}
.tile img,.tile canvas{width:100%;border-radius:5px;display:block}
.tile canvas{position:absolute;inset:0}
.hdr{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.hdr small{color:var(--dim)}
dialog{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:12px;max-width:96vw;padding:10px}
dialog::backdrop{background:rgba(0,0,0,.8)}
.zoomwrap{position:relative;line-height:0}
.zoomwrap img,.zoomwrap canvas{width:min(1200px,92vw);display:block;border-radius:6px}
.zoomwrap canvas{position:absolute;inset:0}
</style></head><body><div class="wrap">
<h1>Calibration inspector &mdash; __DATE__, batch __BATCH__</h1>
<p>__N__ timestamps &times; 3 camera views. The overlay is computed in this page from the
published pose, <code>T_mocap_to_cam</code> and <code>gel_center_in_rigid_mm</code> &mdash; the same
numbers the library uses &mdash; so the sliders below move every tile at once.</p>
<p><b>The two hypotheses look different.</b> A wrong WORLD frame is a fixed 3&#8209;D shift:
it moves both sensors together and the apparent pixel shift changes with depth and view.
A wrong GEL offset lives in each sensor's own rigid frame, so it swings as that hand
rotates and does not touch the other hand. Camera reprojection rmse is __RMSE__&nbsp;mm.</p>
<div class="panel"><div class="row">
<div class="grp"><b>show</b>
<label><input type="checkbox" id="ov" checked> overlay</label>
<label><input type="checkbox" id="sL" checked> left</label>
<label><input type="checkbox" id="sR" checked> right</label>
<label><input type="checkbox" id="ax" checked> axes</label>
<label><input type="checkbox" id="st" checked> stem</label>
<label><input type="checkbox" id="gh" checked> ghost&nbsp;(uncorrected)</label>
</div>
<div class="grp"><b>world offset applied to both sensors (mm)</b>
<label>x<input type="range" id="wx" min="-60" max="60" step="1" value="0"><span class="val" id="wxv">0</span></label>
<label>y<input type="range" id="wy" min="-60" max="60" step="1" value="0"><span class="val" id="wyv">0</span></label>
<label>z<input type="range" id="wz" min="-60" max="60" step="1" value="0"><span class="val" id="wzv">0</span></label>
</div>
<div class="grp"><b>gel offset, <select id="gs"><option value="left">left</option><option value="right">right</option></select> rigid frame (mm)</b>
<label>x<input type="range" id="gx" min="-30" max="30" step="1" value="0"><span class="val" id="gxv">0</span></label>
<label>y<input type="range" id="gy" min="-30" max="30" step="1" value="0"><span class="val" id="gyv">0</span></label>
<label>z<input type="range" id="gz" min="-30" max="30" step="1" value="0"><span class="val" id="gzv">0</span></label>
</div>
<button id="reset">reset</button><button id="copy">copy settings</button>
</div></div>
<div id="tiles"></div>
<dialog id="dlg"><div class="zoomwrap"><img id="zimg"><canvas id="zcv"></canvas></div>
<div style="display:flex;justify-content:space-between;margin-top:8px">
<small id="zcap" style="color:var(--dim)"></small><button id="zclose">close</button></div></dialog>
</div>
<script>
const D = __DATA__;
const VIEWS = ["left","middle","right"];
const COL = {left:"#ffd23f", right:"#4fc3f7"};
const AXC = ["#ff5a5a","#5aff8f","#5aa8ff"];

function quatMat(q){                     // scipy from_quat order: x,y,z,w
  const [x,y,z,w]=q, n=Math.hypot(x,y,z,w)||1;
  const X=x/n,Y=y/n,Z=z/n,W=w/n;
  return [[1-2*(Y*Y+Z*Z), 2*(X*Y-Z*W),   2*(X*Z+Y*W)],
          [2*(X*Y+Z*W),   1-2*(X*X+Z*Z), 2*(Y*Z-X*W)],
          [2*(X*Z-Y*W),   2*(Y*Z+X*W),   1-2*(X*X+Y*Y)]];
}
function mv(M,v){return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
                         M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
                         M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];}
function project(Xw, cam){               // world mm -> pixel
  const T=cam.T;
  const c=[T[0]*Xw[0]+T[1]*Xw[1]+T[2]*Xw[2]+T[3],
           T[4]*Xw[0]+T[5]*Xw[1]+T[6]*Xw[2]+T[7],
           T[8]*Xw[0]+T[9]*Xw[1]+T[10]*Xw[2]+T[11]];
  if(c[2]<=1) return null;
  return [cam.fx*c[0]/c[2]+cam.ppx, cam.fy*c[1]/c[2]+cam.ppy, c[2]];
}
function S(){return {
  ov:ov.checked, L:sL.checked, R:sR.checked, ax:ax.checked, st:st.checked, gh:gh.checked,
  world:[+wx.value,+wy.value,+wz.value], side:gs.value,
  gel:[+gx.value,+gy.value,+gz.value]};}

function points(fr, side, view, s, useDelta){
  const cam=D.cams[view], p=fr.pose[side];
  const R=quatMat(p.slice(3,7));
  const wd = useDelta ? s.world : [0,0,0];
  const gd = (useDelta && s.side===side) ? s.gel : [0,0,0];
  const org=[p[0]*1000+wd[0], p[1]*1000+wd[1], p[2]*1000+wd[2]];
  const gel=D.gel[side].map((v,i)=>v+gd[i]);
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const out={centre:project(add(org,mv(R,gel)),cam), origin:project(org,cam), tips:[]};
  for(let k=0;k<3;k++){const e=[0,0,0];e[k]=60;
    out.tips.push(project(add(org,mv(R,gel.map((v,i)=>v+e[i]))),cam));}
  return out;
}
function drawOn(cv, fr, view, s, scale){
  const g=cv.getContext("2d"); g.clearRect(0,0,cv.width,cv.height);
  if(!s.ov) return;
  for(const side of ["left","right"]){
    if(side==="left"&&!s.L) continue;
    if(side==="right"&&!s.R) continue;
    if(s.gh && (s.world.some(v=>v)||(s.side===side&&s.gel.some(v=>v)))){
      const q=points(fr,side,view,s,false);
      if(q.centre){g.strokeStyle="rgba(255,255,255,.35)";g.lineWidth=1;
        g.beginPath();g.arc(q.centre[0]*scale,q.centre[1]*scale,5*scale,0,7);g.stroke();}
    }
    const r=points(fr,side,view,s,true);
    if(!r.centre) continue;
    const cx=r.centre[0]*scale, cy=r.centre[1]*scale;
    if(s.st&&r.origin){g.strokeStyle="rgba(190,190,190,.9)";g.lineWidth=1*scale;
      g.beginPath();g.moveTo(r.origin[0]*scale,r.origin[1]*scale);g.lineTo(cx,cy);g.stroke();}
    if(s.ax){for(let k=0;k<3;k++){const t=r.tips[k]; if(!t) continue;
      g.strokeStyle=AXC[k];g.lineWidth=2*scale;
      g.beginPath();g.moveTo(cx,cy);g.lineTo(t[0]*scale,t[1]*scale);g.stroke();}}
    g.fillStyle=COL[side];g.beginPath();g.arc(cx,cy,3.2*scale,0,7);g.fill();
    g.strokeStyle="#fff";g.lineWidth=1*scale;g.stroke();
  }
}
const cvs=[];
function build(){
  const host=document.getElementById("tiles");
  D.frames.forEach((fr,i)=>{
    const sec=document.createElement("section");
    sec.innerHTML=`<h2>t${i+1} &mdash; ${fr.episode}, row ${fr.row} (h5 frame ${fr.h5_frame}, `
      +`t=${fr.t_s}s) &nbsp; force L ${fr.force.left} N / R ${fr.force.right} N</h2>`;
    const gr=document.createElement("div"); gr.className="grid";
    VIEWS.forEach(v=>{
      const f=document.createElement("figure");
      f.innerHTML=`<div class="tile"><img src="${fr.img[v]}" width="${fr.w}" height="${fr.h}">`
        +`<canvas width="${fr.w}" height="${fr.h}"></canvas></div>`
        +`<figcaption>${v} camera &middot; rmse ${D.cams[v].rmse_mm.toFixed(1)} mm</figcaption>`;
      gr.appendChild(f);
      const cv=f.querySelector("canvas");
      cvs.push({cv, fr, view:v});
      f.querySelector(".tile").onclick=()=>zoom(fr,v);
    });
    sec.appendChild(gr); host.appendChild(sec);
  });
}
function redraw(){
  const s=S();
  ["wx","wy","wz","gx","gy","gz"].forEach(k=>
    document.getElementById(k+"v").textContent=document.getElementById(k).value);
  cvs.forEach(o=>drawOn(o.cv,o.fr,o.view,s,1));
  if(dlg.open&&dlg._fr) drawZoom();
}
function zoom(fr,v){dlg._fr=fr;dlg._v=v;zimg.src=fr.img[v];
  zcap.textContent=`${fr.episode} row ${fr.row} — ${v} camera`;
  zimg.onload=()=>{zcv.width=fr.w;zcv.height=fr.h;drawZoom();};
  if(zimg.complete){zcv.width=fr.w;zcv.height=fr.h;drawZoom();}
  dlg.showModal();}
function drawZoom(){drawOn(zcv,dlg._fr,dlg._v,S(),1);}
zclose.onclick=()=>dlg.close();
document.getElementById("reset").onclick=()=>{
  ["wx","wy","wz","gx","gy","gz"].forEach(k=>document.getElementById(k).value=0);redraw();};
document.getElementById("copy").onclick=()=>{
  const s=S();navigator.clipboard.writeText(JSON.stringify(
    {date:D.date,batch:D.batch,world_offset_mm:s.world,gel_side:s.side,gel_delta_mm:s.gel}));
  document.getElementById("copy").textContent="copied";
  setTimeout(()=>document.getElementById("copy").textContent="copy settings",1200);};
document.querySelectorAll("input,select").forEach(e=>{
  e.addEventListener("input",redraw);e.addEventListener("change",redraw);});
build(); redraw();
window.__probe=(i,view,side)=>{const r=points(D.frames[i],side,view,S(),true);
  return r.centre?[r.centre[0],r.centre[1]]:null;};
</script></body></html>""" \
        .replace("__DATA__", json.dumps(d)) \
        .replace("__DATE__", d["date"]).replace("__BATCH__", d["batch"].upper()) \
        .replace("__N__", str(n)).replace("__RMSE__", rm)


if __name__ == "__main__":
    raise SystemExit(main())
