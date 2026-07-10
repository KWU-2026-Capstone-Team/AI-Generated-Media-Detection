#!/usr/bin/env python3
"""딥페이크 판별 웹앱 — 영상 업로드 → real/fake + 점수.
werkzeug 기반(추가 설치 불필요). infer.py의 추론 로직 재사용.
실행:  python app.py            (기본 http://0.0.0.0:7860)
       python app.py --port 8000
"""
import argparse, json, os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from infer import load_models, faces_from_video, predict

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[web] 모델 로딩중... (device={DEV.type})", flush=True)
CONV, TEMP = load_models(DEV)
LOCK = threading.Lock()
print(f"[web] 준비 완료 — 공간 {len(CONV)} + 시간 {len(TEMP)} 모델", flush=True)

PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>딥페이크 판별기</title>
<style>
:root{--bg:#090d12;--surface:#111820;--surface2:#151d27;--sunken:#0d131a;--ink:#e8eff5;--ink2:#b4c1cd;
--muted:#7a8896;--line:#212c38;--line2:#2c3947;--accent:#2dd4ee;--accent-soft:rgba(45,212,238,.13);
--wall:#f0788c;--wall-soft:rgba(240,120,140,.13);--solved:#37d99e;--solved-soft:rgba(55,217,158,.14);
--fs:system-ui,-apple-system,"Segoe UI","Apple SD Gothic Neo","Noto Sans KR",sans-serif;
--fm:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:light){:root{--bg:#eaeef3;--surface:#fff;--surface2:#f2f5f9;--sunken:#e4e9f0;
--ink:#0e141b;--ink2:#33404d;--muted:#647180;--line:#d6dee7;--line2:#c3cdd9;--accent:#0e7490;
--accent-soft:rgba(14,116,144,.1);--wall:#c23a52;--wall-soft:rgba(194,58,82,.1);--solved:#0e9f6e;--solved-soft:rgba(14,159,110,.12)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--fs);
line-height:1.6;-webkit-font-smoothing:antialiased;min-height:100vh}
.wrap{max-width:640px;margin:0 auto;padding:56px 22px 80px}
.eyebrow{font-family:var(--fm);font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);
font-weight:600;display:flex;align-items:center;gap:10px;margin-bottom:14px}
.eyebrow::before{content:"";width:22px;height:1px;background:var(--accent)}
h1{font-size:clamp(1.8rem,5vw,2.4rem);letter-spacing:-.02em;margin:0 0 10px;line-height:1.12}
.sub{color:var(--ink2);margin:0 0 32px}
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;
border:1.5px dashed var(--line2);border-radius:18px;background:var(--surface);padding:44px 24px;text-align:center;
cursor:pointer;transition:border-color .2s,background .2s}
.drop:hover,.drop.over{border-color:var(--accent);background:var(--accent-soft)}
.drop .ic{font-size:34px;margin-bottom:12px}
.drop .t{font-weight:600}.drop .s{color:var(--muted);font-size:.9em;font-family:var(--fm);margin-top:6px}
.file{margin-top:14px;font-family:var(--fm);font-size:13px;color:var(--ink2);display:none}
.btn{margin-top:18px;width:100%;border:0;border-radius:12px;background:var(--accent);color:#04222b;
font-weight:700;font-size:1rem;padding:14px;cursor:pointer;font-family:var(--fs);display:none}
.btn:disabled{opacity:.6;cursor:default}
.stage{margin-top:26px}
.loading{display:none;text-align:center;color:var(--muted);font-family:var(--fm);font-size:14px;padding:24px}
.spin{width:30px;height:30px;border:3px solid var(--line2);border-top-color:var(--accent);border-radius:50%;
margin:0 auto 14px;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.result{display:none;background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:hidden}
.verdict{padding:26px;text-align:center;border-bottom:1px solid var(--line)}
.verdict.fake{background:var(--wall-soft)}.verdict.real{background:var(--solved-soft)}
.verdict .big{font-size:2.4rem;font-weight:800;letter-spacing:-.02em}
.verdict.fake .big{color:var(--wall)}.verdict.real .big{color:var(--solved)}
.verdict .pct{font-family:var(--fm);color:var(--ink2);margin-top:4px;font-size:.95em}
.verdict .rsn{font-family:var(--fm);font-size:12px;color:var(--muted);margin-top:10px}
.branches{padding:22px 26px}
.brow{display:grid;grid-template-columns:92px 1fr 52px;align-items:center;gap:12px;margin:14px 0}
.brow .bl{font-family:var(--fm);font-size:12px;color:var(--ink2)}
.track{height:12px;background:var(--sunken);border-radius:100px;overflow:hidden}
.fill{height:100%;border-radius:100px;width:0;transition:width .9s cubic-bezier(.2,.7,.2,1)}
.fill.sp{background:var(--accent)}.fill.tp{background:var(--solved)}
.bv{font-family:var(--fm);font-size:13px;font-weight:700;text-align:right}
.hint{padding:0 26px 22px;color:var(--muted);font-size:.86em}.hint b{color:var(--ink2)}
.err{display:none;background:var(--wall-soft);color:var(--wall);border:1px solid var(--wall);border-radius:12px;
padding:16px;font-family:var(--fm);font-size:13px;margin-top:20px}
.foot{margin-top:34px;font-family:var(--fm);font-size:11.5px;color:var(--muted);text-align:center;line-height:1.7}
</style></head><body><div class="wrap">
<div class="eyebrow">공간 × 시간 이중브랜치</div>
<h1>딥페이크 판별기</h1>
<p class="sub">영상을 올리면 얼굴을 검출해 <b>real / fake</b>를 판정하고 신뢰 점수를 냅니다.</p>

<label class="drop" id="drop">
  <div class="ic">🎬</div>
  <div class="t">영상을 끌어다 놓거나 클릭해서 선택</div>
  <div class="s">mp4 · mov · avi</div>
  <input id="inp" type="file" accept="video/*" hidden>
  <div class="file" id="fname"></div>
</label>
<button class="btn" id="go">분석하기</button>

<div class="stage">
  <div class="loading" id="load"><div class="spin"></div>분석 중… (얼굴 검출 + 이중브랜치 추론)<br>수 초~수십 초 걸릴 수 있어요</div>
  <div class="err" id="err"></div>
  <div class="result" id="res">
    <div class="verdict" id="vd"><div class="big" id="vbig"></div><div class="pct" id="vpct"></div><div class="rsn" id="vrsn"></div></div>
    <div class="branches">
      <div class="brow"><span class="bl">공간 (외형)</span><div class="track"><div class="fill sp" id="spf"></div></div><span class="bv" id="spv"></span></div>
      <div class="brow"><span class="bl">시공간 (동역학)</span><div class="track"><div class="fill tp" id="tpf"></div></div><span class="bv" id="tpv"></span></div>
    </div>
    <div class="hint" id="hint"></div>
  </div>
</div>
<div class="foot">누수0 in-dist AUC 0.940 · FF++ 4기법 전부 ≥0.89<br>공간=ConvNeXt 전체얼굴 · 시공간=3D-CNN 입영역 클립</div>
</div>
<script>
const inp=document.getElementById('inp'),drop=document.getElementById('drop'),go=document.getElementById('go'),
fname=document.getElementById('fname'),load=document.getElementById('load'),res=document.getElementById('res'),err=document.getElementById('err');
let file=null;
function pick(f){file=f;fname.textContent='📁 '+f.name;fname.style.display='block';go.style.display='block';res.style.display='none';err.style.display='none';}
inp.onchange=e=>{if(e.target.files[0])pick(e.target.files[0])};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])pick(e.dataTransfer.files[0])};
go.onclick=async()=>{
  if(!file)return;
  go.disabled=true;load.style.display='block';res.style.display='none';err.style.display='none';
  const fd=new FormData();fd.append('video',file);
  try{
    const r=await fetch('/api/predict',{method:'POST',body:fd});const d=await r.json();
    load.style.display='none';go.disabled=false;
    if(d.error){err.textContent='⚠ '+d.error;err.style.display='block';return;}
    render(d);
  }catch(e){load.style.display='none';go.disabled=false;err.textContent='⚠ 요청 실패: '+e;err.style.display='block';}
};
function render(d){
  const fake=d.verdict==='FAKE';
  document.getElementById('vd').className='verdict '+(fake?'fake':'real');
  document.getElementById('vbig').textContent=fake?'🔴 FAKE':'🟢 REAL';
  document.getElementById('vpct').textContent='종합 fake 확률 '+(d.fused*100).toFixed(1)+'%  ·  얼굴 '+d.n_faces+'프레임';
  document.getElementById('vrsn').textContent='근거: '+d.reason+' (mean≥0.5 또는 브랜치≥0.85 → fake)';
  const spv=document.getElementById('spv'),tpv=document.getElementById('tpv');
  spv.textContent=d.spatial.toFixed(3);tpv.textContent=d.temporal.toFixed(3);
  setTimeout(()=>{document.getElementById('spf').style.width=(d.spatial*100)+'%';document.getElementById('tpf').style.width=(d.temporal*100)+'%';},60);
  let h='';
  if(fake){
    if(d.temporal>=0.85&&d.spatial<0.5)h='<b>시공간 브랜치</b>가 포착 — 얼굴 동역학(입 텍스처 flicker)의 이상. NeuralTextures·미지 조작에 강한 축.';
    else if(d.spatial>=0.85&&d.temporal<0.5)h='<b>공간 브랜치</b>가 포착 — 프레임 외형 아티팩트.';
    else h='두 브랜치가 함께 fake 신호를 냄.';
  }else h='두 브랜치 모두 낮음 → real로 판정.';
  document.getElementById('hint').innerHTML=h;
  res.style.display='block';
}
</script></body></html>"""

def application(environ, start_response):
    req = Request(environ)
    req.max_content_length = 500 * 1024 * 1024  # 500MB
    if req.path == "/" and req.method == "GET":
        return Response(PAGE, mimetype="text/html")(environ, start_response)
    if req.path == "/api/predict" and req.method == "POST":
        f = req.files.get("video")
        if f is None:
            return Response(json.dumps({"error": "영상 파일이 없습니다"}), mimetype="application/json")(environ, start_response)
        suffix = os.path.splitext(f.filename or "")[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix); f.save(tmp.name); tmp.close()
        try:
            faces = faces_from_video(tmp.name)
            if len(faces) < 4:
                res = {"error": f"얼굴을 충분히 검출하지 못했습니다 ({len(faces)}프레임). 얼굴이 또렷한 영상을 올려주세요."}
            else:
                with LOCK:
                    sp, tp, fused = predict(faces, CONV, TEMP, DEV)
                conf = max(sp, tp); is_fake = (fused >= 0.5) or (conf >= 0.85)
                reason = "종합 확률" if fused >= 0.5 else ("공간 브랜치" if sp >= 0.85 else "시공간 브랜치") if is_fake else "-"
                res = {"spatial": round(sp, 3), "temporal": round(tp, 3), "fused": round(fused, 3),
                       "verdict": "FAKE" if is_fake else "REAL", "reason": reason, "n_faces": len(faces)}
        except Exception as e:
            res = {"error": f"처리 오류: {e}"}
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass
        return Response(json.dumps(res), mimetype="application/json")(environ, start_response)
    return Response("not found", status=404)(environ, start_response)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7860); ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"[web] http://{a.host}:{a.port}  (브라우저에서 열기)", flush=True)
    run_simple(a.host, a.port, application, threaded=True)
