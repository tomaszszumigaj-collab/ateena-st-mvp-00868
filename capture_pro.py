
from __future__ import annotations

import json
from typing import Literal

import streamlit as st
import streamlit.components.v1 as components


def render_capture_pro_component(gender: str = 'kobieta', allow_selfie: bool = False, key_suffix: str = 'main') -> None:
    guide_gender = 'female' if gender == 'kobieta' else 'male'
    html = f"""
    <div id="ateena-capture-root-{key_suffix}" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#fff;border:1px solid #d7e0ea;border-radius:18px;padding:12px;">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
        <div>
          <strong>Capture Pro — mobile flow beta</strong>
          <div style="font-size:12px;color:#475569;">Live kamera + landmarki + ostrzejszy quality gate ACCEPT / RETRY / REJECT</div>
        </div>
        <span style="font-size:12px;background:#ccfbf1;color:#115e59;padding:4px 8px;border-radius:999px;">beta</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
        <button id="frontBtn-{key_suffix}" style="padding:10px;border-radius:10px;border:1px solid #cbd5e1;background:#e7faf8;font-weight:700;">FRONT</button>
        <button id="profileBtn-{key_suffix}" style="padding:10px;border-radius:10px;border:1px solid #cbd5e1;background:#fff;font-weight:700;">PROFIL</button>
      </div>
      <div style="font-size:12px;color:#334155;margin-bottom:8px;">Telefon trzymaj na wysokości bioder. Ręce lekko odsunięte od ciała. {'Selfie pełnej sylwetki traktuj jako awaryjne.' if allow_selfie else 'Najlepiej, żeby zdjęcie robiła druga osoba lub telefon był stabilnie ustawiony.'}</div>
      <div style="position:relative;background:#0f172a;border-radius:16px;overflow:hidden;aspect-ratio:9/16;">
        <video id="video-{key_suffix}" playsinline autoplay muted style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover"></video>
        <canvas id="overlay-{key_suffix}" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
        <div style="position:absolute;left:10px;right:10px;bottom:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
          <span id="status-{key_suffix}" style="background:rgba(255,255,255,.92);color:#0f172a;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;">kamera nieaktywna</span>
          <span id="orientation-{key_suffix}" style="background:rgba(255,255,255,.92);color:#0f172a;padding:6px 10px;border-radius:999px;font-size:12px;">orientacja: —</span>
          <span id="score-{key_suffix}" style="background:rgba(255,255,255,.92);color:#0f172a;padding:6px 10px;border-radius:999px;font-size:12px;">score: —</span>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:10px 0;">
        <button id="start-{key_suffix}" style="padding:10px;border-radius:10px;border:none;background:#14b8a6;color:#06231f;font-weight:800;">Start</button>
        <button id="capture-{key_suffix}" disabled style="padding:10px;border-radius:10px;border:none;background:#e2e8f0;color:#334155;font-weight:800;">Capture</button>
        <button id="stop-{key_suffix}" disabled style="padding:10px;border-radius:10px;border:none;background:#e2e8f0;color:#334155;font-weight:800;">Stop</button>
      </div>
      <div id="messages-{key_suffix}" style="display:grid;gap:6px;"></div>
      <div id="checks-{key_suffix}" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;"></div>
      <div style="font-size:11px;color:#64748b;margin-top:8px;">Ten komponent pomaga ustawić ujęcie i wychwycić oczywiste błędy capture. W V7.8 helper jest bardziej rygorystyczny: measurement_ready wymaga kompletu landmarków, pełnej sylwetki, stóp, dłoni, poprawnej orientacji i neutralnej perspektywy.</div>
    </div>
    <script type="module">
      import {{ PoseLandmarker, FilesetResolver }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
      const key = {json.dumps(key_suffix)};
      const els = {{
        video: document.getElementById(`video-${{key}}`),
        overlay: document.getElementById(`overlay-${{key}}`),
        start: document.getElementById(`start-${{key}}`),
        stop: document.getElementById(`stop-${{key}}`),
        capture: document.getElementById(`capture-${{key}}`),
        frontBtn: document.getElementById(`frontBtn-${{key}}`),
        profileBtn: document.getElementById(`profileBtn-${{key}}`),
        status: document.getElementById(`status-${{key}}`),
        orientation: document.getElementById(`orientation-${{key}}`),
        score: document.getElementById(`score-${{key}}`),
        messages: document.getElementById(`messages-${{key}}`),
        checks: document.getElementById(`checks-${{key}}`),
      }};
      let mode = 'front';
      let poseLandmarker = null;
      let stream = null;
      let rafId = null;
      const ctx = els.overlay.getContext('2d');
      const IDX = {{ nose:0, leftShoulder:11,rightShoulder:12,leftElbow:13,rightElbow:14,leftWrist:15,rightWrist:16,leftHip:23,rightHip:24,leftKnee:25,rightKnee:26,leftAnkle:27,rightAnkle:28,leftFoot:31,rightFoot:32 }};
      function visible(lm, idx, t=0.55) {{ return !!lm[idx] && (lm[idx].visibility ?? 1) >= t; }}
      function p(lm, idx) {{ return lm[idx] ? {{x: lm[idx].x, y: lm[idx].y}} : null; }}
      function angle(a,b) {{ return Math.atan2((b.y-a.y),(b.x-a.x))*180/Math.PI; }}
      function angAt(a,b,c) {{ const ab={{x:a.x-b.x,y:a.y-b.y}}, cb={{x:c.x-b.x,y:c.y-b.y}}; const dot=ab.x*cb.x+ab.y*cb.y; const mag=Math.hypot(ab.x,ab.y)*Math.hypot(cb.x,cb.y); if(!mag) return 0; return Math.acos(Math.max(-1,Math.min(1,dot/mag)))*180/Math.PI; }}
      function setMode(m) {{ mode=m; els.frontBtn.style.background=m==='front' ? '#e7faf8' : '#fff'; els.profileBtn.style.background=m==='profile' ? '#e7faf8' : '#fff'; }}
      els.frontBtn.onclick=()=>setMode('front');
      els.profileBtn.onclick=()=>setMode('profile');
      function msg(text) {{ const d=document.createElement('div'); d.style.cssText='padding:8px 10px;border-radius:10px;background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;font-size:13px'; d.textContent=text; return d; }}
      function chk(label, ok, warn=false) {{ const d=document.createElement('div'); d.style.cssText='padding:8px 10px;border-radius:10px;border:1px solid ' + (ok ? '#a7f3d0' : warn ? '#fde68a' : '#fecaca') + ';background:' + (ok ? '#ecfdf5' : warn ? '#fffbeb' : '#fef2f2') + ';color:' + (ok ? '#065f46' : warn ? '#92400e' : '#991b1b') + ';font-size:12px;font-weight:700'; d.textContent=(ok?'✓ ':'✕ ') + label; return d; }}
      function resizeCanvas() {{ const r=els.video.getBoundingClientRect(); els.overlay.width=r.width; els.overlay.height=r.height; }}
      async function initPose() {{
        if (poseLandmarker) return;
        const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {{
          baseOptions: {{ modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' }},
          runningMode:'VIDEO', numPoses:1, outputSegmentationMasks:false,
        }});
      }}
      function evaluate(lm) {{
        const nose=p(lm,IDX.nose), ls=p(lm,IDX.leftShoulder), rs=p(lm,IDX.rightShoulder), lw=p(lm,IDX.leftWrist), rw=p(lm,IDX.rightWrist), lh=p(lm,IDX.leftHip), rh=p(lm,IDX.rightHip), lk=p(lm,IDX.leftKnee), rk=p(lm,IDX.rightKnee), la=p(lm,IDX.leftAnkle), ra=p(lm,IDX.rightAnkle), lf=p(lm,IDX.leftFoot), rf=p(lm,IDX.rightFoot);
        const pts=[nose,ls,rs,lw,rw,lh,rh,lk,rk,la,ra,lf,rf].filter(Boolean);
        const minY=Math.min(...pts.map(v=>v.y)), maxY=Math.max(...pts.map(v=>v.y));
        const bodyHeight=maxY-minY;
        const head=visible(lm, IDX.nose), feet=visible(lm, IDX.leftFoot) && visible(lm, IDX.rightFoot), wrists=visible(lm, IDX.leftWrist,0.35) && visible(lm, IDX.rightWrist,0.35), hips=visible(lm, IDX.leftHip) && visible(lm, IDX.rightHip), knees=visible(lm, IDX.leftKnee) && visible(lm, IDX.rightKnee);
        const shoulderDist=(ls&&rs)?Math.abs(ls.x-rs.x):0, hipDist=(lh&&rh)?Math.abs(lh.x-rh.x):0;
        let orientation='półprofil'; if(shoulderDist>0.16 && hipDist>0.12) orientation='front'; else if(shoulderDist<0.10 && hipDist<0.08) orientation='profil';
        let standing=false,sitting=false; if(lh&&lk&&la&&rh&&rk&&ra) {{ const ak1=angAt(lh,lk,la), ak2=angAt(rh,rk,ra); standing=ak1>150 && ak2>150; sitting=ak1<135 || ak2<135; }}
        let roll=0; if(ls&&rs) {{ roll=Math.abs(angle(ls,rs)); if(roll>90) roll=Math.abs(roll-180); }}
        let centerOk=false; if(ls&&rs&&lh&&rh) {{ const cx=((ls.x+rs.x+lh.x+rh.x)/4); centerOk=cx>0.38 && cx<0.62; }}
        let armsOk=false; if(lw&&rw&&lh&&rh) {{ const lg=Math.abs(lw.x-lh.x), rg=Math.abs(rw.x-rh.x); armsOk = lg>0.035 && rg>0.035 && lg<0.20 && rg<0.20; }}
        const loose = {json.dumps(bool(allow_selfie))} ? false : false;
        const checks = [
          ['głowa w kadrze', head], ['stopy w kadrze', feet], ['biodra wykryte', hips], ['kolana wykryte', knees], ['nadgarstki / dłonie', wrists], ['pełna sylwetka', bodyHeight>0.68], ['użytkownik stoi', standing && !sitting], ['ciało wycentrowane', centerOk], ['ręce lekko odsunięte', armsOk], ['kamera bez przechyłu', roll<=7], [mode==='front'?'orientacja = front':'orientacja = profil', mode==='front'?orientation==='front':orientation==='profil']
        ];
        const messages=[];
        if(!head) messages.push('Pokaż całą głowę / twarz.');
        if(!feet) messages.push('Pokaż obie stopy w całości.');
        if(!wrists) messages.push('Pokaż obie dłonie lub nadgarstki.');
        if(!(standing && !sitting)) messages.push('Stań prosto — zdjęcie siedząc nie nadaje się do analizy.');
        if(!centerOk) messages.push('Stań bliżej środka kadru.');
        if(!armsOk) messages.push('Odsuń ręce od ciała o około 10–15 cm.');
        if(roll>7) messages.push('Wyprostuj telefon — kamera jest przechylona.');
        if(mode==='front' && orientation==='półprofil') messages.push('To nie jest pełny front — obróć ciało bardziej przodem.');
        if(mode==='profile' && orientation==='półprofil') messages.push('To nadal półprofil — obróć się bardziej bokiem.');
        let score = 0; checks.forEach(v => score += v[1] ? 9 : 0); if(bodyHeight>0.68) score += 10; if(roll<=7) score += 5; if(armsOk) score += 5;
        const hardReject = !head || !feet || !(standing && !sitting) || !(bodyHeight>0.68) || (mode==='front' && orientation!=='front') || (mode==='profile' && orientation!=='profil') || roll>15;
        let status='REJECT'; if(!hardReject && score>=85) status='ACCEPT'; else if(!hardReject && score>=70) status='RETRY';
        return {{score, status, orientation, checks, messages}};
      }}
      function renderEval(ev) {{
        els.checks.innerHTML=''; ev.checks.forEach(c=>els.checks.appendChild(chk(c[0], c[1], ev.status==='RETRY' && !c[1])));
        els.messages.innerHTML=''; (ev.messages.length?ev.messages:['Ujęcie wygląda poprawnie.']).slice(0,6).forEach(m=>els.messages.appendChild(msg(m)));
        els.orientation.textContent='orientacja: '+ev.orientation;
        els.score.textContent='score: '+Math.round(ev.score);
        els.status.textContent = ev.status === 'ACCEPT' ? 'ACCEPT — można zrobić zdjęcie' : ev.status === 'RETRY' ? 'RETRY — popraw ujęcie' : 'REJECT — zdjęcie nie nadaje się do analizy';
        els.status.style.background = ev.status === 'ACCEPT' ? 'rgba(16,185,129,.94)' : ev.status === 'RETRY' ? 'rgba(245,158,11,.94)' : 'rgba(239,68,68,.94)';
        els.status.style.color = ev.status === 'REJECT' ? '#fff' : '#052018';
        els.capture.disabled = ev.status !== 'ACCEPT';
      }}
      function draw(lm) {{
        ctx.clearRect(0,0,els.overlay.width, els.overlay.height);
        ctx.strokeStyle='rgba(20,184,166,.95)'; ctx.fillStyle='rgba(20,184,166,.95)'; ctx.lineWidth=3;
        const w=els.overlay.width,h=els.overlay.height;
        const x=w*0.18,y=h*0.05,ww=w*0.64,hh=h*0.88; ctx.setLineDash([8,6]); ctx.strokeStyle='rgba(255,255,255,.7)'; ctx.strokeRect(x,y,ww,hh); ctx.setLineDash([]); ctx.strokeStyle='rgba(20,184,166,.95)';
        const pairs=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,31],[24,26],[26,28],[28,32]];
        pairs.forEach(([a,b])=>{{ if(visible(lm,a,0.35) && visible(lm,b,0.35)) {{ ctx.beginPath(); ctx.moveTo(lm[a].x*w,lm[a].y*h); ctx.lineTo(lm[b].x*w,lm[b].y*h); ctx.stroke(); }} }});
      }}
      async function tick() {{
        if(!stream || !poseLandmarker || els.video.readyState<2) {{ rafId=requestAnimationFrame(tick); return; }}
        resizeCanvas();
        const res = poseLandmarker.detectForVideo(els.video, performance.now());
        ctx.clearRect(0,0,els.overlay.width, els.overlay.height);
        if(res.landmarks && res.landmarks.length) {{ draw(res.landmarks[0]); renderEval(evaluate(res.landmarks[0])); }}
        rafId=requestAnimationFrame(tick);
      }}
      els.start.onclick = async () => {{
        try {{
          await initPose();
          stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode:'environment', width:{{ideal:1280}}, height:{{ideal:720}} }}, audio:false }});
          els.video.srcObject=stream; await els.video.play(); resizeCanvas(); els.start.disabled=true; els.stop.disabled=false; tick();
        }} catch(e) {{ els.status.textContent='Błąd kamery'; els.messages.innerHTML=''; els.messages.appendChild(msg('Nie udało się uruchomić kamery. Użyj HTTPS i Safari / Chrome mobilnego.')); }}
      }};
      els.stop.onclick = () => {{ if(rafId) cancelAnimationFrame(rafId); if(stream) stream.getTracks().forEach(t=>t.stop()); stream=null; els.start.disabled=false; els.stop.disabled=true; els.capture.disabled=true; els.status.textContent='kamera nieaktywna'; }};
      els.capture.onclick = () => {{ els.messages.appendChild(msg('Capture gotowy — w docelowej wersji zdjęcie zostanie automatycznie zapisane do analizy.')); }};
    </script>
    """
    components.html(html, height=1050, scrolling=True)

