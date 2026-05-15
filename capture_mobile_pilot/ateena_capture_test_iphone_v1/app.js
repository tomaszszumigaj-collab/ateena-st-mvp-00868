
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const els = {
  gender: document.getElementById('gender'),
  height: document.getElementById('height'),
  weight: document.getElementById('weight'),
  age: document.getElementById('age'),
  captureRole: document.getElementById('captureRole'),
  clothingFit: document.getElementById('clothingFit'),
  layers: document.getElementById('layers'),
  clothingWarning: document.getElementById('clothingWarning'),
  startCamera: document.getElementById('startCamera'),
  stopCamera: document.getElementById('stopCamera'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  overallStatus: document.getElementById('overallStatus'),
  orientationBadge: document.getElementById('orientationBadge'),
  standingBadge: document.getElementById('standingBadge'),
  scoreBadge: document.getElementById('scoreBadge'),
  checks: document.getElementById('checks'),
  messages: document.getElementById('messages'),
  captureBtn: document.getElementById('captureBtn'),
  forceCaptureBtn: document.getElementById('forceCaptureBtn'),
  guideImage: document.getElementById('guideImage'),
  frontPreview: document.getElementById('frontPreview'),
  profilePreview: document.getElementById('profilePreview'),
  backPreview: document.getElementById('backPreview'),
  frontMeta: document.getElementById('frontMeta'),
  profileMeta: document.getElementById('profileMeta'),
  backMeta: document.getElementById('backMeta'),
  exportJson: document.getElementById('exportJson'),
  sessionSummary: document.getElementById('sessionSummary'),
};

let currentStep = 'front';
let stream = null;
let poseLandmarker = null;
let rafId = null;
let lastEval = null;
let acceptStableFrames = 0;
const ctx = els.overlay.getContext('2d');

const session = {
  createdAt: new Date().toISOString(),
  settings: {},
  captures: { front: null, profile: null, back: null },
  sessionQuality: null
};

const GUIDES = {
  kobieta: {
    front: 'assets/guide_front_female.png',
    profile: 'assets/guide_profile_female.png',
    back: 'assets/guide_back_female.png',
  },
  'mężczyzna': {
    front: 'assets/guide_front_male.png',
    profile: 'assets/guide_profile_male.png',
    back: 'assets/guide_back_male.png',
  }
};

const IDX = {
  nose: 0,
  leftShoulder: 11, rightShoulder: 12,
  leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16,
  leftHip: 23, rightHip: 24,
  leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28,
  leftHeel: 29, rightHeel: 30,
  leftFoot: 31, rightFoot: 32,
};


function sessionQualityScore() {
  const parts = ['front','profile','back'].filter(k => session.captures[k]);
  if (!parts.length) return 0;
  const vals = parts.map(k => Number((((session.captures[k]||{}).meta||{}).evaluation||{}).score || 0));
  return Math.round(vals.reduce((a,b)=>a+b,0) / vals.length);
}
function refreshSessionSummary() {
  const done = ['front','profile','back'].filter(k => !!session.captures[k]);
  const missing = ['front','profile','back'].filter(k => !session.captures[k]);
  const score = sessionQualityScore();
  session.sessionQuality = { done, missing, score };
  if (els.sessionSummary) {
    els.sessionSummary.textContent = done.length ? `Ujęcia gotowe: ${done.join(', ')} | brakujące: ${missing.join(', ') || 'brak'} | średni score sesji: ${score}/100` : 'Brak pełnej sesji.';
  }
}
function nextStepIfNeeded() {
  if (!session.captures.front) currentStep = 'front';
  else if (!session.captures.profile) currentStep = 'profile';
  else if (!session.captures.back) currentStep = 'back';
  document.querySelectorAll('.step-btn').forEach(b => b.classList.toggle('active', b.dataset.step === currentStep));
  setGuide();
}

function setGuide() {
  const gender = els.gender.value;
  els.guideImage.src = GUIDES[gender][currentStep];
}

document.querySelectorAll('.step-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.step-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentStep = btn.dataset.step;
    setGuide();
    updateCaptureButton();
  });
});

function showClothingWarning() {
  const warn = ['loose', 'very_loose'].includes(els.clothingFit.value) || els.layers.value === 'tak';
  els.clothingWarning.style.display = warn ? 'block' : 'none';
}
els.clothingFit.addEventListener('change', showClothingWarning);
els.layers.addEventListener('change', showClothingWarning);
els.gender.addEventListener('change', setGuide);
setGuide();
showClothingWarning();

function visible(lm, idx, threshold = 0.55) {
  return !!lm[idx] && (lm[idx].visibility ?? 1) >= threshold && lm[idx].x >= 0 && lm[idx].x <= 1 && lm[idx].y >= 0 && lm[idx].y <= 1;
}
function point(lm, idx) {
  return lm[idx] ? {x: lm[idx].x, y: lm[idx].y, z: lm[idx].z ?? 0, v: lm[idx].visibility ?? 1} : null;
}
function mid(a, b) {
  return {x: (a.x+b.x)/2, y: (a.y+b.y)/2};
}
function dist(a,b){
  return Math.hypot(a.x-b.x, a.y-b.y);
}
function angleDeg(a, b) {
  return Math.atan2((b.y-a.y), (b.x-a.x)) * 180 / Math.PI;
}
function angleAt(a, b, c) {
  const ab = {x: a.x-b.x, y: a.y-b.y};
  const cb = {x: c.x-b.x, y: c.y-b.y};
  const dot = ab.x*cb.x + ab.y*cb.y;
  const mag = Math.hypot(ab.x,ab.y) * Math.hypot(cb.x,cb.y);
  if (!mag) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot/mag))) * 180 / Math.PI;
}

async function initPose() {
  if (poseLandmarker) return;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    outputSegmentationMasks: false,
  });
}

async function startCamera() {
  await initPose();
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  });
  els.video.srcObject = stream;
  await els.video.play();
  resizeCanvas();
  els.startCamera.disabled = true;
  els.stopCamera.disabled = false;
  els.forceCaptureBtn.disabled = false;
  tick();
}

function stopCamera() {
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  els.video.srcObject = null;
  els.startCamera.disabled = false;
  els.stopCamera.disabled = true;
  els.captureBtn.disabled = true;
  els.forceCaptureBtn.disabled = true;
  setOverallStatus('kamera nieaktywna', 'idle');
  ctx.clearRect(0,0,els.overlay.width, els.overlay.height);
}

function resizeCanvas() {
  const rect = els.video.getBoundingClientRect();
  els.overlay.width = rect.width;
  els.overlay.height = rect.height;
}

window.addEventListener('resize', resizeCanvas);

function drawSkeleton(lm) {
  ctx.clearRect(0,0,els.overlay.width, els.overlay.height);
  const w = els.overlay.width, h = els.overlay.height;
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(20,184,166,0.9)';
  ctx.fillStyle = 'rgba(20,184,166,0.9)';

  const pairs = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[25,27],[27,31],[24,26],[26,28],[28,32]
  ];
  pairs.forEach(([a,b]) => {
    if (visible(lm,a,0.35) && visible(lm,b,0.35)) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x*w, lm[a].y*h);
      ctx.lineTo(lm[b].x*w, lm[b].y*h);
      ctx.stroke();
    }
  });
  Object.values(IDX).forEach(i => {
    if (visible(lm,i,0.35)) {
      ctx.beginPath();
      ctx.arc(lm[i].x*w, lm[i].y*h, 5, 0, Math.PI*2);
      ctx.fill();
    }
  });
}

function evaluatePose(lm) {
  const messages = [];
  const checks = [];

  const reqFront = [IDX.nose, IDX.leftShoulder, IDX.rightShoulder, IDX.leftHip, IDX.rightHip, IDX.leftKnee, IDX.rightKnee, IDX.leftAnkle, IDX.rightAnkle, IDX.leftFoot, IDX.rightFoot];
  const reqProfile = [IDX.nose, IDX.leftShoulder, IDX.rightShoulder, IDX.leftHip, IDX.rightHip, IDX.leftKnee, IDX.rightKnee, IDX.leftAnkle, IDX.rightAnkle, IDX.leftFoot, IDX.rightFoot];

  const nose = point(lm, IDX.nose);
  const ls = point(lm, IDX.leftShoulder), rs = point(lm, IDX.rightShoulder);
  const le = point(lm, IDX.leftElbow), re = point(lm, IDX.rightElbow);
  const lw = point(lm, IDX.leftWrist), rw = point(lm, IDX.rightWrist);
  const lh = point(lm, IDX.leftHip), rh = point(lm, IDX.rightHip);
  const lk = point(lm, IDX.leftKnee), rk = point(lm, IDX.rightKnee);
  const la = point(lm, IDX.leftAnkle), ra = point(lm, IDX.rightAnkle);
  const lf = point(lm, IDX.leftFoot), rf = point(lm, IDX.rightFoot);

  const allPoints = [nose, ls, rs, lw, rw, lh, rh, lk, rk, la, ra, lf, rf].filter(Boolean);
  const minY = Math.min(...allPoints.map(p => p.y));
  const maxY = Math.max(...allPoints.map(p => p.y));
  const bodyHeight = maxY - minY;

  const headVisible = visible(lm, IDX.nose);
  const feetVisible = visible(lm, IDX.leftFoot) && visible(lm, IDX.rightFoot);
  const wristsVisible = visible(lm, IDX.leftWrist, 0.35) && visible(lm, IDX.rightWrist, 0.35);
  const kneesVisible = visible(lm, IDX.leftKnee) && visible(lm, IDX.rightKnee);
  const hipsVisible = visible(lm, IDX.leftHip) && visible(lm, IDX.rightHip);

  const shoulderDist = (ls && rs) ? Math.abs(ls.x-rs.x) : 0;
  const hipDist = (lh && rh) ? Math.abs(lh.x-rh.x) : 0;

  let orientation = 'nieznana';
  if (shoulderDist > 0.16 && hipDist > 0.12) orientation = 'front';
  else if (shoulderDist < 0.10 && hipDist < 0.08) orientation = 'profil';
  else orientation = 'półprofil';

  const shoulderMid = (ls && rs) ? mid(ls, rs) : null;
  const hipMid = (lh && rh) ? mid(lh, rh) : null;
  const ankleMid = (la && ra) ? mid(la, ra) : null;

  let standing = false;
  let sitting = false;
  if (lh && lk && la && rh && rk && ra) {
    const leftKneeAngle = angleAt(lh, lk, la);
    const rightKneeAngle = angleAt(rh, rk, ra);
    standing = leftKneeAngle > 150 && rightKneeAngle > 150 && lh.y < lk.y && lk.y < la.y && rh.y < rk.y && rk.y < ra.y;
    sitting = leftKneeAngle < 135 || rightKneeAngle < 135;
  }

  let centerOk = false;
  if (shoulderMid && hipMid) {
    const cx = (shoulderMid.x + hipMid.x) / 2;
    centerOk = cx > 0.38 && cx < 0.62;
  }

  let cameraRoll = 0;
  if (ls && rs) {
    cameraRoll = Math.abs(angleDeg(ls, rs));
  } else if (lh && rh) {
    cameraRoll = Math.abs(angleDeg(lh, rh));
  }
  if (cameraRoll > 90) cameraRoll = Math.abs(cameraRoll - 180);

  let cameraLevel = 'ok';
  if (shoulderMid && hipMid && nose) {
    const rel = (hipMid.y - nose.y) / Math.max(bodyHeight, 0.001);
    if (rel < 0.32) cameraLevel = 'za wysoko';
    else if (rel > 0.52) cameraLevel = 'za nisko';
  }

  let armsOk = false;
  if (lw && rw && lh && rh) {
    const leftGap = Math.abs(lw.x - lh.x);
    const rightGap = Math.abs(rw.x - rh.x);
    armsOk = leftGap > 0.035 && rightGap > 0.035 && leftGap < 0.20 && rightGap < 0.20;
  }

  const clothingFit = document.getElementById('clothingFit').value;
  const looseClothing = ['loose', 'very_loose'].includes(clothingFit) || document.getElementById('layers').value === 'tak';

  checks.push({label:'głowa w kadrze', ok:headVisible});
  checks.push({label:'stopy w kadrze', ok:feetVisible});
  checks.push({label:'biodra wykryte', ok:hipsVisible});
  checks.push({label:'kolana wykryte', ok:kneesVisible});
  checks.push({label:'nadgarstki / dłonie', ok:wristsVisible});
  checks.push({label:'pełna sylwetka', ok:bodyHeight > 0.68});
  checks.push({label:'użytkownik stoi', ok:standing && !sitting});
  checks.push({label:'ciało wycentrowane', ok:centerOk});
  checks.push({label:'ręce lekko odsunięte', ok:armsOk});
  checks.push({label:'kamera bez przechyłu', ok:cameraRoll <= 7});
  checks.push({label:'ubranie nie jest zbyt luźne', ok:!looseClothing});

  if (currentStep === 'front') {
    checks.push({label:'orientacja = front', ok:orientation === 'front'});
    if (orientation === 'półprofil') messages.push('To nie jest pełny front — obróć ciało bardziej przodem.');
    if (orientation === 'profil') messages.push('To jest profil, a potrzebny jest front.');
  } else if (currentStep === 'profile') {
    checks.push({label:'orientacja = profil', ok:orientation === 'profil'});
    if (orientation === 'półprofil') messages.push('To nadal półprofil — obróć się bardziej bokiem.');
    if (orientation === 'front') messages.push('To jest front, a potrzebny jest profil.');
  } else {
    checks.push({label:'orientacja = tył', ok:orientation === 'front'});
    if (orientation === 'półprofil') messages.push('To nie jest pełny tył — obróć się całkiem plecami do kamery.');
    if (orientation === 'profil') messages.push('To jest profil, a potrzebny jest tył.');
  }

  if (!headVisible) messages.push('Pokaż całą głowę / twarz.');
  if (!feetVisible) messages.push('Pokaż obie stopy w całości.');
  if (!wristsVisible) messages.push('Pokaż obie dłonie lub nadgarstki.');
  if (!(standing && !sitting)) messages.push('Stań prosto — zdjęcie siedząc nie nadaje się do analizy.');
  if (!centerOk) messages.push('Stań bliżej środka kadru.');
  if (!armsOk) messages.push('Odsuń ręce od ciała o około 10–15 cm.');
  if (cameraRoll > 7) messages.push('Wyprostuj telefon — kamera jest przechylona.');
  if (cameraLevel !== 'ok') messages.push(`Ustaw telefon niżej/wyżej — kamera jest ${cameraLevel}.`);
  if (looseClothing) messages.push('Ubranie jest zbyt luźne do dokładnego pomiaru.');

  let score = 0;
  score += headVisible ? 10 : 0;
  score += feetVisible ? 12 : 0;
  score += hipsVisible ? 8 : 0;
  score += kneesVisible ? 8 : 0;
  score += wristsVisible ? 8 : 0;
  score += bodyHeight > 0.68 ? 12 : 0;
  score += (standing && !sitting) ? 12 : 0;
  score += centerOk ? 8 : 0;
  score += armsOk ? 8 : 0;
  score += cameraRoll <= 7 ? 7 : 0;
  score += !looseClothing ? 7 : 0;
  score += ((currentStep === 'front' && orientation === 'front') || (currentStep === 'profile' && orientation === 'profil')) ? 10 : 0;

  const hardReject = !headVisible || !feetVisible || !(standing && !sitting) || !(bodyHeight > 0.68) ||
    ((currentStep === 'front' && orientation !== 'front') || (currentStep === 'profile' && orientation !== 'profil')) ||
    (cameraRoll > 15) || clothingFit === 'very_loose';

  let status = 'REJECT';
  if (!hardReject && score >= 85) status = 'ACCEPT';
  else if (!hardReject && score >= 70) status = 'RETRY';

  return {
    score,
    status,
    orientation,
    standing: standing && !sitting ? 'stoi' : 'niepoprawna / siedzi',
    messages,
    checks,
    cameraRoll: Number(cameraRoll.toFixed(1)),
    cameraLevel,
    hardReject
  };
}

function renderChecks(evalResult) {
  els.checks.innerHTML = '';
  evalResult.checks.forEach(ch => {
    const div = document.createElement('div');
    div.className = 'check ' + (ch.ok ? 'good' : (evalResult.status === 'RETRY' ? 'warn' : 'bad'));
    div.textContent = `${ch.ok ? '✓' : '✕'} ${ch.label}`;
    els.checks.appendChild(div);
  });
}

function renderMessages(evalResult) {
  els.messages.innerHTML = '';
  const msgs = evalResult.messages.slice(0, 6);
  if (!msgs.length) {
    const div = document.createElement('div');
    div.className = 'msg';
    div.textContent = 'Ujęcie wygląda poprawnie.';
    els.messages.appendChild(div);
    return;
  }
  msgs.forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg';
    div.textContent = m;
    els.messages.appendChild(div);
  });
}

function setOverallStatus(text, type) {
  els.overallStatus.textContent = text;
  els.overallStatus.className = 'status';
  if (type === 'good') els.overallStatus.classList.add('status-good');
  else if (type === 'warn') els.overallStatus.classList.add('status-warn');
  else if (type === 'bad') els.overallStatus.classList.add('status-bad');
  else els.overallStatus.classList.add('status-idle');
}

function updateCaptureButton() {
  els.captureBtn.disabled = !(lastEval && lastEval.status === 'ACCEPT' && stream);
  els.forceCaptureBtn.disabled = !stream;
}

function drawGuideFrame() {
  const w = els.overlay.width, h = els.overlay.height;
  ctx.strokeStyle = 'rgba(255,255,255,.65)';
  ctx.lineWidth = 2;
  ctx.setLineDash([8,6]);
  const x = w * 0.18, y = h * 0.05, ww = w * 0.64, hh = h * 0.88;
  ctx.strokeRect(x, y, ww, hh);
  ctx.setLineDash([]);
}

async function tick() {
  if (!stream || !poseLandmarker || els.video.readyState < 2) {
    rafId = requestAnimationFrame(tick);
    return;
  }
  resizeCanvas();
  const results = poseLandmarker.detectForVideo(els.video, performance.now());
  ctx.clearRect(0,0,els.overlay.width, els.overlay.height);
  drawGuideFrame();

  if (results.landmarks && results.landmarks.length) {
    const lm = results.landmarks[0];
    drawSkeleton(lm);
    lastEval = evaluatePose(lm);
    renderChecks(lastEval);
    renderMessages(lastEval);
    els.orientationBadge.textContent = `orientacja: ${lastEval.orientation}`;
    els.standingBadge.textContent = `pozycja: ${lastEval.standing}`;
    els.scoreBadge.textContent = `score: ${Math.round(lastEval.score)}`;
    if (lastEval.status === 'ACCEPT') {
      acceptStableFrames += 1;
      setOverallStatus('ACCEPT — można zrobić zdjęcie', 'good');
      if (acceptStableFrames >= 25 && !session.captures[currentStep]) {
        captureFrame(false);
      }
    } else if (lastEval.status === 'RETRY') {
      acceptStableFrames = 0;
      setOverallStatus('RETRY — popraw ujęcie', 'warn');
    } else {
      acceptStableFrames = 0;
      setOverallStatus('REJECT — zdjęcie nie nadaje się do analizy', 'bad');
    }
  } else {
    lastEval = { status: 'REJECT', score: 0, orientation: '—', standing: '—', messages:['Nie wykryto sylwetki.'], checks:[] };
    renderChecks(lastEval);
    renderMessages(lastEval);
    els.orientationBadge.textContent = 'orientacja: —';
    els.standingBadge.textContent = 'pozycja: —';
    els.scoreBadge.textContent = 'score: 0';
    setOverallStatus('REJECT — brak wykrytej sylwetki', 'bad');
  }

  updateCaptureButton();
  rafId = requestAnimationFrame(tick);
}

function captureFrame(forced = false) {
  if (!stream) return;
  const canvas = document.createElement('canvas');
  canvas.width = els.video.videoWidth;
  canvas.height = els.video.videoHeight;
  const c = canvas.getContext('2d');
  c.drawImage(els.video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

  const meta = {
    step: currentStep,
    accepted: forced ? false : (lastEval?.status === 'ACCEPT'),
    forced,
    evaluation: lastEval,
    settings: collectSettings(),
    capturedAt: new Date().toISOString()
  };

  session.settings = collectSettings();
  session.captures[currentStep] = { image: dataUrl, meta };
  refreshSessionSummary();

  if (currentStep === 'front') {
    els.frontPreview.src = dataUrl;
    els.frontMeta.textContent = JSON.stringify(meta, null, 2);
  } else if (currentStep === 'profile') {
    els.profilePreview.src = dataUrl;
    els.profileMeta.textContent = JSON.stringify(meta, null, 2);
  } else {
    els.backPreview.src = dataUrl;
    els.backMeta.textContent = JSON.stringify(meta, null, 2);
  }
  nextStepIfNeeded();
}

function collectSettings() {
  return {
    gender: els.gender.value,
    height: Number(els.height.value),
    weight: Number(els.weight.value),
    age: Number(els.age.value),
    captureRole: els.captureRole.value,
    clothingFit: els.clothingFit.value,
    layers: els.layers.value,
  };
}

els.startCamera.addEventListener('click', async () => {
  try {
    await startCamera();
  } catch (e) {
    alert('Nie udało się uruchomić kamery. Otwórz stronę przez HTTPS w Safari i daj dostęp do aparatu.\n\n' + e.message);
  }
});
els.stopCamera.addEventListener('click', stopCamera);
els.captureBtn.addEventListener('click', () => captureFrame(false));
els.forceCaptureBtn.addEventListener('click', () => captureFrame(true));
els.exportJson.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ateena_capture_test_session.json';
  a.click();
  URL.revokeObjectURL(url);
});
