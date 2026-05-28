from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .settings import settings
from .iris_runtime import IrisRuntime

_DOCS_HTML = """\
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris App — API</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:2rem}
  h1{color:#7dd3fc;font-size:1.4rem;margin-bottom:.3rem}
  .version{color:#555;font-size:.85rem;margin-bottom:2rem}
  .endpoint{border:1px solid #222;border-radius:6px;padding:1.2rem 1.4rem;margin-bottom:1.4rem}
  .method-get{color:#4ade80}
  .method-post{color:#fb923c}
  .path{color:#e0e0e0;font-size:1.1rem;margin-left:.5rem}
  .desc{color:#a3a3a3;margin:.6rem 0 1rem}
  .label{color:#555;font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem}
  pre{background:#111;border:1px solid #1e1e1e;border-radius:4px;padding:.8rem 1rem;overflow-x:auto;font-size:.85rem;line-height:1.5}
  .comment{color:#5c6370}
  .response-fields{margin-top:.8rem}
  .field{display:flex;gap:.6rem;align-items:baseline;padding:.25rem 0;border-bottom:1px solid #1a1a1a}
  .field:last-child{border:none}
  .fname{color:#7dd3fc;min-width:140px}
  .ftype{color:#c084fc;min-width:70px}
  .fdesc{color:#a3a3a3;font-size:.82rem}
</style>
</head>
<body>

<h1>Iris App</h1>
<div class="version">v0.2.0 &mdash; FastAPI · InsightFace · SQLite</div>

<div class="endpoint">
  <div>
    <span class="method-get">GET</span>
    <span class="path">/health</span>
  </div>
  <p class="desc">Verifica se o serviço está no ar, qual modelo está carregado, o threshold de similaridade configurado e quais providers ONNX estão ativos.</p>
  <div class="label">Exemplo</div>
  <pre><span class="comment"># checar saude do servico</span>
curl http://localhost:8081/health</pre>
  <div class="label" style="margin-top:.8rem">Resposta</div>
  <div class="response-fields">
    <div class="field"><span class="fname">status</span><span class="ftype">string</span><span class="fdesc">"ok" quando o servico esta operacional</span></div>
    <div class="field"><span class="fname">model</span><span class="ftype">string</span><span class="fdesc">nome do modelo InsightFace carregado no runtime atual</span></div>
    <div class="field"><span class="fname">threshold</span><span class="ftype">float</span><span class="fdesc">limite minimo de similaridade para aceitar um match</span></div>
    <div class="field"><span class="fname">providers</span><span class="ftype">list</span><span class="fdesc">providers ONNX em uso (ex: ["CPUExecutionProvider"])</span></div>
    <div class="field"><span class="fname">mqtt_topic</span><span class="ftype">string</span><span class="fdesc">topico MQTT que o worker esta escutando</span></div>
  </div>
</div>

<div class="endpoint">
  <div>
    <span class="method-get">GET</span>
    <span class="path">/subjects</span>
  </div>
  <p class="desc">Lista todos os sujeitos cadastrados no banco de embeddings, com quantas amostras cada um tem e quando foi o último cadastro.</p>
  <div class="label">Exemplo</div>
  <pre><span class="comment"># listar rostos cadastrados</span>
curl http://localhost:8081/subjects</pre>
  <div class="label" style="margin-top:.8rem">Resposta</div>
  <div class="response-fields">
    <div class="field"><span class="fname">subjects</span><span class="ftype">list</span><span class="fdesc">array de objetos com subject, samples e last_sample_at</span></div>
  </div>
</div>

<div class="endpoint">
  <div>
    <span class="method-post">POST</span>
    <span class="path">/enroll</span>
  </div>
  <p class="desc">Cadastra um rosto. Envia um nome (subject) e uma imagem contendo exatamente um rosto visível. O maior rosto detectado é extraído e salvo como embedding ArcFace no SQLite. Quanto mais amostras por sujeito, melhor a assertividade.</p>
  <div class="label">Campos (multipart/form-data)</div>
  <div class="response-fields" style="margin-bottom:.8rem">
    <div class="field"><span class="fname">subject</span><span class="ftype">string</span><span class="fdesc">nome do sujeito a cadastrar (obrigatorio)</span></div>
    <div class="field"><span class="fname">file</span><span class="ftype">image</span><span class="fdesc">imagem JPG/PNG com o rosto (obrigatorio)</span></div>
  </div>
  <div class="label">Exemplo</div>
  <pre><span class="comment"># cadastrar rosto</span>
curl -F "subject=paulo" \\
     -F "file=@/caminho/foto.jpg" \\
     http://localhost:8081/enroll

<span class="comment"># cadastrar multiplas amostras do mesmo sujeito (aumenta assertividade)</span>
for f in fotos/paulo_*.jpg; do
  curl -F "subject=paulo" -F "file=@$f" http://localhost:8081/enroll
done</pre>
  <div class="label" style="margin-top:.8rem">Resposta</div>
  <div class="response-fields">
    <div class="field"><span class="fname">status</span><span class="ftype">string</span><span class="fdesc">"enrolled"</span></div>
    <div class="field"><span class="fname">subject</span><span class="ftype">string</span><span class="fdesc">nome conforme enviado</span></div>
    <div class="field"><span class="fname">embedding_id</span><span class="ftype">int</span><span class="fdesc">id da linha inserida no SQLite</span></div>
    <div class="field"><span class="fname">face.bbox</span><span class="ftype">list</span><span class="fdesc">bounding box do rosto detectado [x1,y1,x2,y2]</span></div>
    <div class="field"><span class="fname">face.det_score</span><span class="ftype">float</span><span class="fdesc">confianca da deteccao do rosto (0.0 a 1.0)</span></div>
  </div>
</div>

<div class="endpoint">
  <div>
    <span class="method-post">POST</span>
    <span class="path">/recognize</span>
  </div>
  <p class="desc">Reconhece o maior rosto de uma imagem comparando seu embedding contra todos os cadastrados via similaridade cosseno. Retorna o melhor candidato e a lista completa de scores para análise de assertividade.</p>
  <div class="label">Campos (multipart/form-data)</div>
  <div class="response-fields" style="margin-bottom:.8rem">
    <div class="field"><span class="fname">file</span><span class="ftype">image</span><span class="fdesc">imagem JPG/PNG com o rosto a reconhecer (obrigatorio)</span></div>
  </div>
  <div class="label">Exemplo</div>
  <pre><span class="comment"># reconhecer e ver scores de todos os candidatos</span>
curl -F "file=@/caminho/foto_teste.jpg" \\
     http://localhost:8081/recognize

<span class="comment"># usando jq para ver so o melhor resultado</span>
curl -s -F "file=@foto.jpg" http://localhost:8081/recognize \\
  | python3 -m json.tool</pre>
  <div class="label" style="margin-top:.8rem">Resposta</div>
  <div class="response-fields">
    <div class="field"><span class="fname">status</span><span class="ftype">string</span><span class="fdesc">"matched" se similarity &ge; threshold, "no_match" caso contrario</span></div>
    <div class="field"><span class="fname">subject</span><span class="ftype">string</span><span class="fdesc">nome do sujeito reconhecido, ou null</span></div>
    <div class="field"><span class="fname">similarity</span><span class="ftype">float</span><span class="fdesc">score cosseno do melhor candidato (0.0 a 1.0). Threshold padrao: 0.55</span></div>
    <div class="field"><span class="fname">face.bbox</span><span class="ftype">list</span><span class="fdesc">bounding box do rosto detectado na imagem de entrada</span></div>
    <div class="field"><span class="fname">face.det_score</span><span class="ftype">float</span><span class="fdesc">confianca da deteccao (quanto mais proximo de 1.0, melhor o rosto)</span></div>
    <div class="field"><span class="fname">candidates</span><span class="ftype">list</span><span class="fdesc">todos os sujeitos com seus scores, ordenados do mais similar ao menos. Use para debugar assertividade</span></div>
  </div>
</div>

<div class="endpoint" style="border-color:#2a2a00">
  <div>
    <span style="color:#facc15">⚠</span>
    <span class="path" style="color:#facc15;font-size:1rem">Assertividade baixa? Cadastre com fotos da própria câmera</span>
  </div>
  <p class="desc" style="margin-top:.6rem">Fotos de perfil com iluminacao, angulo e distancia diferentes da camera real tendem a gerar embeddings fracos para reconhecimento no stream. O threshold padrao e <strong style="color:#e0e0e0">0.55</strong>. Para bons resultados, cadastre usando capturas geradas pelo proprio worker do Iris App.</p>
  <div class="label" style="margin-top:.8rem">Passo a passo — cadastrar pela ultima captura do worker</div>
  <pre><span class="comment"># 1. baixar a ultima captura processada</span>
curl -s http://localhost:8081/captures/latest/image -o /tmp/iris_latest.jpg

<span class="comment"># 2. conferir como ficou a imagem</span>
xdg-open /tmp/iris_latest.jpg

<span class="comment"># 3. cadastrar com essa foto</span>
curl -F "subject=paulo" -F "file=@/tmp/iris_latest.jpg" http://localhost:8081/enroll</pre>
  <div class="label" style="margin-top:.8rem">Cadastrar múltiplas amostras (recomendado)</div>
  <pre><span class="comment"># repetir com 3-5 capturas diferentes para melhorar assertividade</span>
for ID in "capture-id-1" "capture-id-2" "capture-id-3"; do
  curl -s "http://localhost:8081/captures/${ID}/image" -o "/tmp/${ID}.jpg"
  curl -F "subject=paulo" -F "file=@/tmp/${ID}.jpg" http://localhost:8081/enroll
done</pre>
  <div class="label" style="margin-top:.8rem">Listar IDs de capturas recentes</div>
  <pre><span class="comment"># ver capturas recentes e escolher IDs para re-cadastro</span>
curl -s "http://localhost:8081/captures?limit=10" | python3 -m json.tool</pre>
  <div class="label" style="margin-top:.8rem">Monitorar reconhecimentos em tempo real</div>
  <pre><span class="comment"># acompanhar log ao vivo — aparece cada vez que o worker processa uma captura valida</span>
tail -f /data/events/recognitions.jsonl

<span class="comment"># ou via MQTT</span>
mosquitto_sub -h localhost -p 1883 -t "facial/recognitions"</pre>
</div>

<div style="margin-top:2rem;color:#333;font-size:.78rem">
  Banco de embeddings: /data/faces/faces.sqlite3 &mdash;
  Log de eventos: /data/events/recognitions.jsonl &mdash;
  <a href="/docs" style="color:#555">Swagger /docs</a>
</div>

</body>
</html>
"""

_DASHBOARD_HTML = """\
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris App</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#101214;color:#e7e9ea;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid #252a2f;background:#15181b}
  h1{font-size:17px;font-weight:650;margin:0}
  .pill{font-size:12px;color:#9aa3ad;border:1px solid #313840;border-radius:999px;padding:5px 9px}
  main{display:grid;grid-template-columns:minmax(360px,1.2fr) minmax(360px,.8fr);gap:16px;padding:16px;max-width:1440px;margin:0 auto}
  section{min-width:0}
  .panel{background:#171a1e;border:1px solid #292f35;border-radius:8px;overflow:hidden}
  .panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#a8b1ba;margin:0;padding:12px 14px;border-bottom:1px solid #292f35}
  .image-wrap{position:relative;background:#08090a;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
  .image-wrap img{width:100%;height:100%;object-fit:contain;display:block}
  .empty{color:#66707a;font-size:14px}
  .status-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:12px}
  .controls{display:grid;grid-template-columns:1fr 1fr auto;gap:8px;padding:12px;border-top:1px solid #292f35}
  input,select,button{height:36px;border-radius:6px;border:1px solid #303842;background:#101317;color:#e7e9ea;padding:0 10px;font:inherit;font-size:13px}
  button{cursor:pointer;background:#223145;border-color:#385272}
  button.secondary{background:#1b2025;border-color:#303842}
  button.danger{background:#3a1f23;border-color:#66313a}
  button:hover{filter:brightness(1.12)}
  .metric{background:#111417;border:1px solid #262b31;border-radius:6px;padding:10px}
  .metric .label{font-size:11px;color:#7f8994;margin-bottom:6px}
  .metric .value{font-size:18px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .result{padding:14px;display:grid;gap:10px}
  .row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #252a2f;padding-bottom:8px}
  .row:last-child{border-bottom:0;padding-bottom:0}
  .key{color:#8e98a3;font-size:13px}
  .val{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;text-align:right;overflow-wrap:anywhere}
  .events{max-height:520px;overflow:auto}
  .subjects{max-height:260px;overflow:auto;padding:10px 12px}
  .subject-item{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;border-bottom:1px solid #252a2f;padding:8px 0}
  .subject-item:last-child{border-bottom:0}
  .subject-meta{color:#8e98a3;font-size:12px;margin-top:3px}
  .samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;padding:12px}
  .sample{border:1px solid #292f35;border-radius:6px;background:#111417;overflow:hidden}
  .sample img{width:100%;aspect-ratio:4/3;object-fit:cover;background:#08090a;display:block}
  .sample-body{padding:8px;display:grid;gap:6px}
  .sample-id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#9aa3ad;overflow-wrap:anywhere}
  .upload-grid{display:grid;grid-template-columns:minmax(320px,1fr) minmax(320px,1fr);gap:16px;padding:16px}
  .upload-stack{display:grid;gap:12px}
  .upload-preview{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .preview-box{border:1px solid #292f35;border-radius:6px;background:#111417;overflow:hidden}
  .preview-box img,.preview-box canvas{width:100%;aspect-ratio:4/3;object-fit:contain;background:#08090a;display:block}
  .preview-box .empty{aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}
  .panel-body{padding:12px}
  .muted{color:#8e98a3;font-size:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #252a2f;vertical-align:top}
  th{position:sticky;top:0;background:#171a1e;color:#98a3ad;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  tr:hover td{background:#1b1f24}
  .matched{color:#59d185;font-weight:650}
  .no-match{color:#f2b15d;font-weight:650}
  .error{color:#ff7777}
  @media (max-width:900px){main{grid-template-columns:1fr}.status-grid{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Iris App</h1>
  <div class="pill" id="runtime">carregando</div>
</header>
<main>
  <section class="panel">
    <h2>Captura analisada</h2>
    <div class="image-wrap">
      <img id="latestImage" alt="" hidden>
      <div id="emptyImage" class="empty">aguardando primeira captura</div>
    </div>
    <div class="status-grid">
      <div class="metric"><div class="label">Capturas</div><div class="value" id="capturesSeen">0</div></div>
      <div class="metric"><div class="label">Eventos</div><div class="value" id="eventsWritten">0</div></div>
      <div class="metric"><div class="label">Frames</div><div class="value" id="framesRead">0</div></div>
      <div class="metric"><div class="label">Erros</div><div class="value" id="errors">0</div></div>
    </div>
    <div class="controls">
      <select id="subjectSelect"></select>
      <input id="subjectInput" placeholder="novo sujeito ou edite o nome selecionado">
      <button id="enrollLatest" type="button">Cadastrar captura atual</button>
    </div>
  </section>
  <section class="panel">
    <h2>Resultado atual</h2>
    <div class="result">
      <div class="row"><span class="key">status</span><span class="val" id="status">-</span></div>
      <div class="row"><span class="key">subject</span><span class="val" id="subject">-</span></div>
      <div class="row"><span class="key">similarity</span><span class="val" id="similarity">-</span></div>
      <div class="row"><span class="key">face score</span><span class="val" id="faceScore">-</span></div>
      <div class="row"><span class="key">bbox</span><span class="val" id="bbox">-</span></div>
      <div class="row"><span class="key">captured_at</span><span class="val" id="capturedAt">-</span></div>
      <div class="row"><span class="key">last error</span><span class="val error" id="lastError">-</span></div>
    </div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>Upload Manual</h2>
    <div class="upload-grid">
      <div class="upload-stack">
        <div class="panel-body" style="padding-bottom:0">
          <div class="muted">Cadastro manual pelo navegador</div>
        </div>
        <div class="controls" style="border-top:0">
          <input id="uploadEnrollSubject" placeholder="sujeito para cadastrar">
          <input id="uploadEnrollFile" type="file" accept="image/*">
          <button id="uploadEnrollButton" type="button">Enviar cadastro</button>
        </div>
        <div class="upload-preview">
          <div class="preview-box">
            <img id="uploadEnrollPreview" alt="" hidden>
            <div id="uploadEnrollPreviewEmpty" class="empty">sem imagem</div>
          </div>
          <div class="preview-box">
            <canvas id="uploadEnrollCrop" hidden></canvas>
            <div id="uploadEnrollCropEmpty" class="empty">sem crop</div>
          </div>
        </div>
        <div class="result panel-body" id="uploadEnrollResult">
          <div class="row"><span class="key">status</span><span class="val">-</span></div>
        </div>
      </div>
      <div class="upload-stack">
        <div class="panel-body" style="padding-bottom:0">
          <div class="muted">Comparacao manual pelo navegador</div>
        </div>
        <div class="controls" style="border-top:0">
          <input id="uploadCompareHint" value="compare" readonly>
          <input id="uploadCompareFile" type="file" accept="image/*">
          <button id="uploadCompareButton" type="button">Enviar comparacao</button>
        </div>
        <div class="upload-preview">
          <div class="preview-box">
            <img id="uploadComparePreview" alt="" hidden>
            <div id="uploadComparePreviewEmpty" class="empty">sem imagem</div>
          </div>
          <div class="preview-box">
            <canvas id="uploadCompareCrop" hidden></canvas>
            <div id="uploadCompareCropEmpty" class="empty">sem crop</div>
          </div>
        </div>
        <div class="result panel-body" id="uploadCompareResult">
          <div class="row"><span class="key">status</span><span class="val">-</span></div>
        </div>
      </div>
    </div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>Rostos cadastrados</h2>
    <div class="subjects" id="subjectsList"></div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>Amostras do sujeito selecionado</h2>
    <div class="samples" id="samplesList"><div class="empty">selecione um sujeito</div></div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>Log de comparacoes</h2>
    <div class="events">
      <table>
        <thead><tr><th>#</th><th>capture_id</th><th>timestamp</th><th>status</th><th>subject</th><th>similarity</th><th>det_score</th><th>candidates</th></tr></thead>
        <tbody id="eventRows"></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const fmt = (value) => value === null || value === undefined ? "-" : value;
const fixed = (value) => typeof value === "number" ? value.toFixed(4) : "-";
let latestCaptureId = null;
let knownSubjects = [];
let activeSubject = "";
let streamRunning = false;
let streamPreviewReady = false;
let latestFallbackImageUrl = null;
let enrollUploadUrl = null;
let compareUploadUrl = null;

function fileToObjectUrl(file) {
  return file ? URL.createObjectURL(file) : null;
}

function replaceObjectUrl(currentUrl, file) {
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  return fileToObjectUrl(file);
}

function setPreviewImage(imageId, emptyId, url) {
  const image = document.getElementById(imageId);
  const empty = document.getElementById(emptyId);
  if (!url) {
    image.hidden = true;
    empty.hidden = false;
    image.removeAttribute("src");
    return;
  }
  image.src = url;
  image.hidden = false;
  empty.hidden = true;
}

function setResultBox(targetId, rows) {
  const target = document.getElementById(targetId);
  target.innerHTML = rows.map(([key, value, cls]) => `
    <div class="row"><span class="key">${key}</span><span class="val ${cls || ""}">${value}</span></div>
  `).join("");
}

function drawDetectedCrop(previewId, canvasId, emptyId, bbox) {
  const image = document.getElementById(previewId);
  const canvas = document.getElementById(canvasId);
  const empty = document.getElementById(emptyId);
  if (!bbox || image.naturalWidth === 0 || image.naturalHeight === 0) {
    canvas.hidden = true;
    empty.hidden = false;
    return;
  }
  const [x1, y1, x2, y2] = bbox.map(value => Number(value));
  const width = Math.max(1, Math.round(x2 - x1));
  const height = Math.max(1, Math.round(y2 - y1));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, x1, y1, width, height, 0, 0, width, height);
  canvas.hidden = false;
  empty.hidden = true;
}

async function uploadEnroll() {
  const subject = document.getElementById("uploadEnrollSubject").value.trim();
  const fileInput = document.getElementById("uploadEnrollFile");
  const file = fileInput.files[0];
  if (!subject || !file) return;
  const form = new FormData();
  form.append("subject", subject);
  form.append("file", file);
  const res = await fetch("/enroll", {method: "POST", body: form});
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.detail || "falha no cadastro");
  const face = payload.face || {};
  setResultBox("uploadEnrollResult", [
    ["status", payload.status || "-", "matched"],
    ["subject", payload.subject || "-", ""],
    ["embedding_id", fmt(payload.embedding_id), ""],
    ["face score", fixed(face.det_score), ""],
    ["bbox", face.bbox ? face.bbox.map(v => Number(v).toFixed(1)).join(", ") : "-", ""],
  ]);
  drawDetectedCrop("uploadEnrollPreview", "uploadEnrollCrop", "uploadEnrollCropEmpty", face.bbox);
  activeSubject = subject;
  document.getElementById("subjectInput").value = subject;
  await refresh();
  await refreshSamples();
}

async function uploadCompare() {
  const fileInput = document.getElementById("uploadCompareFile");
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/compare", {method: "POST", body: form});
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.detail || "falha na comparacao");
  const detector = payload.detector || {};
  const recognition = payload.recognition || {};
  const quality = payload.quality || {};
  const statusClass = recognition.status === "matched" ? "matched" : "no-match";
  setResultBox("uploadCompareResult", [
    ["status", recognition.status || "-", statusClass],
    ["subject", recognition.subject || "-", ""],
    ["similarity", fixed(recognition.similarity), ""],
    ["detector score", fixed(detector.score), ""],
    ["quality", quality.accepted ? "accepted" : (quality.reason || "rejected"), quality.accepted ? "matched" : "no-match"],
    ["bbox", detector.bbox ? detector.bbox.map(v => Number(v).toFixed(1)).join(", ") : "-", ""],
  ]);
  drawDetectedCrop("uploadComparePreview", "uploadCompareCrop", "uploadCompareCropEmpty", detector.bbox);
}

function selectedSubject() {
  const typed = document.getElementById("subjectInput").value.trim();
  const selected = document.getElementById("subjectSelect").value.trim();
  return typed || selected;
}

async function enrollLatest() {
  const subject = selectedSubject();
  if (!latestCaptureId || !subject) return;
  await fetch(`/captures/${encodeURIComponent(latestCaptureId)}/enroll?subject=${encodeURIComponent(subject)}`, {method: "POST"});
  activeSubject = subject;
  await refresh();
  await refreshSamples();
}

async function deleteSubject(subject) {
  if (!subject) return;
  await fetch(`/subjects/${encodeURIComponent(subject)}`, {method: "DELETE"});
  if (activeSubject === subject) activeSubject = "";
  await refresh();
  await refreshSamples();
}

async function deleteSample(sampleId) {
  if (!sampleId) return;
  await fetch(`/samples/${encodeURIComponent(sampleId)}`, {method: "DELETE"});
  await refresh();
  await refreshSamples();
}

async function refreshSamples() {
  const subject = selectedSubject();
  activeSubject = subject;
  const target = document.getElementById("samplesList");
  if (!subject) {
    target.innerHTML = `<div class="empty">selecione um sujeito</div>`;
    return;
  }
  const res = await fetch(`/subjects/${encodeURIComponent(subject)}/samples`, {cache: "no-store"});
  const payload = await res.json();
  const samples = payload.samples || [];
  target.innerHTML = samples.length ? samples.map(sample => `
    <div class="sample">
      ${sample.image_url ? `<img src="${sample.image_url}?t=${Date.now()}" alt="">` : `<div class="empty" style="aspect-ratio:4/3;display:flex;align-items:center;justify-content:center">sem imagem</div>`}
      <div class="sample-body">
        <div class="sample-id">#${sample.id} | ${sample.source || "-"}</div>
        <div class="subject-meta">${sample.created_at || "-"}</div>
        <button class="danger" type="button" onclick="deleteSample(${sample.id})">remover amostra</button>
      </div>
    </div>
  `).join("") : `<div class="empty">nenhuma amostra para ${subject}</div>`;
}

function renderSubjects(subjects) {
  const select = document.getElementById("subjectSelect");
  const previous = select.value;
  select.innerHTML = `<option value="">selecionar cadastrado</option>` + subjects.map(
    item => `<option value="${item.subject}">${item.subject}</option>`
  ).join("");
  if (subjects.some(item => item.subject === previous)) select.value = previous;

  document.getElementById("subjectsList").innerHTML = subjects.length ? subjects.map(item => `
    <div class="subject-item">
      <div>
        <div>${item.subject}</div>
        <div class="subject-meta">${item.samples} amostra(s) | ultimo cadastro: ${item.last_sample_at || "-"}</div>
      </div>
      <button class="secondary" type="button" onclick="document.getElementById('subjectInput').value='${item.subject}'; refreshSamples()">usar</button>
      <button class="danger" type="button" onclick="deleteSubject('${item.subject}')">remover</button>
    </div>
  `).join("") : `<div class="empty">nenhum rosto cadastrado</div>`;
}

function refreshPreviewImage() {
  const image = document.getElementById("latestImage");
  const empty = document.getElementById("emptyImage");
  if (streamRunning && streamPreviewReady) {
    image.src = `/preview/latest.jpg?t=${Date.now()}`;
    image.hidden = false;
    empty.hidden = true;
    return;
  }
  if (latestFallbackImageUrl) {
    image.src = `${latestFallbackImageUrl}?t=${Date.now()}`;
    image.hidden = false;
    empty.hidden = true;
    return;
  }
  image.hidden = true;
  empty.hidden = false;
}

async function refresh() {
  const [healthRes, capturesRes, subjectsRes] = await Promise.all([
    fetch("/health", {cache: "no-store"}),
    fetch("/captures?limit=25", {cache: "no-store"}),
    fetch("/subjects", {cache: "no-store"}),
  ]);
  const health = await healthRes.json();
  const captures = await capturesRes.json();
  const subjectsPayload = await subjectsRes.json();
  const stream = health.stream || {};
  const events = captures.events || [];
  const latest = events[0];
  streamRunning = Boolean(stream.running);
  streamPreviewReady = Boolean(stream.preview_at);
  latestFallbackImageUrl = latest && latest.image_url ? latest.image_url : null;
  knownSubjects = subjectsPayload.subjects || [];
  renderSubjects(knownSubjects);

  document.getElementById("runtime").textContent =
    `${health.model} | ${health.providers.join(",")} | ${stream.running ? "rodando" : "parado"}`;
  document.getElementById("capturesSeen").textContent = fmt(stream.captures_seen);
  document.getElementById("eventsWritten").textContent = fmt(stream.events_written);
  document.getElementById("framesRead").textContent = fmt(stream.frames_read);
  document.getElementById("errors").textContent = fmt(stream.recognition_errors);
  document.getElementById("lastError").textContent = stream.last_error || "-";

  refreshPreviewImage();

  if (latest) {
    latestCaptureId = latest.capture_id;
    const rec = latest.recognition || {};
    const face = rec.face || {};
    document.getElementById("status").textContent = rec.status || "-";
    document.getElementById("status").className = rec.status === "matched" ? "val matched" : "val no-match";
    document.getElementById("subject").textContent = rec.subject || "-";
    document.getElementById("similarity").textContent = fixed(rec.similarity);
    document.getElementById("faceScore").textContent = fixed(face.det_score);
    document.getElementById("bbox").textContent = face.bbox ? face.bbox.map(v => Number(v).toFixed(1)).join(", ") : "-";
    document.getElementById("capturedAt").textContent = latest.captured_at || "-";
  }

  const rows = events.map((event) => {
    const rec = event.recognition || {};
    const face = rec.face || {};
    const candidates = (rec.candidates || []).map(c => `${c.subject}:${fixed(c.similarity)}`).join(" | ");
    const statusClass = rec.status === "matched" ? "matched" : "no-match";
    return `<tr>
      <td>${event.capture_number}</td>
      <td>${event.capture_id}</td>
      <td>${event.captured_at}</td>
      <td class="${statusClass}">${rec.status || "-"}</td>
      <td>${rec.subject || "-"}</td>
      <td>${fixed(rec.similarity)}</td>
      <td>${fixed(face.det_score)}</td>
      <td>${candidates || "-"}</td>
    </tr>`;
  }).join("");
  document.getElementById("eventRows").innerHTML = rows;
}

document.getElementById("enrollLatest").addEventListener("click", () => enrollLatest().catch(console.error));
document.getElementById("uploadEnrollButton").addEventListener("click", () => uploadEnroll().catch(console.error));
document.getElementById("uploadCompareButton").addEventListener("click", () => uploadCompare().catch(console.error));
document.getElementById("uploadEnrollFile").addEventListener("change", (event) => {
  enrollUploadUrl = replaceObjectUrl(enrollUploadUrl, event.target.files[0]);
  setPreviewImage("uploadEnrollPreview", "uploadEnrollPreviewEmpty", enrollUploadUrl);
  drawDetectedCrop("uploadEnrollPreview", "uploadEnrollCrop", "uploadEnrollCropEmpty", null);
});
document.getElementById("uploadCompareFile").addEventListener("change", (event) => {
  compareUploadUrl = replaceObjectUrl(compareUploadUrl, event.target.files[0]);
  setPreviewImage("uploadComparePreview", "uploadComparePreviewEmpty", compareUploadUrl);
  drawDetectedCrop("uploadComparePreview", "uploadCompareCrop", "uploadCompareCropEmpty", null);
});
document.getElementById("subjectSelect").addEventListener("change", (event) => {
  if (event.target.value) document.getElementById("subjectInput").value = event.target.value;
  refreshSamples().catch(console.error);
});
document.getElementById("subjectInput").addEventListener("change", () => refreshSamples().catch(console.error));
refresh().catch(console.error);
refreshSamples().catch(console.error);
setInterval(() => refreshPreviewImage(), 1000);
setInterval(() => refresh().catch(console.error), 3000);
</script>
</body>
</html>
"""

runtime = IrisRuntime(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="Iris App", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def docs_page() -> HTMLResponse:
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/api-help", response_class=HTMLResponse, include_in_schema=False)
def api_help_page() -> HTMLResponse:
    return HTMLResponse(content=_DOCS_HTML)


@app.get("/health")
def health() -> dict:
    return runtime.health()


@app.get("/debug/engine")
def debug_engine() -> dict:
    return runtime.engine_status()


@app.get("/debug/pipeline")
def debug_pipeline() -> dict:
    return runtime.debug_pipeline()


@app.get("/cameras")
def cameras() -> dict:
    return runtime.cameras_status()


@app.get("/cameras/{camera_id}/status")
def camera_status(camera_id: str) -> dict:
    try:
        return runtime.camera_status(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="camera nao encontrada") from exc


@app.post("/cameras/{camera_id}/start")
def camera_start(camera_id: str) -> dict:
    try:
        return runtime.start_camera(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/cameras/{camera_id}/stop")
def camera_stop(camera_id: str) -> dict:
    try:
        return runtime.stop_camera(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/subjects")
def subjects() -> dict:
    return runtime.subjects()


@app.delete("/subjects/{subject}")
def delete_subject(subject: str) -> dict:
    return runtime.delete_subject(subject)


@app.get("/subjects/{subject}/samples")
def subject_samples(subject: str) -> dict:
    return runtime.subject_samples(subject)


@app.delete("/samples/{sample_id}")
def delete_sample(sample_id: int) -> dict:
    return runtime.delete_sample(sample_id)


@app.get("/stream/status")
def stream_status() -> dict:
    return runtime.stream_status()


@app.post("/stream/start")
def stream_start() -> dict:
    return runtime.start_stream()


@app.post("/stream/stop")
def stream_stop() -> dict:
    return runtime.stop_stream()


@app.get("/captures")
def captures(limit: int = 20) -> dict:
    return runtime.recent_events(limit=limit)


@app.get("/captures/latest")
def latest_capture() -> dict:
    return runtime.latest_event()


@app.get("/captures/{capture_id}/image")
def capture_image(capture_id: str) -> FileResponse:
    try:
        return FileResponse(runtime.capture_image_path(capture_id), media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="captura nao encontrada") from exc


@app.get("/events")
def events(limit: int = 20, camera_id: str | None = None) -> dict:
    return runtime.recent_events(limit=limit, camera_id=camera_id)


@app.get("/events/{event_id}")
def event_by_id(event_id: str) -> dict:
    try:
        return runtime.event_by_id(event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="evento nao encontrado") from exc


@app.get("/events/{event_id}/frame.jpg")
def event_frame_image(event_id: str) -> FileResponse:
    try:
        return FileResponse(runtime.frame_image_path(event_id), media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="frame do evento nao encontrado") from exc


@app.get("/events/{event_id}/face.jpg")
def event_face_image(event_id: str) -> FileResponse:
    try:
        return FileResponse(runtime.face_image_path(event_id), media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="crop facial nao encontrado") from exc


@app.get("/preview/latest.jpg")
def latest_preview_image() -> Response:
    payload = runtime.latest_preview_jpeg()
    if not payload:
        raise HTTPException(status_code=404, detail="preview indisponivel")
    return Response(content=payload, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/captures/{capture_id}/enroll")
def enroll_capture(capture_id: str, subject: str) -> dict:
    try:
        return runtime.enroll_capture(subject=subject, capture_id=capture_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="captura nao encontrada") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/events/{event_id}/enroll")
def enroll_event(event_id: str, subject: str) -> dict:
    try:
        return runtime.enroll_event(subject=subject, event_id=event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="crop facial nao encontrado") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/enroll")
async def enroll(subject: str = Form(...), file: UploadFile = File(...)) -> dict:
    try:
        return runtime.enroll(subject=subject, filename=file.filename, payload=await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)) -> dict:
    try:
        return runtime.recognize_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/compare")
async def compare(file: UploadFile = File(...)) -> dict:
    try:
        return runtime.compare_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
