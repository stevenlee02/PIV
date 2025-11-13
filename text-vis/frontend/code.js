import { ForceGraph } from "./forceGraph.js";

const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const infoEl = document.getElementById("info");
const svg = document.getElementById("graph");
const detailsEl = document.getElementById("details");

function resizeSVG() {
  const width = window.innerWidth - 100;
  const height = window.innerHeight - 200;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
}
resizeSVG();
window.addEventListener("resize", resizeSVG);

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return alert("Please select a file first.");

  statusEl.textContent = "Analyzing... (this may take a minute)";
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("http://127.0.0.1:8001/analyze", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    statusEl.textContent = "Done!";
    drawGraph(data);
  } catch (err) {
    console.error("❌ Backend error:", err);
    statusEl.textContent = "Error occurred.";
  }
});

function drawGraph(data) {
  // 清空 SVG
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const width = svg.clientWidth;
  const height = svg.clientHeight;

  // 节点半径、力参数都做轻微调整，避免重叠太严重
  const fg = ForceGraph(
    { 
      nodes: data.nodes.map((d, i) => ({ ...d, id: d.id || `node-${i}` })), 
      links: data.links 
    },
    {
      nodeId: d => d.id,
      nodeTitle: d => `${d.id}\nCount: ${d.value}`,
      nodeRadius: d => 3 + Math.log2(d.value + 1),  // 🔹 节点更小
      linkStrokeWidth: d => 0.3 + Math.sqrt(d.value) * 0.3, // 🔹 线更细
      nodeStrength: -300,   // 🔹 增加斥力
      linkStrength: 0.05,   // 🔹 减弱连线拉力
      width,
      height,
    }
  );

  // 点击节点显示人物详情
  const nodes = fg.querySelectorAll("circle");
  nodes.forEach(n => {
    n.addEventListener("click", () => {
      const d = n.__data__;
      detailsEl.innerHTML = `
        <h3>🧍 ${d.id}</h3>
        <p>Appears <b>${d.value}</b> times in text.</p>
        <p>Click a connection line to see shared scenes.</p>
      `;
    });
  });

  // 点击连线显示上下文片段（与后端 sorted key 对齐）
  const links = fg.querySelectorAll("line");
  links.forEach(line => {
    line.addEventListener("click", () => {
      const d = line.__data__;
      const key = [d.source.id, d.target.id].sort().join("|"); // 与后端保持一致
      const ctx = data.contexts[key];
      if (!ctx || ctx.length === 0) {
        infoEl.innerHTML = `<p><b>${d.source.id}</b> & <b>${d.target.id}</b>: No context found.</p>`;
      } else {
        const snippets = ctx
          .slice(0, 3)
          .map(s => `<blockquote>${s.trim()}...</blockquote>`)
          .join("");
        infoEl.innerHTML = `<h3>📖 ${d.source.id} & ${d.target.id}</h3>${snippets}`;
      }
    });
  });

  svg.appendChild(fg);
}
