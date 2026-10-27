// The start screen, and the status view once a replay runs.

import init, { WebApp } from "./pkg/pelorus_web.js";
import { HEIGHT_ABOVE_M, HEIGHT_BELOW_M, PointCloudView, turboGradient } from "./viewer.js";
import { ratslangEditor } from "./rl-editor.js";

const SPIRES = { cloud: "/hesai/pandar", imu: "/alphasense_driver_ros/imu", config: "configs/oxford-spires.rl" };
const GRAND_TOUR = { cloud: "/boxi/hesai/points", imu: "/boxi/stim320/imu", config: "configs/grandtour.rl" };
const HELIPR = { cloud: "/ouster/points", imu: "/imu/data_raw", config: "configs/helipr-ouster.rl" };

/** The demo recordings, grouped by dataset. */
const DEMOS = [
  {
    dataset: "Oxford Spires, carried by hand",
    demos: [
      { title: "Keble College", detail: "30 s, 60 MB", url: "demo/keble_college_02_30s.mcap", ...SPIRES },
    ],
  },
  {
    dataset: "GrandTour, ANYmal legged robot",
    demos: [
      { title: "sbb1", detail: "30 s, 88 MB", url: "demo/grand_tour_sbb1_30s.mcap", ...GRAND_TOUR },
      { title: "eth1", detail: "30 s, 90 MB", url: "demo/grand_tour_eth1_30s.mcap", ...GRAND_TOUR },
      { title: "snow1", detail: "30 s, 81 MB", url: "demo/grand_tour_snow1_30s.mcap", ...GRAND_TOUR },
    ],
  },
  {
    dataset: "HeLiPR, car with an Ouster OS2-128",
    demos: [
      { title: "town02", detail: "20 s, 76 MB", url: "demo/helipr_town02_20s.mcap", ...HELIPR },
      { title: "bridge02", detail: "20 s, 63 MB", url: "demo/helipr_bridge02_20s.mcap", ...HELIPR },
    ],
  },
];

const PRESETS = [
  ["Default", ""],
  ["Oxford Spires (Hesai)", "configs/oxford-spires.rl"],
  ["GrandTour (Hesai, STIM320)", "configs/grandtour.rl"],
  ["HeLiPR (Ouster OS2-128)", "configs/helipr-ouster.rl"],
  ["Newer College (Ouster)", "configs/newer-college.rl"],
];

const $ = (id) => document.getElementById(id);
// Progress and errors go under the section the user acted in: the demo list or
// their own file.
let statusId = "own-status";
const status = (text) => {
  for (const id of ["demo-status", "own-status"]) $(id).textContent = id === statusId ? text : "";
};

// The hero cycles through the demo renders, unless motion is reduced.
{
  const renders = [...document.querySelectorAll(".hero-renders figure")];
  let shown = 0;
  if (renders.length > 1 && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    setInterval(() => {
      if (document.hidden || $("start").hidden) return;
      renders[shown].classList.remove("shown");
      shown = (shown + 1) % renders.length;
      renders[shown].classList.add("shown");
    }, 7000);
  }
}

await init();
let worker = null;
let app = null;
let cloudView = null;
let bagName = "";
let pendingPlay = null;
// The bag and settings of the running replay, for a restart with Space.
let currentFile = null;
let currentPlay = null;
// Log lines that arrive before the view exists.
const earlyLogs = [];

function log(level, message) {
  if (app) app.log(level, message);
  else earlyLogs.push([level, message]);
}

async function loadConfig(url) {
  if (!url) return "";
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: ${response.status}`);
  return response.text();
}

function fillSelect(select, options, preferred) {
  select.replaceChildren(
    ...options.map(([label, value]) => new Option(label, value, false, value === preferred)),
  );
}

function showTopics(topics) {
  const label = (t) => `${t.topic} (${t.messages} messages)`;
  const clouds = topics.filter((t) => t.type === "sensor_msgs/msg/PointCloud2");
  const imus = topics.filter((t) => t.type === "sensor_msgs/msg/Imu");
  if (clouds.length === 0) throw new Error("the bag has no sensor_msgs/msg/PointCloud2 topic");
  fillSelect($("cloud-topic"), clouds.map((t) => [label(t), t.topic]));
  fillSelect(
    $("imu-topic"),
    [...imus.map((t) => [label(t), t.topic]), ["None (Lidar only)", ""]],
  );
  $("own-options").hidden = false;
  status("Choose the topics and a configuration, then start.");
}

/**
 * Pick the processing backend for this browser. A hardware WebGPU adapter
 * selects WebGPU, anything else runs on the WebAssembly CPU path. The worker
 * still falls back to CPU if WebGPU fails to start later on.
 */
async function probeBackend() {
  try {
    // Adapter discovery can hang on some systems, so it gets a deadline.
    const adapter = await Promise.race([
      navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }) ?? null,
      new Promise((resolve) => setTimeout(() => resolve(null), 3000)),
    ]);
    // A software adapter is slower than the CPU path.
    if (!adapter || (adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter)) return "cpu";
    return "webgpu";
  } catch {
    return "cpu";
  }
}
const backend = probeBackend();

async function start(play) {
  play = { backend: await backend, ...play };
  currentPlay = play;
  worker.postMessage({ type: "play", ...play });
}

function open(file) {
  currentFile = file;
  worker.postMessage({ type: "open", file });
}

/**
 * End the replay and return to the start page at the demo list. Through the
 * history entry the replay added, so the back button does the same.
 */
function backToDemos() {
  if (history.state?.replay) history.back();
  else leaveReplay();
}

function leaveReplay() {
  worker.terminate();
  try {
    sessionStorage.setItem("pelorus.scrollToDemos", "1");
  } catch {
    // The start page then opens at the top.
  }
  location.reload();
}

window.addEventListener("popstate", () => {
  if (!$("run").hidden) leaveReplay();
});

// After a replay the start page opens at the demo list, ready for the next one.
try {
  if (sessionStorage.getItem("pelorus.scrollToDemos")) {
    sessionStorage.removeItem("pelorus.scrollToDemos");
    history.scrollRestoration = "manual";
    requestAnimationFrame(() => $("demos").closest("section").scrollIntoView());
  }
} catch {
  // Storage can be unavailable, for example in private windows.
}

/** Replay the same bag with the same settings from the start, in a fresh worker. */
function restart() {
  worker.terminate();
  worker = createWorker();
  app.restart();
  cloudView?.clear();
  app.log("info", `Restarting ${bagName} from the beginning`);
  pendingPlay = currentPlay;
  open(currentFile);
}

function createWorker() {
  const created = new Worker("./worker.js", { type: "module" });
  created.onerror = onWorkerError;
  created.onmessage = onWorkerMessage;
  return created;
}

function onWorkerError(event) {
  const message = event.message || "the worker failed to start";
  if (app) app.log("error", message);
  else status(`Error: ${message}`);
}

async function onWorkerMessage(event) {
  const message = event.data;
  try {
    switch (message.type) {
      case "topics":
        if (pendingPlay) {
          start(pendingPlay);
          pendingPlay = null;
        } else {
          showTopics(message.topics);
        }
        break;
      case "started":
        // A restart keeps the view.
        if (app) {
          app.setBackend(message.backend);
          if (message.fallback) app.log("warn", message.fallback);
          break;
        }
        $("start").hidden = true;
        $("run").hidden = false;
        // The replay gets its own history entry, so the back button or gesture
        // returns to the start page instead of leaving the site.
        history.pushState({ replay: true }, "");
        try {
          cloudView = new PointCloudView($("cloud"));
        } catch (error) {
          $("layout").hidden = true;
          log("warn", `No 3D view: ${error.message}`);
        }
        app = new WebApp(
          "tui",
          bagName,
          message.vendor,
          backToDemos,
          restart,
          // Phones fit a usable number of columns with a smaller font.
          matchMedia("(max-width: 700px)").matches ? 11 : 14,
        );
        setLayout(cloudView ? "split" : "tui-only");
        app.setBackend(message.backend);
        if (message.fallback) app.log("warn", message.fallback);
        for (const [level, text] of earlyLogs.splice(0)) app.log(level, text);
        app.log("info", `Replaying ${bagName}. Press Space to restart, q to stop, ? for help.`);
        break;
      case "scan":
        app.pushScan(message.values, message.ms);
        cloudView?.addScan(message.points, [message.values[1], message.values[2], message.values[3]]);
        updateLegend();
        break;
      case "backend":
        app.setBackend(message.backend);
        if (message.fallback) app.log("warn", message.fallback);
        break;
      case "progress":
        log("info", `Replayed ${Math.round(message.value * 100)} % of the bag`);
        break;
      case "done":
        log(
          "info",
          `Replay finished: ${message.clouds} clouds in ${message.seconds.toFixed(1)} s. Press Space to replay, q to start over.`,
        );
        break;
      case "error":
        if (app) app.log("error", message.message);
        else status(`Error: ${message.message}`);
        break;
    }
  } catch (error) {
    status(`Error: ${error.message}`);
  }
}

worker = createWorker();

/** Send a key to the TUI, which only hears keys that reach its canvas. */
function sendKey(init) {
  $("tui").querySelector("canvas")?.dispatchEvent(new KeyboardEvent("keydown", init));
}

// While a replay runs, keys pressed anywhere on the page go to the TUI, and
// Space and Tab do not scroll the page or move the focus.
window.addEventListener(
  "keydown",
  (event) => {
    if ($("run").hidden) return;
    if (event.key === " " || event.key === "Tab") event.preventDefault();
    if (event.target !== $("tui").querySelector("canvas")) sendKey(event);
  },
  { capture: true },
);

// The touch buttons, for devices without a keyboard.
for (const button of document.querySelectorAll("#controls [data-key]")) {
  button.onclick = () => {
    button.blur();
    sendKey({ key: button.dataset.key });
  };
}
$("follow").onclick = () => {
  $("follow").blur();
  if (cloudView) cloudView.follow = true;
};

// The divider between the panes. Its position is the share of the first pane,
// which is the TUI side by side and the 3D view stacked, kept per orientation.
const portrait = matchMedia("(max-aspect-ratio: 1/1)");
const splitKey = () => (portrait.matches ? "pelorus.split.portrait" : "pelorus.split.landscape");
function applySplit() {
  let share = null;
  try {
    share = localStorage.getItem(splitKey());
  } catch {
    // Storage can be unavailable, for example in private windows.
  }
  // Without a stored position the 3D view gets the larger share.
  const fallback = portrait.matches ? "55%" : "40%";
  $("run").style.setProperty("--split", share ? `${share}%` : fallback);
}
function storeSplit(share) {
  $("run").style.setProperty("--split", `${share}%`);
  try {
    localStorage.setItem(splitKey(), String(share));
  } catch {
    // The split then only lasts for this page.
  }
}
portrait.addEventListener("change", applySplit);
applySplit();

const divider = $("divider");
divider.addEventListener("pointerdown", (event) => {
  divider.setPointerCapture(event.pointerId);
  divider.classList.add("dragging");
});
divider.addEventListener("pointermove", (event) => {
  if (!divider.hasPointerCapture(event.pointerId)) return;
  const run = $("run").getBoundingClientRect();
  const panes = portrait.matches ? run.height : run.width;
  const offset = portrait.matches ? event.clientY - run.top : event.clientX - run.left;
  storeSplit(Math.round(Math.min(85, Math.max(15, (offset / panes) * 100)) * 10) / 10);
});
const endDrag = () => divider.classList.remove("dragging");
divider.addEventListener("pointerup", endDrag);
divider.addEventListener("pointercancel", endDrag);
divider.addEventListener("dblclick", () => storeSplit(portrait.matches ? 55 : 40));

document.querySelector("#legend .ramp").style.background = turboGradient();

/** Show what the point colours mean, and the colour switch once intensities exist. */
function updateLegend() {
  if (!cloudView) return;
  const hasIntensity = cloudView.hasIntensity;
  $("color-by").hidden = !hasIntensity;
  const byIntensity = hasIntensity && cloudView.colorBy === "intensity";
  $("legend").className = byIntensity ? "by-intensity" : "by-height";
  for (const button of $("color-by").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.colorBy === cloudView.colorBy));
  }
  if (byIntensity) {
    const [low, high] = cloudView.intensityRange;
    $("legend-low").textContent = low.toFixed(0);
    $("legend-name").textContent = "intensity";
    $("legend-high").textContent = high.toFixed(0);
  } else {
    $("legend-low").textContent = `${HEIGHT_BELOW_M} m below`;
    $("legend-name").textContent = "height";
    $("legend-high").textContent = `${HEIGHT_ABOVE_M} m above`;
  }
}
for (const button of $("color-by").querySelectorAll("button")) {
  button.onclick = () => {
    button.blur();
    if (!cloudView) return;
    cloudView.colorBy = button.dataset.colorBy;
    updateLegend();
  };
}

/** Switch between the TUI, both side by side, and the 3D view. */
function setLayout(layout) {
  $("run").className = layout;
  for (const button of $("layout").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.layout === layout));
  }
  // The TUI draws its own trajectory only when the 3D view is hidden.
  app?.setTrajectoryElsewhere(Boolean(cloudView) && layout !== "tui-only");
}
for (const button of $("layout").querySelectorAll("button")) {
  button.onclick = () => {
    button.blur();
    setLayout(button.dataset.layout);
  };
}

/** Fetch a file and report the received fraction, or null without a known size, and the bytes. */
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    onProgress(total ? received / total : null, received);
  }
  return new Blob(parts);
}

/**
 * Download a demo bag and replay it with its topics and configuration. The
 * clicked Run button fills up with the download and shows its progress.
 */
async function runDemo(demo, button, buttons) {
  statusId = "demo-status";
  status("");
  for (const other of buttons) other.disabled = true;
  button.classList.add("loading");
  button.setAttribute("aria-busy", "true");
  const show = (fraction, text) => {
    button.style.setProperty("--progress", `${fraction * 100}%`);
    button.textContent = text;
  };
  show(0, "0 %");
  try {
    const blob = await fetchWithProgress(demo.url, (fraction, received) =>
      fraction === null
        ? show(0, `${(received / 1e6).toFixed(0)} MB`)
        : show(fraction, `${Math.floor(fraction * 100)} %`),
    );
    show(1, "Opening");
    bagName = demo.name;
    pendingPlay = { cloud: demo.cloud, imu: demo.imu, config: await loadConfig(demo.config) };
    open(blob);
  } catch (error) {
    button.classList.remove("loading");
    button.removeAttribute("aria-busy");
    button.textContent = "Run";
    for (const other of buttons) other.disabled = false;
    status(`Error: ${error.message}`);
  }
}

{
  const buttons = [];
  for (const group of DEMOS) {
    const heading = document.createElement("li");
    heading.className = "dataset";
    heading.textContent = group.dataset;
    $("demos").append(heading);
    for (const demo of group.demos) {
      const item = document.createElement("li");
      const text = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = demo.title;
      const detail = document.createElement("span");
      detail.textContent = demo.detail;
      text.append(title, detail);
      const button = document.createElement("button");
      button.className = "primary";
      button.textContent = "Run";
      button.setAttribute("aria-label", `Run the ${demo.title} demo`);
      demo.name = `${group.dataset.split(",")[0]} ${demo.title}`;
      button.onclick = () => runDemo(demo, button, buttons);
      buttons.push(button);
      item.append(text, button);
      $("demos").append(item);
    }
  }
}

$("bag-file").onchange = () => {
  const file = $("bag-file").files[0];
  if (!file) return;
  $("bag-file").nextElementSibling.textContent = file.name;
  statusId = "own-status";
  bagName = file.name;
  status(`Reading the summary of ${file.name}`);
  open(file);
};

fillSelect($("preset"), PRESETS, "");
const refreshConfig = await ratslangEditor($("config"));
$("preset").onchange = async () => {
  try {
    $("config").value = await loadConfig($("preset").value);
    refreshConfig();
  } catch (error) {
    status(`Error: ${error.message}`);
  }
};

$("run-own").onclick = () => {
  start({
    cloud: $("cloud-topic").value,
    imu: $("imu-topic").value || null,
    config: $("config").value,
  });
};
