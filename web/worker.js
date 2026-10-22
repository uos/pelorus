// Replays a bag through Pelorus off the main thread.
//
// Messages in:  {type: "open", file}  then  {type: "play", cloud, imu, config}
// Messages out: {type: "topics", topics}, {type: "started", vendor},
//               {type: "scan", values, ms, points}, {type: "progress", value},
//               {type: "done", clouds, seconds}, {type: "error", message}

import init, { BagStep, McapBag, Odometer } from "./pkg/pelorus_web.js";

const ready = init();
let file = null;
let bag = null;

/** Answer the byte range the bag asks for. */
async function read() {
  const offset = bag.readOffset();
  const bytes = await file.slice(offset, offset + bag.readLength()).arrayBuffer();
  bag.supply(new Uint8Array(bytes));
}

async function open(message) {
  file = message.file;
  bag = new McapBag(file.size);
  while (!bag.loadSummary()) await read();
  const topics = bag.topics().map((t) => ({
    topic: t.topic,
    type: t.messageType,
    messages: t.messages,
  }));
  postMessage({ type: "topics", topics });
}

async function play(message) {
  // Without an IMU topic Pelorus runs on the lidar alone.
  const config = message.imu ? message.config : `${message.config}\nimu.lidar_only = true\n`;
  const odometer = new Odometer(config, true);
  bag.play(message.cloud, message.imu ?? undefined);
  postMessage({ type: "started", vendor: odometer.lidarVendor() });

  const started = performance.now();
  let clouds = 0;
  let nextProgress = 0.1;
  for (;;) {
    const cloudStarted = performance.now();
    const step = bag.next(odometer);
    if (step === BagStep.NeedData) {
      await read();
      continue;
    }
    if (step === BagStep.End) break;
    clouds++;
    const scans = [];
    // Time spent on the 3D view's points, which is not part of the odometry.
    let viewMs = 0;
    for (let scan = odometer.nextScan(); scan; scan = odometer.nextScan()) {
      const viewStarted = performance.now();
      // Full resolution. The 3D view thins its history itself.
      scans.push({ values: scan.toArray(), points: odometer.worldPoints(0) });
      viewMs += performance.now() - viewStarted;
      scan.free();
    }
    const ms = performance.now() - cloudStarted - viewMs;
    for (const { values, points } of scans) {
      postMessage({ type: "scan", values, ms, points }, [values.buffer, points.buffer]);
    }
    const progress = bag.progress();
    if (progress >= nextProgress) {
      postMessage({ type: "progress", value: progress });
      nextProgress += 0.1;
    }
  }
  postMessage({ type: "done", clouds, seconds: (performance.now() - started) / 1000 });
}

onmessage = async (event) => {
  try {
    await ready;
    if (event.data.type === "open") await open(event.data);
    else if (event.data.type === "play") await play(event.data);
  } catch (error) {
    postMessage({ type: "error", message: error?.message ?? String(error) });
  }
};
