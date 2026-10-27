const recordings = {
  spires: { file: 'demo/keble_college_02_30s.mcap', config: 'configs/oxford-spires.rl', cloud: '/hesai/pandar', imu: '/alphasense_driver_ros/imu' },
  helipr: { file: 'demo/helipr_town02_20s.mcap', config: 'configs/helipr-ouster.rl', cloud: '/ouster/points', imu: '/imu/data_raw' },
};
const result = document.querySelector('#result');
const button = document.querySelector('#run');
const report = (message) => { result.textContent += `${message}\n`; };

async function fetchChecked(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

function replay(file, config, recording, backend) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./worker.js', {type: 'module'});
    const times = [];
    const poses = [];
    let actualBackend;
    let fallback;
    const fail = (error) => { worker.terminate(); reject(error); };
    worker.onerror = (event) => fail(new Error(event.message));
    worker.onmessage = ({data}) => {
      if (data.type === 'topics') {
        worker.postMessage({type:'play', config, cloud:recording.cloud, imu:recording.imu, backend});
      } else if (data.type === 'started' || data.type === 'backend') {
        actualBackend = data.backend;
        fallback = data.fallback ?? fallback;
        report(`${backend}: ${actualBackend}${fallback ? ` (${fallback})` : ''}`);
      } else if (data.type === 'scan') {
        times.push(data.ms);
        poses.push(data.values);
      } else if (data.type === 'progress') {
        report(`${backend}: ${Math.round(data.value * 100)}%`);
      } else if (data.type === 'error') {
        fail(new Error(data.message));
      } else if (data.type === 'done') {
        worker.terminate();
        times.sort((a,b) => a-b);
        const percentile = (p) => times[Math.min(times.length-1, Math.floor(times.length*p))] ?? NaN;
        const stats = {backend:actualBackend, fallback, seconds:data.seconds, clouds:data.clouds, poses,
          mean:times.reduce((a,b) => a+b,0)/times.length, p50:percentile(0.5), p95:percentile(0.95)};
        report(`${backend}: ${stats.clouds} clouds in ${stats.seconds.toFixed(2)} s; ` +
          `scan mean ${stats.mean.toFixed(2)} ms, p50 ${stats.p50.toFixed(2)} ms, p95 ${stats.p95.toFixed(2)} ms`);
        resolve(stats);
      }
    };
    worker.postMessage({type:'open', file});
  });
}

button.onclick = async () => {
  button.disabled = true;
  result.textContent = 'Downloading recording and configuration…\n';
  try {
    const recording = recordings[document.querySelector('#recording').value];
    const [bag, config] = await Promise.all([
      fetchChecked(recording.file).then(r => r.blob()),
      fetchChecked(recording.config).then(r => r.text()),
    ]);
    const file = new File([bag], 'benchmark.mcap');
    const mapType = document.querySelector('#map').value;
    const comparisonConfig = `${config}\nregistration.map.resolution.type = ${mapType}\n`;
    report(`Map: ${mapType}`);
    const cpu = await replay(file, comparisonConfig, recording, 'cpu');
    const gpu = await replay(file, comparisonConfig, recording, 'webgpu');
    if (gpu.fallback || gpu.backend !== 'WebGPU (hybrid)') {
      report('No GPU speed comparison: this run used CPU fallback.');
      return;
    }
    report(`Replay speedup: ${(cpu.seconds/gpu.seconds).toFixed(2)}×`);
    const reference = new Map(cpu.poses.map(p => [p[0],p]));
    let maximum = 0;
    let matched = 0;
    for (const pose of gpu.poses) {
      const expected = reference.get(pose[0]);
      if (!expected) continue;
      matched++;
      maximum = Math.max(maximum, Math.hypot(pose[1]-expected[1], pose[2]-expected[2], pose[3]-expected[3]));
    }
    report(`Matched poses: ${matched}/${cpu.poses.length} CPU and ${gpu.poses.length} GPU; maximum position difference ${maximum.toFixed(4)} m.`);
    report('One-run measurements; repeat on the target hardware. Position agreement is not a ground-truth accuracy evaluation.');
  } catch (error) {
    report(`Failed: ${error.message ?? String(error)}`);
  } finally {
    button.disabled = false;
  }
};
