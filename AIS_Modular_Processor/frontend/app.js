// App Constants
const API_BASE = 'http://localhost:8000';
let map, drawControl, currentFileId = null;
let currentDrawnLayer = null;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initTabs();
    loadRegions();
});

function initMap() {
    map = L.map('map', {
        zoomControl: false
    }).setView([15.0, 85.0], 4);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB'
    }).addTo(map);

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Leaflet Draw Setup
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    drawControl = new L.Control.Draw({
        edit: { featureGroup: drawnItems },
        draw: {
            polygon: {
                allowIntersection: false,
                showArea: true,
                drawError: { color: '#e1e100', message: 'No intersections allowed!' },
                shapeOptions: { color: '#38bdf8' }
            },
            polyline: false, rectangle: true, circle: false, marker: false, circlemarker: false
        }
    });

    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (e) => {
        if (currentDrawnLayer) drawnItems.removeLayer(currentDrawnLayer);
        currentDrawnLayer = e.layer;

        // Add toggle functionality for drawn region
        currentDrawnLayer.bindPopup(`
            <b>Selected Area</b><br>
            <button onclick="toggleRegionView()" class="btn secondary" style="margin-top:5px; padding:4px;">Hide/Show Matrix</button>
        `);
        drawnItems.addLayer(currentDrawnLayer);
    });
}

function toggleRegionView() {
    if (currentDrawnLayer) {
        if (map.hasLayer(currentDrawnLayer)) {
            map.removeLayer(currentDrawnLayer);
        } else {
            map.addLayer(currentDrawnLayer);
        }
    }
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn, .tab-content section').forEach(el => el.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-tab`).classList.add('active');
        });
    });
}

// File Upload Logic
document.getElementById('ais-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const status = document.getElementById('file-status');
    status.innerText = 'Uploading...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        const data = await res.json();
        currentFileId = data.file_id;
        status.innerText = `✓ ${file.filename} ready`;
    } catch (err) {
        status.innerText = '❌ Upload failed';
        console.error(err);
    }
});

async function loadRegions() {
    const select = document.getElementById('saved-regions');
    try {
        const res = await fetch(`${API_BASE}/regions`);
        const regions = await res.json();
        regions.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.id;
            opt.innerText = r.name;
            select.appendChild(opt);
        });
    } catch (err) {
        console.warn('API not available for regions yet');
    }
}

async function saveCurrentRegion() {
    if (!currentDrawnLayer) return alert('Please draw a polygon on the map first.');

    const name = prompt('Region Name:', 'My Custom Area');
    if (!name) return;

    const geojson = currentDrawnLayer.toGeoJSON();

    try {
        const res = await fetch(`${API_BASE}/regions?name=${encodeURIComponent(name)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(geojson)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'API error during save');
        alert(`Region saved with ID: ${data.id}`);
        loadRegions();
    } catch (err) {
        alert(`Failed to save region: ${err.message}`);
    }
}

async function startAnalysis() {
    if (!currentFileId) return alert('Please select a Parquet file first.');
    const regionId = document.getElementById('saved-regions').value;
    if (!regionId && !currentDrawnLayer) return alert('Please select or draw a region.');

    const overlay = document.getElementById('status-overlay');
    overlay.classList.remove('hidden');

    try {
        let finalRegionId = regionId;
        if (!finalRegionId) {
            const name = `Quick Run ${new Date().toLocaleTimeString()}`;
            const gj = currentDrawnLayer.toGeoJSON();
            const res = await fetch(`${API_BASE}/regions?name=${encodeURIComponent(name)}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(gj)
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Region creation aborted');
            finalRegionId = data.id;
        }

        const config = {
            plsn: { k: 1400, r: 0.0001 },
            nlsn: { gamma: 0.05 },
            tuning: document.getElementById('k-values').value ? {
                k_values: document.getElementById('k-values').value.split(',').map(v => parseInt(v.trim())),
                r_values: document.getElementById('r-values').value.split(',').map(v => parseFloat(v.trim()))
            } : null
        };

        const res = await fetch(`${API_BASE}/analyze/plsn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ais_file_id: currentFileId,
                region_id: parseInt(finalRegionId),
                config: config
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Analysis rejected by server');
        pollStatus(data.task_id, data.run_id, config, 'PLSN');
    } catch (err) {
        overlay.classList.add('hidden');
        alert(`Processing failed to start: ${err.message}`);
    }
}

async function startNlsn(runId, config) {
    const overlay = document.getElementById('status-overlay');
    overlay.classList.remove('hidden');
    document.getElementById('status-title').innerText = "INITIATING NLSN";
    document.getElementById('status-desc').innerText = "Gathering spatial state...";
    if (document.getElementById('status-progress-bar')) {
        document.getElementById('status-progress-bar').style.width = `0%`;
        document.getElementById('status-percent').innerText = `0%`;
    }

    try {
        const res = await fetch(`${API_BASE}/analyze/nlsn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: runId,
                config: config
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'NLSN Analysis rejected');
        pollStatus(data.task_id, runId, config, 'NLSN');
    } catch (err) {
        overlay.classList.add('hidden');
        alert(`NLSN Processing failed to start: ${err.message}`);
    }
}

async function startRlsn(runId, config) {
    const overlay = document.getElementById('status-overlay');
    overlay.classList.remove('hidden');
    document.getElementById('status-title').innerText = "INITIATING RLSN";
    document.getElementById('status-desc').innerText = "Gathering spatial state...";
    if (document.getElementById('status-progress-bar')) {
        document.getElementById('status-progress-bar').style.width = `0%`;
        document.getElementById('status-percent').innerText = `0%`;
    }

    try {
        const res = await fetch(`${API_BASE}/analyze/rlsn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: runId,
                config: config
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'RLSN Analysis rejected');
        pollStatus(data.task_id, runId, config, 'RLSN');
    } catch (err) {
        overlay.classList.add('hidden');
        alert(`RLSN Processing failed to start: ${err.message}`);
    }
}

async function pollStatus(taskId, runId, config, phase) {
    const title = document.getElementById('status-title');
    const desc = document.getElementById('status-desc');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/status/${taskId}`);
            const data = await res.json();

            let msg = data.status.toUpperCase();
            if (data.status.includes('processing_')) {
                const subPhase = data.status.split('_')[1].toUpperCase();
                msg = `${subPhase} ANALYSIS`;
                desc.innerText = `Extracting ${subPhase} patterns & saving to DB...`;
            } else if (data.status === 'processing') {
                msg = 'DATA PREP';
                desc.innerText = 'Clipping GeoJSON Region & Loading Parquet...';
            }

            if (data.metadata) {
                if (data.metadata.message) desc.innerText = data.metadata.message;
                if (data.metadata.progress !== undefined) {
                    document.getElementById('progress-container').style.display = 'block';
                    document.getElementById('status-progress-bar').style.width = `${data.metadata.progress}%`;
                    document.getElementById('status-percent').innerText = `${data.metadata.progress}%`;
                }
                if (data.metadata.elapsed !== undefined) {
                    document.getElementById('status-time').innerText = `Elapsed: ${data.metadata.elapsed}s`;
                }
            } else {
                if (document.getElementById('progress-container')) {
                    document.getElementById('progress-container').style.display = 'none';
                }
            }

            title.innerText = msg;

            if (data.status === 'completed_plsn' && phase === 'PLSN') {
                clearInterval(interval);
                document.getElementById('status-overlay').classList.add('hidden');

                const fullRes = await fetch(`${API_BASE}/runs/${runId}`);
                const fullData = await fullRes.json();
                if (fullData.tuning_results && fullData.tuning_results.length > 0) {
                    renderTuningChart(fullData.tuning_results);
                }

                setTimeout(() => {
                    if (confirm(`PLSN Extraction Complete!\nFound ${data.metadata?.plsn_nodes || 0} Port Nodes.\n\nProceed to extract NLSN Shipping Lanes?`)) {
                        startNlsn(runId, config);
                    } else {
                        loadRuns();
                    }
                }, 100);
            }
            else if (data.status === 'completed_nlsn' && phase === 'NLSN') {
                clearInterval(interval);
                document.getElementById('status-overlay').classList.add('hidden');

                setTimeout(() => {
                    if (confirm(`NLSN Extraction Complete!\nProcessed edges dynamically.\n\nProceed to extract RLSN Trajectories?`)) {
                        startRlsn(runId, config);
                    } else {
                        loadRuns();
                    }
                }, 100);
            }
            else if (data.status === 'completed' && phase === 'RLSN') {
                clearInterval(interval);
                document.getElementById('status-overlay').classList.add('hidden');
                loadRuns();

                setTimeout(() => {
                    if (confirm(`Pipeline Fully Complete!\nGenerated ${data.metadata?.rlsn_routes || 0} Routes.\n\nView results in the Integrated Network Dashboard?`)) {
                        window.location.href = "integrated.html";
                    }
                }, 100);
            }
            else if (data.status === 'failed') {
                clearInterval(interval);
                alert(`Error during ${phase}: ${data.metadata?.error || 'Unknown error'}`);
                document.getElementById('status-overlay').classList.add('hidden');
            }
        } catch (err) {
            clearInterval(interval);
            console.error('Polling failed');
        }
    }, 1500);
}

function renderTuningChart(tuningData) {
    const ctx = document.getElementById('tuning-chart').getContext('2d');

    // Clear existing chart if any
    if (window.tuningChartInstance) window.tuningChartInstance.destroy();

    const labels = tuningData.map(d => `K:${d.k} R:${d.r}`);
    const clusters = tuningData.map(d => d.clusters_found);
    const noise = tuningData.map(d => d.noise_points / 100);

    window.tuningChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Clusters Found',
                    data: clusters,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.2)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Noise (x100)',
                    data: noise,
                    borderColor: '#f43f5e',
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { size: 10 } } }
            },
            scales: {
                y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

async function loadRuns() {
    const container = document.getElementById('runs-list');
    try {
        const res = await fetch(`${API_BASE}/runs`);
        const runs = await res.json();
        container.innerHTML = runs.map(run => `
            <div class="card run-entry" onclick="viewRunDetail(${run.id})">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <h4 style="font-size:0.9rem">${run.run_name}</h4>
                    <span class="status-msg" style="color:${run.status === 'completed' ? '#10b981' : '#f43f5e'}">${run.status}</span>
                </div>
                <p class="desc" style="margin-top:4px">${new Date(run.created_at).toLocaleString()}</p>
            </div>
        `).join('') || '<p class="empty-state">No recent runs</p>';
    } catch (err) { }
}

async function viewRunDetail(runId) {
    const res = await fetch(`${API_BASE}/runs/${runId}`);
    const data = await res.json();
    if (data.tuning_results && data.tuning_results.length > 0) {
        renderTuningChart(data.tuning_results);
        document.querySelector('[data-tab="tuning"]').click();
    }
}
