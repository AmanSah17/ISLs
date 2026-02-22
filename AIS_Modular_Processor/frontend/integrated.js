const API_BASE = "http://localhost:8000";

let map;
let plsnLayerGroup;
let nlsnLayerGroup;
let rlsnLayerGroup;
let layerControl;

// Initialize Leaflet Map
function initMap() {
    map = L.map('integrated-map', {
        center: [36.0, -118.0],
        zoom: 4,
        zoomControl: false // Custom position
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Modern Dark Tile Layer (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB',
        maxZoom: 19
    }).addTo(map);

    // Initialize Empty Layer Groups
    plsnLayerGroup = L.layerGroup();
    nlsnLayerGroup = L.layerGroup();
    rlsnLayerGroup = L.layerGroup();

    // Setup Layer Control Toggle Box
    const baseMaps = {};
    const overlayMaps = {
        "<span style='color: #ef4444; font-weight:bold;'>PLSN Ports</span>": plsnLayerGroup,
        "<span style='color: #60a5fa; font-weight:bold;'>NLSN Shipping Lanes</span>": nlsnLayerGroup,
        "<span style='color: #10b981; font-weight:bold;'>RLSN Trajectories</span>": rlsnLayerGroup
    };
    layerControl = L.control.layers(baseMaps, overlayMaps, { position: 'topright' }).addTo(map);
}

// Fetch all available runs and populate the dropdown
async function loadAvailableRuns() {
    try {
        const res = await fetch(`${API_BASE}/runs`);
        const runs = await res.json();

        const selector = document.getElementById('run-selector');
        selector.innerHTML = '<option value="">-- Load a Run --</option>'; // clear

        // Filter out completely empty or failed runs for a better experience
        const validRuns = runs.filter(r => r.status === 'completed');

        validRuns.forEach(run => {
            const opt = document.createElement('option');
            opt.value = run.id;
            // Format to show hyperparams
            const plsn_k = run.config?.plsn?.k || '?';
            const plsn_r = run.config?.plsn?.r || '?';
            opt.textContent = `Run ${run.id}: ${run.run_name} (K:${plsn_k}, R:${plsn_r})`;
            selector.appendChild(opt);
        });

    } catch (err) {
        console.error("Failed to load runs list", err);
        alert("Could not connect to API to fetch runs.");
    }
}

// Main logic to fetch the unified GeoJSON and distribute to layers
async function renderNetworkForRun(runId) {
    if (!runId) return;

    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('hidden');

    try {
        // Clear all existing map layers
        plsnLayerGroup.clearLayers();
        nlsnLayerGroup.clearLayers();
        rlsnLayerGroup.clearLayers();

        const res = await fetch(`${API_BASE}/runs/${runId}/network`);
        if (!res.ok) throw new Error("Failed to fetch network geometries.");
        const featureCollection = await res.json();

        let validBounds = new L.LatLngBounds();

        // Use L.geoJSON to parse and distribute styling natively
        L.geoJSON(featureCollection, {
            style: function (feature) {
                const layerType = feature.properties.layer;
                if (layerType === 'NLSN') {
                    // Shipping Edges styling
                    return { color: '#60a5fa', weight: 3, opacity: 0.8, dashArray: '5, 5' };
                } else if (layerType === 'RLSN') {
                    // Trajectories styling
                    return { color: '#10b981', weight: 1, opacity: 0.4 };
                }
            },
            pointToLayer: function (feature, latlng) {
                // PLSN Ports styling (points)
                if (feature.properties.layer === 'PLSN') {
                    return L.circleMarker(latlng, {
                        radius: 6,
                        fillColor: '#ef4444',
                        color: '#b91c1c',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    });
                }
            },
            onEachFeature: function (feature, layer) {
                // Add popups and push to respective groups
                const p = feature.properties;
                validBounds.extend(layer.getBounds ? layer.getBounds() : layer.getLatLng());

                if (p.layer === 'PLSN') {
                    layer.bindPopup(`<b>PLSN Port</b><br>ID: ${p.db_id}`);
                    plsnLayerGroup.addLayer(layer);
                } else if (p.layer === 'NLSN') {
                    layer.bindPopup(`<b>NLSN Edge</b><br>ID: ${p.db_id}`);
                    nlsnLayerGroup.addLayer(layer);
                } else if (p.layer === 'RLSN') {
                    layer.bindPopup(`<b>RLSN Trajectory</b><br>MMSI: ${p.mmsi}`);
                    rlsnLayerGroup.addLayer(layer);
                }
            }
        });

        // Add all layer groups to map by default on load
        plsnLayerGroup.addTo(map);
        nlsnLayerGroup.addTo(map);
        rlsnLayerGroup.addTo(map);

        // Fly to bounding box of all geometries if valid
        if (validBounds.isValid()) {
            map.flyToBounds(validBounds, { padding: [50, 50], duration: 1.5 });
        } else {
            alert("No spatial geometries found for this run.");
        }

    } catch (err) {
        console.error(err);
        alert(`Error rendering network: ${err.message}`);
    } finally {
        overlay.classList.add('hidden');
    }
}

// Lifecycle Initialization
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadAvailableRuns();

    // Bind Selector Logic
    document.getElementById('run-selector').addEventListener('change', (e) => {
        renderNetworkForRun(e.target.value);
    });
});
