from shiny import App, ui, render, reactive
import requests
import json
import time
import html
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/1132-NCCU-DataScience/finalproject-finalproject_group7/main/predictions/latest.json"

MAP_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; }}

  #map {{ width: 100%; height: 100vh; }}

  /* ── Top control bar ── */
  #control-bar {{
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 1000; display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.96); border-radius: 32px;
    padding: 6px 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    border: 1px solid rgba(0,0,0,0.08);
  }}
  .mode-btn {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 24px; border: none; cursor: pointer;
    font-size: 13px; font-weight: 500; transition: all .2s;
    background: transparent; color: #555;
    white-space: nowrap;
  }}
  .mode-btn .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .mode-btn.active {{ background: #1a1a2e; color: #fff; }}
  .mode-btn.active .dot {{ background: #fff; }}
  .mode-sep {{ width: 1px; height: 20px; background: #ddd; }}

  /* ── Legend panel (right side) ── */
  #legend-toggle {{
    position: absolute; top: 12px; right: 14px; z-index: 1001;
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255,255,255,0.96); border: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0 2px 10px rgba(0,0,0,0.14); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; transition: transform .2s;
  }}
  #legend-toggle:hover {{ transform: scale(1.08); }}

  #legend-panel {{
    position: absolute; top: 58px; right: 14px; z-index: 1001;
    background: rgba(255,255,255,0.97); border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.09); box-shadow: 0 4px 20px rgba(0,0,0,0.14);
    padding: 14px 16px; min-width: 200px;
    display: none; flex-direction: column; gap: 8px;
  }}
  #legend-panel.open {{ display: flex; }}
  .legend-title {{ font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 2px; }}
  .legend-row {{ display: flex; align-items: center; gap: 10px; font-size: 13px; color: #333; }}
  .legend-circle {{
    width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0;
    border: 1.5px solid rgba(0,0,0,0.15);
  }}
  .legend-section {{ display: flex; flex-direction: column; gap: 5px; }}
  .legend-divider {{ border: none; border-top: 1px solid #eee; margin: 2px 0; }}
  .legend-note {{ font-size: 11px; color: #999; margin-top: 2px; }}

  /* marker cluster overrides */
  .marker-cluster-small  {{ background-color: rgba(100,160,220,.4); }}
  .marker-cluster-small div  {{ background-color: rgba(80,130,200,.6); }}
  .marker-cluster-medium {{ background-color: rgba(230,140,50,.4); }}
  .marker-cluster-medium div {{ background-color: rgba(210,110,30,.6); }}
  .marker-cluster-large  {{ background-color: rgba(200,60,60,.35); }}
  .marker-cluster-large div  {{ background-color: rgba(180,40,40,.6); }}
  .marker-cluster div {{ color: #fff; font-size: 12px; font-weight: 600; }}
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
</head>
<body>
<div id="map"></div>

<!-- Control bar -->
<div id="control-bar">
  <button class="mode-btn active" id="btn-shortage" onclick="setMode('shortage')">
    <span class="dot" style="background:#e53"></span> 缺車率
  </button>
  <div class="mode-sep"></div>
  <button class="mode-btn" id="btn-lisa" onclick="setMode('lisa')">
    <span class="dot" style="background:#7a6bd8"></span> LISA 空間
  </button>
</div>

<!-- Legend toggle -->
<div id="legend-toggle" onclick="toggleLegend()" title="圖例說明" aria-label="顯示圖例">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color:#444">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="8" stroke-width="3"/><line x1="12" y1="12" x2="12" y2="16"/>
  </svg>
</div>
<div id="legend-panel">
  <div id="legend-shortage-section" class="legend-section">
    <div class="legend-title">缺車率</div>
    <div class="legend-row"><span class="legend-circle" style="background:#c0392b"></span> &gt; 80%　嚴重缺車</div>
    <div class="legend-row"><span class="legend-circle" style="background:#e67e22"></span> 60–80%　高缺車</div>
    <div class="legend-row"><span class="legend-circle" style="background:#f1c40f"></span> 40–60%　中度缺車</div>
    <div class="legend-row"><span class="legend-circle" style="background:#27ae60"></span> 20–40%　尚可</div>
    <div class="legend-row"><span class="legend-circle" style="background:#2980b9"></span> &lt; 20%　充足</div>
  </div>
  <div id="legend-lisa-section" class="legend-section" style="display:none">
    <div class="legend-title">LISA 空間群聚</div>
    <div class="legend-row"><span class="legend-circle" style="background:#c0392b"></span> HH — 高-高群聚</div>
    <div class="legend-row"><span class="legend-circle" style="background:#2980b9"></span> LL — 低-低群聚</div>
    <div class="legend-row"><span class="legend-circle" style="background:#e67e22"></span> HL — 高-低異常</div>
    <div class="legend-row"><span class="legend-circle" style="background:#85c1e9"></span> LH — 低-高異常</div>
    <div class="legend-row"><span class="legend-circle" style="background:#7f8c8d"></span> NS — 不顯著</div>
  </div>

</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script>
const RAW_DATA = {data_json};

const map = L.map('map').setView([25.0330, 121.5654], 13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
}}).addTo(map);

let currentMode = 'shortage';
let clusterGroup = null;

function shortageColor(rate) {{
  if (rate > 0.8) return '#c0392b';
  if (rate > 0.6) return '#e67e22';
  if (rate > 0.4) return '#f1c40f';
  if (rate > 0.2) return '#27ae60';
  return '#2980b9';
}}

const lisaColorMap = {{
  HH: '#c0392b', LL: '#2980b9', HL: '#e67e22', LH: '#85c1e9', NS: '#7f8c8d'
}};

function buildMarkers(mode) {{
  if (clusterGroup) map.removeLayer(clusterGroup);

  clusterGroup = L.markerClusterGroup({{
    maxClusterRadius: 60,
    iconCreateFunction: function(cluster) {{
      const children = cluster.getAllChildMarkers();
      const count = children.length;
      const size = count > 50 ? 48 : count > 20 ? 38 : 30;
      let bgColor;
      if (mode === 'shortage') {{
        const avgRate = children.reduce((s,m) => s + m.options._shortageRate, 0) / count;
        bgColor = shortageColor(avgRate);
      }} else {{
        const freq = {{}};
        children.forEach(m => {{ const t = m.options._lisaType; freq[t] = (freq[t]||0)+1; }});
        const dominant = Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0];
        bgColor = lisaColorMap[dominant] || '#bdc3c7';
      }}
      return L.divIcon({{
        html: `<div style="width:${{size}}px;height:${{size}}px;border-radius:50%;background:${{bgColor}}cc;border:2.5px solid ${{bgColor}};display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:700;text-shadow:0 1px 2px rgba(0,0,0,.4)">${{count}}</div>`,
        className: '',
        iconSize: [size, size],
        iconAnchor: [size/2, size/2]
      }});
    }}
  }});

  RAW_DATA.forEach(row => {{
    const isSignificant = row.moran_p_value < 0.05;
    const lisaLabel = isSignificant ? row.moran_type : 'NS（不顯著）';
    const color = mode === 'shortage'
      ? shortageColor(row.shortage_rate)
      : lisaColorMap[isSignificant ? row.moran_type : 'NS'];

    const radius = 6;
    const opacity = (mode === 'lisa' && !isSignificant) ? 0.35 : 0.78;

    const popup = `
      <div style="font-family:sans-serif;font-size:13px;line-height:1.6">
        <b style="font-size:14px">站點 ${{row.sno}}</b><br>
        當下缺車率：<b style="color:#e74c3c">${{Math.round(row.shortage_rate * 100)}}%</b><br>
        可借車輛：${{row.available_bikes}} / ${{row.total_capacity}}<br>
        LISA 群聚：${{lisaLabel}}<br>
        <hr style="margin:5px 0;border:none;border-top:1px solid #eee"/>
        60分缺車機率：<b style="color:#c0392b">${{Math.round(row.pred_prob * 100)}}%</b>
      </div>`;

    const lisaType = (row.moran_p_value < 0.05) ? row.moran_type : 'NS';
    const marker = L.circleMarker([row.lat, row.lng], {{
      radius, color, fillColor: color,
      weight: 1.5, fillOpacity: opacity,
      stroke: true,
      _shortageRate: row.shortage_rate,
      _lisaType: lisaType
    }}).bindPopup(popup, {{ maxWidth: 260 }});

    clusterGroup.addLayer(marker);
  }});

  map.addLayer(clusterGroup);
}}

function setMode(mode) {{
  currentMode = mode;
  document.getElementById('btn-shortage').classList.toggle('active', mode === 'shortage');
  document.getElementById('btn-lisa').classList.toggle('active', mode === 'lisa');
  document.getElementById('legend-shortage-section').style.display = mode === 'shortage' ? 'flex' : 'none';
  document.getElementById('legend-lisa-section').style.display = mode === 'lisa' ? 'flex' : 'none';
  buildMarkers(mode);
}}

function toggleLegend() {{
  document.getElementById('legend-panel').classList.toggle('open');
}}

buildMarkers('shortage');
</script>
</body>
</html>
"""

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { margin: 0; padding: 0; }
        .bslib-page-fluid { padding: 0 !important; }
        #status-bar {
            display: flex; align-items: center; gap: 16px;
            padding: 8px 18px; background: #fff;
            border-bottom: 1px solid #e8e8e8;
            font-size: 13px; font-family: sans-serif;
        }
    """),
    ui.output_ui("status_banner"),
    ui.output_ui("map_view"),
)

def server(input, output, session):
    @reactive.poll(lambda: int(time.time() / 100))
    def fetch_latest_data():
        """Read the latest inference results from local JSON"""
        with open("predictions/latest.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # @reactive.poll(lambda: int(time.time() / 60))
    # def fetch_latest_data():
    #     ts = int(time.time() / 60)
    #     busted_url = f"{DATA_URL}?t={ts}"
    #     return requests.get(busted_url, timeout=10).json()

    @render.ui
    def status_banner():
        data = fetch_latest_data()
        status = data.get("health_status", "ok")
        update_time = data.get("update_time", "未知時間")
        model_ver = data.get("model_version", "—")

        icons = {
            "ok": ("🟢", "#1a7a4a", "系統正常"),
            "stale": ("🟡", "#ece332", "資料延遲中"),
            "degraded": ("🟠", "#d35400", "嚴重延遲"),
            "error": ("🔴", "#b71c1c", "系統異常")
        }
        icon, color, label = icons.get(status, icons["error"])

        return ui.HTML(f"""
        <div id="status-bar">
          <span style="color:{color};font-weight:600">{icon} {label}</span>
          <span style="color:#888">模型版本：{model_ver}</span>
          <span style="color:#888">更新時間：{update_time}</span>
        </div>""")

    @render.ui
    def map_view():
        data = fetch_latest_data()
        predictions = data.get("predictions", [])
        data_json = json.dumps(predictions, ensure_ascii=False)
        map_html = MAP_TEMPLATE.format(data_json=data_json)
        return ui.HTML(f'<iframe srcdoc="{html.escape(map_html)}" style="width:100%;height:calc(100vh - 42px);border:none;display:block"></iframe>')


app = App(app_ui, server)