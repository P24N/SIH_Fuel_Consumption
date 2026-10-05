'use strict';
function parseCSV(source) {
  const rows = []; let row = [], field = '', quoted = false;
  source = source.replace(/^\uFEFF/, '');
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (c === '"') {
      if (quoted && source[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (c === ',' && !quoted) { row.push(field); field = ''; }
    else if ((c === '\n' || c === '\r') && !quoted) {
      if (c === '\r' && source[i + 1] === '\n') i++;
      row.push(field); if (row.some(v => v.trim())) rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (quoted) throw new Error('CSV contains an unclosed quote.');
  row.push(field); if (row.some(v => v.trim())) rows.push(row);
  if (rows.length < 2) throw new Error('CSV needs a header and at least one data row.');
  const headers = rows.shift().map(v => v.trim());
  if (new Set(headers).size !== headers.length) throw new Error('CSV headers must be unique.');
  return rows.map(values => {
    if (values.length !== headers.length) throw new Error('CSV row has the wrong number of columns.');
    return Object.fromEntries(headers.map((key, i) => [key, values[i]]));
  });
}
function toCSV(rows) {
  if (!rows.length) return '';
  const keys = Object.keys(rows[0]);
  const escape = value => {
    let s = String(value ?? '');
    // Prevent spreadsheet formula execution from user-controlled names.
    if (/^[=+@\t\r]/.test(s) || (/^-/.test(s) && !Number.isFinite(Number(s)))) s = "'" + s;
    return '"' + s.replaceAll('"', '""') + '"';
  };
  return [keys, ...rows.map(r => keys.map(k => r[k]))].map(r => r.map(escape).join(',')).join('\r\n');
}
if (typeof module !== 'undefined') module.exports = {parseCSV, toCSV};
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = {fleet: [], routes: [], choices: {}, busy: false, ready: false};
  const fleetKeys = ['Vessel', 'Ship_Type', 'Engine_Type', 'Maintenance_Status', 'Fuel_Type', 'Draft_meters', 'Max_Capacity'];
  const routeKeys = ['route_id', 'route_type', 'distance_nm', 'cargo_weight_tons', 'weather_condition', 'max_arrival_hours'];
  const numeric = new Set(['Draft_meters', 'Max_Capacity', 'distance_nm', 'cargo_weight_tons', 'max_arrival_hours']);
  function el(tag, text, className) {
    const n = document.createElement(tag); if (text !== undefined) n.textContent = text;
    if (className) n.className = className; return n;
  }
  function message(s) { $('message').textContent = s; }
  async function api(action, body) {
    const response = await fetch('/api?action=' + encodeURIComponent(action), {
      method: body ? 'POST' : 'GET', cache: 'no-store',
      headers: body ? {'Content-Type': 'application/json'} : {},
      body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(65000)
    });
    let data;
    try { data = await response.json(); } catch { throw new Error('Server returned an unexpected response. Try again after the deployment is ready.'); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  function download(rows, filename) {
    const url = URL.createObjectURL(new Blob([toCSV(rows)], {type: 'text/csv;charset=utf-8'}));
    const a = el('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function editor(kind) {
    const keys = kind === 'fleet' ? fleetKeys : routeKeys;
    const table = el('table'), head = el('tr');
    for (const key of keys) head.append(el('th', key.replaceAll('_', ' ')));
    head.append(el('th', '')); const thead = el('thead'); thead.append(head); table.append(thead);
    const tbody = el('tbody');
    state[kind].forEach((row, i) => {
      const tr = el('tr');
      for (const key of keys) {
        const td = el('td'); let input;
        if (state.choices[key]) {
          input = el('select');
          for (const option of state.choices[key]) { const o = el('option', option); o.value = option; input.append(o); }
        } else { input = el('input'); input.type = numeric.has(key) ? 'number' : 'text'; input.step = 'any'; }
        input.value = row[key]; input.setAttribute('aria-label', `${kind} row ${i + 1} ${key}`);
        input.addEventListener('change', () => { row[key] = numeric.has(key) ? Number(input.value) : input.value; });
        td.append(input); tr.append(td);
      }
      const td = el('td'), remove = el('button', 'Remove');
      remove.onclick = () => { state[kind].splice(i, 1); editor(kind); }; td.append(remove); tr.append(td); tbody.append(tr);
    });
    table.append(tbody); $(kind + '-editor').replaceChildren(table);
  }
  function table(rows) {
    const wrap = el('div', undefined, 'table-wrap');
    if (!rows?.length) { wrap.append(el('p', 'No result rows.')); return wrap; }
    const t = el('table'), tr = el('tr'), keys = Object.keys(rows[0]); keys.forEach(k => tr.append(el('th', k)));
    const head = el('thead'); head.append(tr); t.append(head); const body = el('tbody');
    rows.forEach(row => { const r = el('tr'); keys.forEach(k => r.append(el('td', typeof row[k] === 'number' ? row[k].toLocaleString(undefined, {maximumFractionDigits: 3}) : row[k] ?? 'Unavailable'))); body.append(r); });
    t.append(body); wrap.append(t); return wrap;
  }
  function chart(history) {
    const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 600 150'); svg.setAttribute('class', 'chart'); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Objective value by generation');
    const lo = Math.min(...history), hi = Math.max(...history), span = hi - lo || 1;
    const line = document.createElementNS(ns, 'polyline');
    line.setAttribute('points', history.map((v, i) => `${20 + i / Math.max(1, history.length - 1) * 560},${120 - (v - lo) / span * 90}`).join(' ')); svg.append(line);
    const label = document.createElementNS(ns, 'text'); label.setAttribute('x', '20'); label.setAttribute('y', '145');
    label.textContent = `Generation 1 → ${history.length} · Objective ${hi.toFixed(1)} → ${history.at(-1).toFixed(1)}`; svg.append(label); return svg;
  }
  function report(target, title, data) {
    const section = el('div', undefined, 'result'); section.append(el('h2', title));
    for (const w of data.warnings || []) section.append(el('div', w, 'warning'));
    const metrics = el('div', undefined, 'metrics');
    const values = [];
    if (data.runtime !== undefined) values.push(['Solver time', data.runtime.toFixed(3) + ' s']);
    if (data.variables !== undefined) values.push(['Decision variables', data.variables]);
    if (data.rows?.[0]?.['Assigned Vessel']) {
      const sum = key => data.rows.reduce((s, r) => s + (r[key] || 0), 0);
      values.push(['Energy', (sum('Energy (kWh)') / 1000).toFixed(1) + ' MWh'], ['Emissions', (sum('Emissions (kg CO2e)') / 1000).toFixed(2) + ' t'], ['Effective cost', '$' + sum('Total Cost (USD)').toLocaleString()]);
    }
    values.forEach(([k, v]) => { const box = el('div', undefined, 'metric'); box.append(el('span', k), el('strong', v)); metrics.append(box); });
    section.append(metrics);
    if (data.history?.length) section.append(chart(data.history));
    if (data.rows) { section.append(table(data.rows)); const b = el('button', 'Download results CSV'); b.onclick = () => download(data.rows, title.toLowerCase().replaceAll(' ', '-') + '.csv'); section.append(b); }
    if (data.qga) report(section, 'QGA', data.qga);
    if (data.milp) report(section, 'MILP', data.milp);
    target.append(section);
  }
  function controls() {
    const n = id => Number($(id).value);
    return {fleet: state.fleet, routes: state.routes, weights: [n('energy'), n('emissions'), n('cost')],
      carbon_tax: n('tax'), generations: n('generations'), population: n('population'), seed: n('seed'),
      n_routes: n('n_routes'), n_vessels: n('n_vessels')};
  }
  document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => {
    document.querySelectorAll('.panel').forEach(p => p.hidden = p.id !== button.dataset.tab);
    document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('selected', b === button));
  });
  document.querySelectorAll('.run').forEach(button => {
    button.disabled = true;
    button.onclick = async () => {
      if (state.busy || !state.ready) return;
      state.busy = true; document.querySelectorAll('.run').forEach(b => b.disabled = true);
      const action = button.dataset.action; message('Running ' + action + '… Keep this tab open.');
      // Capture an immutable input snapshot before awaiting the server.
      const payload = structuredClone(controls());
      try {
        const data = await api(action, payload), target = $('result-' + action); target.replaceChildren();
        report(target, {qga: 'QGA dispatch', milp: 'Classical MILP', stress: 'Scaling benchmark', dynamic: 'Speed & berth benchmark', fuels: 'Fuel pathways'}[action], data);
        message('Complete. Results reflect the inputs at the time you started this run.');
      } catch (error) { message(error.name === 'TimeoutError' ? 'This run timed out. Reduce the problem size and retry.' : error.message); }
      finally { state.busy = false; document.querySelectorAll('.run').forEach(b => b.disabled = false); }
    };
  });
  for (const kind of ['fleet', 'routes']) {
    $('export-' + kind).onclick = () => download(state[kind], kind + '.csv');
    $('add-' + kind).onclick = () => {
      if (!state.ready) return;
      if (state[kind].length >= 20) return message('Maximum 20 rows per scenario.');
      const row = kind === 'fleet' ? {Vessel: 'Vessel-' + (state.fleet.length + 1), Ship_Type: 'Bulk Carrier', Engine_Type: 'Four-Stroke', Maintenance_Status: 'Good', Fuel_Type: 'Heavy Fuel Oil (HFO)', Draft_meters: 10, Max_Capacity: 50000} : {route_id: 'VOY-' + (state.routes.length + 1), route_type: 'Coastal', distance_nm: 500, cargo_weight_tons: 10000, weather_condition: 'Calm', max_arrival_hours: 60};
      state[kind].push(row); editor(kind);
    };
    $('import-' + kind).onchange = async event => {
      const file = event.target.files[0]; if (!file || !state.ready) return;
      try {
        if (file.size > 100000) throw new Error('CSV must be under 100 KB.');
        const rows = parseCSV(await file.text()), keys = kind === 'fleet' ? fleetKeys : routeKeys;
        if (rows.length > 20) throw new Error('Maximum 20 rows per scenario.');
        rows.forEach((row, i) => {
          // Accept CSV exports from the original Streamlit fleet table too.
          if (kind === 'fleet' && !row.Vessel) row.Vessel = row.Ship_Type;
          keys.forEach(k => { if (!(k in row)) throw new Error('Missing CSV column: ' + k); if (numeric.has(k)) { row[k] = Number(row[k]); if (!Number.isFinite(row[k])) throw new Error('Invalid number in row ' + (i + 1)); } if (state.choices[k] && !state.choices[k].includes(row[k])) throw new Error('Invalid ' + k + ' in row ' + (i + 1)); });
        });
        state[kind] = rows; editor(kind); message('Imported ' + rows.length + ' rows.');
      } catch (e) { message(e.message); } finally { event.target.value = ''; }
    };
  }
  (async () => {
    try {
      const [defaults] = await Promise.all([api('defaults'), api('health')]);
      Object.assign(state, defaults); editor('fleet'); editor('routes'); state.ready = true;
      $('health').textContent = '● Model verified · Ready'; document.querySelectorAll('.run').forEach(b => b.disabled = false);
    } catch (e) { $('health').textContent = 'Model unavailable'; message(e.message + ' Reload to retry.'); }
  })();
}
