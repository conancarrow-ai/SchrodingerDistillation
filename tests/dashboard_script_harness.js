const fs = require('fs');
const script = fs.readFileSync(process.argv[2], 'utf8');
const TABS = ['runs', 'data', 'queue', 'benchmark'];

function storage(backing) {
  return { getItem: k => (k in backing ? backing[k] : null), setItem: (k, v) => { backing[k] = String(v); }, removeItem: k => { delete backing[k]; } };
}
function element(env, props) {
  return Object.assign({
    listeners: {}, attributes: {}, className: '',
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    fire(type) { (this.listeners[type] || []).forEach(fn => fn.call(this)); },
    scrollIntoView() { env.scrolledIntoView = this.id; },
    getAttribute(name) { return this.attributes[name]; },
  }, props);
}
function makeEnv(opts) {
  const env = { localBacking: opts.localBacking || {}, sessionBacking: opts.sessionBacking || {}, timeouts: [], scrolledTo: null, scrolledIntoView: null, replacedHash: null, reloaded: false };
  env.localStorage = storage(env.localBacking);
  env.sessionStorage = storage(env.sessionBacking);
  env.details = opts.detailIds.map(id => element(env, { id, open: opts.serverOpen.includes(id) }));
  env.buttons = TABS.map(name => element(env, { attributes: { 'data-tab': name } }));
  env.panels = TABS.map(name => element(env, { id: 'tab-' + name, className: 'tab' }));
  const byId = {};
  env.details.forEach(d => { byId[d.id] = d; }); env.panels.forEach(p => { byId[p.id] = p; });
  env.document = {
    querySelectorAll(sel) { if (sel === 'details[id]') return env.details; if (sel === 'nav.tabs button') return env.buttons; if (sel === 'section.tab') return env.panels; return []; },
    getElementById(id) { return byId[id] || null; },
  };
  env.window = { setTimeout(fn, ms) { env.timeouts.push({ fn, ms }); }, location: { hash: opts.hash || '', reload() { env.reloaded = true; } },
                 pageYOffset: opts.pageYOffset || 0, scrollTo(x, y) { env.scrolledTo = y; }, addEventListener() {} };
  env.history = { replaceState(a, b, url) { env.replacedHash = url; }, scrollRestoration: 'auto' };
  return env;
}
function run(env, reloadMs) {
  new Function('RELOAD_MS', 'document', 'window', 'localStorage', 'sessionStorage', 'history', script)(reloadMs, env.document, env.window, env.localStorage, env.sessionStorage, env.history);
}
function assert(condition, message) { if (!condition) { console.error('FAIL: ' + message); process.exitCode = 1; } else { console.log('ok: ' + message); } }
const ids = ['run-a', 'run-b'];
const local = {}, session = {};

// first visit: the server opens the running run-a; the user opens run-b, closes run-a, scrolls, and the page reloads itself
let env = makeEnv({ detailIds: ids, serverOpen: ['run-a'], localBacking: local, sessionBacking: session, pageYOffset: 1234 });
run(env, 120000);
assert(env.history.scrollRestoration === 'manual', 'browser scroll restoration switched off');
assert(env.panels[0].className === 'tab active', 'runs tab active by default');
assert(env.details[0].open === true && env.details[1].open === false, 'server defaults kept on the first visit');
env.details[1].open = true; env.details[1].fire('toggle');
env.details[0].open = false; env.details[0].fire('toggle');
assert(JSON.parse(local.sparse_dsb_dashboard_open)['run-b'] === true && JSON.parse(local.sparse_dsb_dashboard_open)['run-a'] === false, 'toggles stored');
assert(env.timeouts.length === 1 && env.timeouts[0].ms === 120000, 'one reload timer');
env.timeouts[0].fn();
assert(env.reloaded && session.sparse_dsb_dashboard_scroll === '1234', 'scroll position saved before the reload');

// the reload: the server again opens run-a, but the stored choices and the scroll position win
env = makeEnv({ detailIds: ids, serverOpen: ['run-a'], localBacking: local, sessionBacking: session });
run(env, 120000);
assert(env.details[0].open === false && env.details[1].open === true, 'open sections restored after the reload');
assert(env.scrolledTo === 1234 && !('sparse_dsb_dashboard_scroll' in session), 'scroll position restored once');

// a run link: opens that run, shows the runs tab, drops the hash, records the choice, no scroll restoration
env = makeEnv({ detailIds: ids, serverOpen: [], localBacking: {}, sessionBacking: { sparse_dsb_dashboard_scroll: '500' }, hash: '#run-b' });
run(env, 120000);
assert(env.details[1].open === true && env.scrolledIntoView === 'run-b' && env.panels[0].className === 'tab active', 'run link opens and shows the run');
assert(env.replacedHash === '#runs' && env.scrolledTo === null, 'hash dropped, no scroll restoration on a jump');
assert(JSON.parse(env.localBacking.sparse_dsb_dashboard_open)['run-b'] === true, 'run opened by link is recorded');

// a new run appears later: no stored entry, so the server default applies; tab click persists
env = makeEnv({ detailIds: ['run-a', 'run-b', 'run-c'], serverOpen: ['run-c'], localBacking: local, sessionBacking: {} });
run(env, 0);
assert(env.details[2].open === true && env.timeouts.length === 0, 'new run keeps the server default; no timer when RELOAD_MS is 0');
env.buttons[1].fire('click');
assert(env.panels[1].className === 'tab active' && local.sparse_dsb_dashboard_tab === 'data' && env.replacedHash === '#data', 'tab click persisted');
