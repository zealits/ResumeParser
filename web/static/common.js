/* Shared helpers: token storage, API calls, formatting, Square SDK loading. */

const TOKEN_KEY = 'rp_token';
const USER_KEY = 'rp_user';

/* ------------------------------------------------------------------ auth -- */
const Auth = {
  get token() {
    try { return localStorage.getItem(TOKEN_KEY); } catch (e) { return null; }
  },
  get user() {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); }
    catch (e) { return null; }
  },
  save(auth) {
    try {
      localStorage.setItem(TOKEN_KEY, auth.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(auth.user || {}));
    } catch (e) { /* private mode - session stays in memory only */ }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch (e) { /* ignore */ }
  },
  isAdmin() {
    const u = Auth.user;
    return !!u && u.role === 'admin';
  },
  /* Bounce to a login page unless a token is present. `adminOnly` also checks
     the cached role - the real gate is server-side on every admin endpoint. */
  require(loginPath, adminOnly) {
    if (!Auth.token) { location.replace(loginPath); return false; }
    if (adminOnly && !Auth.isAdmin()) {
      Auth.clear();
      location.replace(loginPath + '?denied=1');
      return false;
    }
    return true;
  },
  logout(to) {
    Auth.clear();
    location.replace(to || '/');
  }
};

/* ------------------------------------------------------------------- api -- */
class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

/* Pull a readable message out of FastAPI's many error shapes. */
function errorMessage(payload, fallback) {
  const d = payload && payload.detail !== undefined ? payload.detail : payload;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    // Pydantic validation errors
    const first = d[0];
    if (first && first.msg) {
      const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : '';
      return field ? field + ': ' + first.msg : first.msg;
    }
  }
  if (d && typeof d === 'object') return d.message || d.error || fallback;
  return fallback;
}

async function api(path, options) {
  const opts = options || {};
  const headers = Object.assign({}, opts.headers || {});
  if (opts.body !== undefined && !(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  if (Auth.token && opts.auth !== false) {
    headers['Authorization'] = 'Bearer ' + Auth.token;
  }

  let res;
  try {
    res = await fetch(path, {
      method: opts.method || 'GET',
      headers: headers,
      body: opts.body !== undefined && !(opts.body instanceof FormData)
        ? JSON.stringify(opts.body)
        : opts.body
    });
  } catch (e) {
    throw new ApiError('Could not reach the server. Is the API running?', 0, null);
  }

  let payload = null;
  const text = await res.text();
  if (text) { try { payload = JSON.parse(text); } catch (e) { payload = { detail: text }; } }

  if (!res.ok) {
    // An expired or revoked token should not leave a half-logged-in page.
    if (res.status === 401 && opts.auth !== false) {
      Auth.clear();
      if (opts.redirectOn401 !== false) {
        location.replace(opts.loginPath || '/login?expired=1');
      }
    }
    throw new ApiError(
      errorMessage(payload, 'Request failed (' + res.status + ')'),
      res.status,
      payload
    );
  }
  return payload;
}

/* ------------------------------------------------------------------- fmt -- */
const fmt = {
  int(n) { return Number(n || 0).toLocaleString(); },
  money(cents) {
    if (cents === null || cents === undefined) return '—';
    return '$' + (Number(cents) / 100).toLocaleString(undefined, {
      minimumFractionDigits: 2, maximumFractionDigits: 2
    });
  },
  date(v) {
    if (!v) return '—';
    const d = new Date(v);
    if (isNaN(d)) return '—';
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  },
  dateShort(v) {
    if (!v) return '—';
    const d = new Date(v);
    if (isNaN(d)) return '—';
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  },
  title(s) {
    if (!s) return '—';
    return String(s).replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
};

/* Escape before interpolating anything server- or user-supplied into HTML. */
function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/* ------------------------------------------------------------------- ui --- */
function showAlert(el, kind, message) {
  if (!el) return;
  if (!message) { el.className = 'hide'; el.textContent = ''; return; }
  el.className = 'alert alert-' + kind;
  el.textContent = message;
  el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* Put a button into a spinner state and hand back a restore function. */
function busy(btn, label) {
  if (!btn) return function () {};
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>' + esc(label || 'Working…');
  return function () { btn.disabled = false; btn.innerHTML = original; };
}

function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

/* ---------------------------------------------------------------- square -- */
/* The Web Payments SDK must come from Square's own CDN: it renders the card
   fields inside an iframe on Square's origin, which is exactly why raw card
   numbers never reach our server. */
function squareSdkUrl(environment) {
  return environment === 'production'
    ? 'https://web.squarecdn.com/v1/square.js'
    : 'https://sandbox.web.squarecdn.com/v1/square.js';
}

function loadSquareSdk(environment) {
  return new Promise(function (resolve, reject) {
    if (window.Square) { resolve(window.Square); return; }
    const s = document.createElement('script');
    s.src = squareSdkUrl(environment);
    s.onload = function () {
      window.Square ? resolve(window.Square)
                    : reject(new Error('Square SDK loaded but unavailable.'));
    };
    s.onerror = function () {
      reject(new Error('Could not load the Square payment SDK. Check your connection.'));
    };
    document.head.appendChild(s);
  });
}

/* Mount Square's card field into `containerId`. Returns the card object whose
   tokenize() yields the single-use token we send to our server. */
async function mountSquareCard(cfg, containerId) {
  const Square = await loadSquareSdk(cfg.environment);
  const payments = Square.payments(cfg.application_id, cfg.location_id);
  const card = await payments.card({
    style: {
      input: { color: '#e9eef9', fontSize: '15px' },
      '.input-container': { borderColor: '#27334b', borderRadius: '9px' },
      '.input-container.is-focus': { borderColor: '#6366f1' },
      '.input-container.is-error': { borderColor: '#f43f5e' },
      '.message-text': { color: '#a7b6d1' },
      '.message-text.is-error': { color: '#fda4b4' }
    }
  });
  await card.attach('#' + containerId);
  return card;
}

/* Tokenize, converting Square's error shape into a plain message. */
async function tokenizeCard(card) {
  const result = await card.tokenize();
  if (result.status === 'OK') return result.token;
  const errs = result.errors || [];
  throw new Error(errs.length ? (errs[0].message || errs[0].code) : 'Could not read those card details.');
}

/* A stable key per checkout attempt, so a double-click or a retried request
   cannot charge the card twice. */
function idempotencyKey() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'k-' + Date.now() + '-' + Math.random().toString(16).slice(2);
}
