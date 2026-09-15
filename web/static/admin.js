/* Superadmin console: users, revenue, credit grants, account actions. */

(function () {
  // adminOnly: the cached role gates the UI; every endpoint below is also
  // guarded server-side by get_current_admin.
  if (!Auth.require('/superadmin', true)) return;

  const msg = document.getElementById('msg');
  const COST_ENV = {
    parse_resume: 'CREDIT_COST_PARSE_RESUME',
    parse_resume_text: 'CREDIT_COST_PARSE_RESUME_TEXT',
    github_analysis: 'CREDIT_COST_GITHUB_ANALYSIS',
    ranked_candidates: 'CREDIT_COST_RANKED_CANDIDATES',
    generate_test: 'CREDIT_COST_GENERATE_TEST',
    evaluate_test: 'CREDIT_COST_EVALUATE_TEST'
  };

  let users = [];
  let grantTarget = null;
  let confirmAction = null;

  const me = Auth.user || {};
  document.getElementById('who').textContent =
    (me.username || 'admin') + ' · ' + (me.email || '');

  document.getElementById('logout').addEventListener('click', function () {
    Auth.logout('/superadmin');
  });
  document.getElementById('refresh').addEventListener('click', function () { load(); });

  document.querySelectorAll('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('on'); });
      tab.classList.add('on');
      document.querySelectorAll('[data-panel]').forEach(function (p) {
        p.hidden = p.dataset.panel !== tab.dataset.tab;
      });
    });
  });

  /* ------------------------------------------------------------ load --- */
  async function load() {
    let list, analytics, revenue, ledger, costs;
    try {
      const res = await Promise.all([
        api('/auth/admin/users?limit=500', { loginPath: '/superadmin?expired=1' }),
        api('/auth/admin/analytics?days=30', { loginPath: '/superadmin?expired=1' }),
        api('/api/billing/admin/revenue?days=30', { loginPath: '/superadmin?expired=1' }),
        api('/api/billing/admin/transactions?limit=100', { loginPath: '/superadmin?expired=1' }),
        api('/api/billing/admin/credit-costs', { loginPath: '/superadmin?expired=1' })
      ]);
      list = res[0]; analytics = res[1]; revenue = res[2]; ledger = res[3]; costs = res[4];
    } catch (e) {
      showAlert(msg, 'error', e.message);
      return;
    }

    users = list.users || [];

    document.getElementById('m-users').textContent = fmt.int(analytics.total_users);
    document.getElementById('m-active').textContent =
      fmt.int(analytics.active_users) + ' active';
    document.getElementById('m-revenue').textContent = revenue.gross_display;
    document.getElementById('m-payments').textContent =
      fmt.int(revenue.payments) + ' payment' + (revenue.payments === 1 ? '' : 's');
    document.getElementById('m-sold').textContent = fmt.int(revenue.credits_sold);
    document.getElementById('m-calls').textContent = fmt.int(analytics.total_api_calls);
    document.getElementById('m-avg').textContent =
      (analytics.average_response_time || 0).toFixed(2) + 's average';

    renderUsers();
    renderLedger(ledger.transactions || []);

    document.getElementById('costs-body').innerHTML =
      (costs.credit_costs || []).map(function (c) {
        return '<tr>' +
          '<td>' + esc(c.label) + '</td>' +
          '<td class="mono muted" style="font-size:.8rem">' +
            esc(COST_ENV[c.key] || '—') + '</td>' +
          '<td class="num"><span class="badge badge-info">' + c.credits + '</span></td>' +
        '</tr>';
      }).join('');
  }

  /* ----------------------------------------------------------- users --- */
  const STATUS_BADGE = { active: 'badge-ok', suspended: 'badge-bad', inactive: 'badge-neutral' };

  function renderUsers() {
    const q = document.getElementById('search').value.trim().toLowerCase();
    const body = document.getElementById('users-body');

    const rows = users.filter(function (u) {
      if (!q) return true;
      return [u.username, u.email, u.plan_key, u.subscription_tier, u.company_name]
        .some(function (v) { return v && String(v).toLowerCase().includes(q); });
    });

    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="8" class="empty">' +
        (q ? 'No accounts match that filter.' : 'No accounts yet.') + '</td></tr>';
      return;
    }

    body.innerHTML = rows.map(function (u) {
      const status = (u.status || 'active').toLowerCase();
      const isAdmin = u.role === 'admin';
      const isSelf = u.id === me.id;
      return '<tr>' +
        '<td><div style="font-weight:600">' + esc(u.username) + '</div>' +
          '<div class="muted" style="font-size:.82rem">' + esc(u.email) + '</div></td>' +
        '<td>' + esc(fmt.title(u.plan_key || u.subscription_tier)) + '</td>' +
        '<td>' + (isAdmin
          ? '<span class="badge badge-warn">Admin</span>'
          : '<span class="badge badge-neutral">User</span>') + '</td>' +
        '<td><span class="badge ' + (STATUS_BADGE[status] || 'badge-neutral') + '">' +
          esc(fmt.title(status)) + '</span></td>' +
        '<td class="num" style="font-weight:600">' + fmt.int(u.credits_balance || 0) + '</td>' +
        '<td class="num dim">' + fmt.int(u.credits_used || 0) + '</td>' +
        '<td class="nowrap dim" style="font-size:.85rem">' +
          esc(fmt.dateShort(u.created_at)) + '</td>' +
        '<td><div style="display:flex;gap:5px;flex-wrap:wrap">' +
          '<button class="btn btn-ghost btn-sm" data-act="grant" data-id="' +
            esc(u.id) + '">+ Credits</button>' +
          (status === 'suspended'
            ? '<button class="btn btn-ghost btn-sm" data-act="activate" data-id="' +
                esc(u.id) + '">Activate</button>'
            : '<button class="btn btn-ghost btn-sm" data-act="suspend" data-id="' +
                esc(u.id) + '"' + (isSelf ? ' disabled title="You cannot suspend yourself"' : '') +
                '>Suspend</button>') +
          '<button class="btn btn-danger btn-sm" data-act="delete" data-id="' +
            esc(u.id) + '"' + (isSelf ? ' disabled title="You cannot delete yourself"' : '') +
            '>Delete</button>' +
        '</div></td>' +
      '</tr>';
    }).join('');
  }

  document.getElementById('search').addEventListener('input', renderUsers);

  /* Delegated so re-rendering the table keeps the handlers. */
  document.getElementById('users-body').addEventListener('click', function (e) {
    const btn = e.target.closest('button[data-act]');
    if (!btn || btn.disabled) return;
    const user = users.find(function (u) { return u.id === btn.dataset.id; });
    if (!user) return;

    switch (btn.dataset.act) {
      case 'grant': openGrant(user); break;
      case 'suspend':
        ask('Suspend ' + user.username + '?',
            'They will be locked out immediately. Credits are untouched and you can reactivate later.',
            'Suspend',
            function () { return userAction(user, 'PUT', '/auth/admin/users/' + user.id + '/suspend', 'suspended'); });
        break;
      case 'activate':
        userAction(user, 'PUT', '/auth/admin/users/' + user.id + '/activate', 'activated');
        break;
      case 'delete':
        ask('Delete ' + user.username + '?',
            'This permanently removes the account and its remaining ' +
            fmt.int(user.credits_balance || 0) + ' credits. This cannot be undone.',
            'Delete permanently',
            function () { return userAction(user, 'DELETE', '/auth/admin/users/' + user.id, 'deleted'); });
        break;
    }
  });

  async function userAction(user, method, path, verb) {
    try {
      await api(path, { method: method, loginPath: '/superadmin?expired=1' });
      showAlert(msg, 'ok', user.username + ' ' + verb + '.');
      await load();
    } catch (e) {
      showAlert(msg, 'error', e.message);
    }
  }

  /* ---------------------------------------------------------- confirm -- */
  const confirmModal = document.getElementById('confirm-modal');

  function ask(title, text, label, action) {
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-text').textContent = text;
    document.getElementById('confirm-yes').textContent = label;
    confirmAction = action;
    confirmModal.classList.remove('hide');
  }
  function closeConfirm() {
    confirmModal.classList.add('hide');
    confirmAction = null;
  }
  document.getElementById('confirm-no').addEventListener('click', closeConfirm);
  confirmModal.addEventListener('click', function (e) {
    if (e.target === confirmModal) closeConfirm();
  });
  document.getElementById('confirm-yes').addEventListener('click', async function () {
    const action = confirmAction;
    const done = busy(this, 'Working…');
    closeConfirm();
    if (action) await action();
    done();
  });

  /* ------------------------------------------------------------ grant -- */
  const grantModal = document.getElementById('grant-modal');

  function openGrant(user) {
    grantTarget = user;
    document.getElementById('grant-who').textContent =
      user.username + ' (' + user.email + ') · balance ' +
      fmt.int(user.credits_balance || 0);
    showAlert(document.getElementById('grant-msg'), null, null);
    document.getElementById('grant-note').value = '';
    grantModal.classList.remove('hide');
  }
  function closeGrant() { grantModal.classList.add('hide'); grantTarget = null; }

  document.getElementById('grant-cancel').addEventListener('click', closeGrant);
  grantModal.addEventListener('click', function (e) {
    if (e.target === grantModal) closeGrant();
  });

  document.getElementById('grant-save').addEventListener('click', async function () {
    if (!grantTarget) return;
    const gmsg = document.getElementById('grant-msg');
    const amount = parseInt(document.getElementById('grant-amount').value, 10);
    if (!amount || amount < 1) {
      showAlert(gmsg, 'error', 'Enter a credit amount of 1 or more.');
      return;
    }

    const done = busy(this, 'Granting…');
    try {
      const res = await api('/api/billing/admin/users/' + grantTarget.id + '/credits', {
        method: 'POST',
        body: { credits: amount, note: document.getElementById('grant-note').value || null },
        loginPath: '/superadmin?expired=1'
      });
      closeGrant();
      done();
      showAlert(msg, 'ok', res.message);
      await load();
    } catch (e) {
      done();
      showAlert(gmsg, 'error', e.message);
    }
  });

  /* ----------------------------------------------------------- ledger -- */
  const TX_BADGE = {
    purchase: 'badge-ok', topup: 'badge-ok', signup_grant: 'badge-info',
    signup_purchase: 'badge-ok', admin_grant: 'badge-info',
    debit: 'badge-neutral', refund: 'badge-warn'
  };

  function renderLedger(rows) {
    const body = document.getElementById('ledger-body');
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6" class="empty">No credit activity yet.</td></tr>';
      return;
    }
    body.innerHTML = rows.map(function (r) {
      const credits = Number(r.credits || 0);
      const detail = r.operation
        ? fmt.title(r.operation)
        : (r.plan_key ? fmt.title(r.plan_key) + ' plan' : (r.note || '—'));
      return '<tr>' +
        '<td class="nowrap dim" style="font-size:.85rem">' +
          esc(fmt.date(r.created_at)) + '</td>' +
        '<td>' + esc(r.username || '—') + '</td>' +
        '<td><span class="badge ' + (TX_BADGE[r.type] || 'badge-neutral') + '">' +
          esc(fmt.title(r.type)) + '</span></td>' +
        '<td>' + esc(detail) + '</td>' +
        '<td class="num" style="color:' + (credits >= 0 ? '#6ee7b7' : '#fda4b4') + '">' +
          (credits >= 0 ? '+' : '') + fmt.int(credits) + '</td>' +
        '<td class="num dim">' + (r.amount_cents ? fmt.money(r.amount_cents) : '—') + '</td>' +
      '</tr>';
    }).join('');
  }

  load();
})();
