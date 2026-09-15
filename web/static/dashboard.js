/* Customer dashboard: balance, usage, ledger, and card top-ups. */

(function () {
  if (!Auth.require('/login')) return;

  const msg = document.getElementById('msg');
  let plans = [];
  let cfg = null;
  let costs = [];
  let balance = 0;
  let topupCard = null;
  let topupKey = idempotencyKey();

  /* --------------------------------------------------------- tabs ------ */
  document.querySelectorAll('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('on'); });
      tab.classList.add('on');
      const name = tab.dataset.tab;
      document.querySelectorAll('[data-panel]').forEach(function (p) {
        p.hidden = p.dataset.panel !== name;
      });
    });
  });

  document.getElementById('logout').addEventListener('click', function () {
    Auth.logout('/');
  });

  /* ------------------------------------------------------ chip/meter --- */
  function paintBalance(bal, planCredits) {
    balance = bal;
    const chip = document.getElementById('chip');
    const meter = document.getElementById('m-meter');
    document.getElementById('chip-bal').textContent = fmt.int(bal);
    document.getElementById('m-balance').textContent = fmt.int(bal);

    // Meter is balance against the size of the plan bundle - a rough "how much
    // of what I bought is left".
    const denom = Math.max(planCredits || 0, bal, 1);
    const pct = Math.max(0, Math.min(100, (bal / denom) * 100));
    meter.querySelector('i').style.width = pct + '%';

    chip.classList.remove('low', 'empty-bal');
    meter.classList.remove('low', 'out');
    if (bal <= 0) { chip.classList.add('empty-bal'); meter.classList.add('out'); }
    else if (pct < 20) { chip.classList.add('low'); meter.classList.add('low'); }

    document.getElementById('m-balance-sub').textContent = bal <= 0
      ? 'Out of credits — top up to keep calling'
      : 'of ' + fmt.int(denom) + ' in your bundle';

    renderCosts();
  }

  function renderCosts() {
    const body = document.getElementById('costs-body');
    if (!costs.length) {
      body.innerHTML = '<tr><td colspan="3" class="empty">No metered endpoints.</td></tr>';
      return;
    }
    body.innerHTML = costs.map(function (c) {
      const afford = Math.floor(balance / c.credits);
      return '<tr>' +
        '<td>' + esc(c.label) + '</td>' +
        '<td class="num"><span class="badge badge-info">' + c.credits + '</span></td>' +
        '<td class="num ' + (afford ? 'dim' : '') + '">' +
          (afford ? fmt.int(afford) + ' calls' : '<span class="badge badge-bad">0</span>') +
        '</td></tr>';
    }).join('');
  }

  /* ------------------------------------------------------------ load --- */
  async function load() {
    let me, tx, usage;
    try {
      const res = await Promise.all([
        api('/api/billing/me'),
        api('/api/billing/transactions?limit=100'),
        api('/auth/usage-stats?days=30'),
        api('/api/billing/plans', { auth: false }),
        api('/api/billing/config', { auth: false })
      ]);
      me = res[0]; tx = res[1]; usage = res[2]; plans = res[3].plans; cfg = res[4];
    } catch (e) {
      showAlert(msg, 'error', e.message);
      return;
    }

    costs = me.credit_costs || [];

    document.getElementById('hello').textContent =
      'Hello, ' + (me.username || 'there');
    document.getElementById('sub').textContent = me.email || '';

    const plan = plans.find(function (p) { return p.key === me.plan_key; });
    document.getElementById('m-plan').textContent =
      plan ? plan.name : fmt.title(me.subscription_tier);
    document.getElementById('m-tier').textContent =
      plan ? plan.price_display + ' · ' + fmt.int(plan.credits) + ' credits' : '—';
    document.getElementById('m-used').textContent = fmt.int(me.credits_used);

    paintBalance(me.credits_balance, plan ? plan.credits : null);

    /* usage */
    document.getElementById('m-calls').textContent = fmt.int(usage.total_api_calls);
    document.getElementById('m-success').textContent =
      usage.total_api_calls ? usage.success_rate + '% success' : 'No calls yet';

    const days = (usage.daily_breakdown || []).slice().reverse();
    document.getElementById('usage-body').innerHTML = days.length
      ? days.map(function (d) {
          const calls = d.api_calls || 0;
          const avg = calls ? (d.total_processing_time || 0) / calls : 0;
          return '<tr>' +
            '<td>' + esc(fmt.dateShort(d.date)) + '</td>' +
            '<td class="num">' + fmt.int(calls) + '</td>' +
            '<td class="num dim">' + avg.toFixed(2) + 's</td>' +
          '</tr>';
        }).join('')
      : '<tr><td colspan="3" class="empty">No calls in the last 30 days.</td></tr>';

    /* account table */
    document.getElementById('account-body').innerHTML = [
      ['Username', esc(me.username)],
      ['Email', esc(me.email)],
      ['Plan', esc(plan ? plan.name : fmt.title(me.subscription_tier))],
      ['Role', '<span class="badge badge-neutral">' + esc(fmt.title(me.role)) + '</span>'],
      ['Credits remaining', fmt.int(me.credits_balance)],
      ['Credits spent', fmt.int(me.credits_used)]
    ].map(function (r) {
      return '<tr><td class="muted">' + r[0] + '</td><td>' + r[1] + '</td></tr>';
    }).join('');

    renderLedger(tx.transactions || []);
    renderToken();
    setupTopup();
  }

  /* ---------------------------------------------------------- ledger --- */
  const TX_BADGE = {
    purchase: 'badge-ok', topup: 'badge-ok', signup_grant: 'badge-info',
    signup_purchase: 'badge-ok', admin_grant: 'badge-info',
    debit: 'badge-neutral', refund: 'badge-warn'
  };

  function renderLedger(rows) {
    const body = document.getElementById('ledger-body');
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty">Nothing yet.</td></tr>';
      return;
    }
    body.innerHTML = rows.map(function (r) {
      const credits = Number(r.credits || 0);
      const detail = r.operation
        ? fmt.title(r.operation)
        : (r.plan_key ? fmt.title(r.plan_key) + ' plan' : (r.note || '—'));
      const amount = r.amount_cents ? ' · ' + fmt.money(r.amount_cents) : '';
      return '<tr>' +
        '<td class="nowrap dim">' + esc(fmt.date(r.created_at)) + '</td>' +
        '<td><span class="badge ' + (TX_BADGE[r.type] || 'badge-neutral') + '">' +
          esc(fmt.title(r.type)) + '</span></td>' +
        '<td>' + esc(detail) + '<span class="muted">' + esc(amount) + '</span></td>' +
        '<td class="num" style="color:' + (credits >= 0 ? '#6ee7b7' : '#fda4b4') + '">' +
          (credits >= 0 ? '+' : '') + fmt.int(credits) + '</td>' +
        '<td class="num dim">' +
          (r.balance_after === null || r.balance_after === undefined
            ? '—' : fmt.int(r.balance_after)) + '</td>' +
      '</tr>';
    }).join('');
  }

  /* ----------------------------------------------------------- token --- */
  function renderToken() {
    const token = Auth.token || '';
    document.getElementById('token-box').value = token;
    const shown = token ? token.slice(0, 12) + '…' : '$TOKEN';
    document.getElementById('curl-example').innerHTML =
      '<span class="tok-cmd">curl</span> -X POST ' +
      esc(location.origin) + '/parse-resume \\\n' +
      '  -H <span class="tok-str">"Authorization: Bearer ' + esc(shown) + '"</span> \\\n' +
      '  -F <span class="tok-str">"file=@resume.pdf"</span>';

    document.getElementById('copy-token').addEventListener('click', function () {
      const box = document.getElementById('token-box');
      box.select();
      navigator.clipboard.writeText(box.value).then(function () {
        const b = document.getElementById('copy-token');
        b.textContent = 'Copied';
        setTimeout(function () { b.textContent = 'Copy token'; }, 1600);
      }).catch(function () { /* selection is the fallback */ });
    });
  }

  /* ----------------------------------------------------------- topup --- */
  function setupTopup() {
    const modal = document.getElementById('topup-modal');
    const select = document.getElementById('topup-plan');
    const total = document.getElementById('topup-total');
    const tmsg = document.getElementById('topup-msg');
    const payBtn = document.getElementById('topup-pay');
    const hint = document.getElementById('topup-card-hint');

    const paid = plans.filter(function (p) { return p.price_cents > 0; });

    select.innerHTML = paid.map(function (p) {
      return '<option value="' + esc(p.key) + '">' +
        esc(p.name) + ' — ' + fmt.int(p.credits) + ' credits · ' +
        esc(p.price_display) + '</option>';
    }).join('');

    function selected() {
      return paid.find(function (p) { return p.key === select.value; });
    }
    function paintTotal() {
      const p = selected();
      total.textContent = p ? p.price_display : '—';
    }
    select.addEventListener('change', paintTotal);
    paintTotal();

    document.getElementById('topup-btn').addEventListener('click', async function () {
      showAlert(tmsg, null, null);
      modal.classList.remove('hide');

      if (!paid.length) {
        showAlert(tmsg, 'warn', 'No paid bundles are configured.');
        payBtn.disabled = true;
        return;
      }
      if (!cfg.square.enabled) {
        showAlert(tmsg, 'warn',
          'Card payments are not configured on this server. Ask an admin to ' +
          'set the Square credentials, or to grant credits manually.');
        document.getElementById('topup-card-field').hidden = true;
        payBtn.disabled = true;
        return;
      }
      // Mount once, then reuse across opens.
      if (!topupCard) {
        try {
          topupCard = await mountSquareCard(cfg.square, 'topup-card');
          hint.textContent = cfg.square.environment === 'sandbox'
            ? 'Sandbox — test card 4111 1111 1111 1111, any future expiry, CVV 111.'
            : 'Secured by Square.';
        } catch (e) {
          showAlert(tmsg, 'error', e.message);
          payBtn.disabled = true;
        }
      }
    });

    function close() { modal.classList.add('hide'); }
    document.getElementById('topup-cancel').addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });

    payBtn.addEventListener('click', async function () {
      const plan = selected();
      if (!plan || !topupCard) return;
      showAlert(tmsg, null, null);

      const done = busy(payBtn, 'Charging…');
      try {
        const token = await tokenizeCard(topupCard);
        const res = await api('/api/billing/topup', {
          method: 'POST',
          body: { plan_key: plan.key, source_id: token, idempotency_key: topupKey }
        });
        close();
        done();
        showAlert(msg, 'ok', res.message +
          (res.receipt_url ? ' Receipt: ' + res.receipt_url : ''));
        topupKey = idempotencyKey();
        await load();
      } catch (err) {
        done();
        showAlert(tmsg, 'error', err.message);
        // Fresh key + fresh card entry for the next attempt.
        topupKey = idempotencyKey();
        hint.textContent = 'Re-enter the card details to try again.';
      }
    });
  }

  if (qs('welcome')) {
    showAlert(msg, 'ok', 'Account created and credits added. Welcome aboard.');
  }

  load();
})();
