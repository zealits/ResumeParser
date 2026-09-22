/* Landing page: render plans and credit costs from the API. */

(async function () {
  const grid = document.getElementById('price-grid');
  const costBody = document.getElementById('cost-body');
  const warn = document.getElementById('pay-warning');

  // Already signed in? Point the nav at the right place.
  if (Auth.token) {
    const login = document.getElementById('nav-login');
    const cta = document.getElementById('nav-cta');
    const dest = Auth.isAdmin() ? '/superadmin/console' : '/dashboard';
    if (login) { login.textContent = 'Log out'; login.href = '#';
      login.addEventListener('click', function (e) { e.preventDefault(); Auth.logout('/'); }); }
    if (cta) { cta.textContent = Auth.isAdmin() ? 'Admin console' : 'Dashboard'; cta.href = dest; }
  }

  let data;
  try {
    data = await api('/api/billing/plans', { auth: false });
  } catch (e) {
    grid.innerHTML = '<div class="card" style="grid-column:1/-1">' +
      '<p style="margin:0">Could not load pricing: ' + esc(e.message) + '</p></div>';
    costBody.innerHTML = '<tr><td colspan="3" class="empty">Unavailable</td></tr>';
    return;
  }

  /* ----------------------------------------------------------- plans --- */
  grid.innerHTML = data.plans.map(function (p) {
    const free = p.price_cents === 0;
    const contact = !!p.contact_sales;
    const per = contact
      ? 'Custom volume'
      : p.per_credit
        ? '$' + p.per_credit.toFixed(3) + ' per credit'
        : 'Included';
    const creditsLabel = contact
      ? 'Custom credits'
      : fmt.int(p.credits) + ' credits';
    const href = contact
      ? 'mailto:' + encodeURIComponent(data.sales_email || '') +
        '?subject=' + encodeURIComponent('Enterprise plan inquiry')
      : '/signup?plan=' + encodeURIComponent(p.key);
    const cta = contact
      ? 'Contact sales'
      : (free ? 'Start free' : 'Choose ' + esc(p.name));
    return '' +
      '<div class="plan' + (p.popular ? ' popular' : '') + '">' +
        (p.popular ? '<span class="plan-badge">Most popular</span>' : '') +
        '<div class="plan-name">' + esc(p.name) + '</div>' +
        '<p class="plan-tag">' + esc(p.tagline || '') + '</p>' +
        '<div class="plan-price">' +
          '<span class="amt">' + esc(p.price_display) + '</span>' +
          '<span class="per">' + (contact ? '' : (free ? 'forever' : 'one-time')) + '</span>' +
        '</div>' +
        '<div class="plan-credits">' +
          '<b>' + esc(creditsLabel) + '</b>' +
          '<span>' + esc(per) + '</span>' +
        '</div>' +
        '<ul>' + (p.features || []).map(function (f) {
          return '<li>' + esc(f) + '</li>';
        }).join('') + '</ul>' +
        '<a class="btn ' + (p.popular ? 'btn-primary' : 'btn-ghost') + ' btn-block" ' +
          'href="' + href + '">' +
          cta +
        '</a>' +
      '</div>';
  }).join('');

  /* Payments not configured yet - say so rather than letting a card form fail. */
  if (!data.payments_enabled) {
    warn.className = 'alert alert-warn';
    warn.style.maxWidth = '760px';
    warn.style.margin = '0 auto 26px';
    warn.textContent = 'Card payments are not configured on this server yet, so ' +
      'paid plans cannot be purchased. Contact sales for Enterprise, or ask an admin ' +
      'to configure Square. ' +
      '(Set SQUARE_ACCESS_TOKEN, SQUARE_APPLICATION_ID and SQUARE_LOCATION_ID in .env.)';
  }

  /* ----------------------------------------------------------- costs --- */
  costBody.innerHTML = data.credit_costs.map(function (c) {
    return '<tr>' +
      '<td>' + esc(c.label) + '</td>' +
      '<td class="num"><span class="badge badge-info">' + c.credits + '</span></td>' +
      '<td class="num dim">' + fmt.int(Math.floor(1000 / c.credits)) + '</td>' +
      '</tr>';
  }).join('');

  /* ----------------------------------------------------- hero figures --- */
  document.getElementById('stat-endpoints').textContent = data.credit_costs.length;

  // Cheapest purchasable plan is the entry price.
  const entry = data.plans.find(function (p) {
    return !p.contact_sales && p.price_cents > 0;
  }) || data.plans[0];
  if (entry) {
    document.getElementById('stat-entry').textContent = entry.price_display;
  }

  const paid = data.plans.filter(function (p) { return p.per_credit; });
  if (paid.length) {
    const best = paid.reduce(function (a, b) { return a.per_credit < b.per_credit ? a : b; });
    document.getElementById('stat-cheapest').textContent = '$' + best.per_credit.toFixed(3);
  }
})();
