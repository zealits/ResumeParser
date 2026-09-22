/* Signup + checkout.

   Paid plans mount Square's card field; the card is tokenized in Square's
   iframe and only the resulting single-use token is POSTed to our server. */

(async function () {
  const msg = document.getElementById('msg');
  const btn = document.getElementById('submit');
  const form = document.getElementById('signup-form');
  const cardBlock = document.getElementById('card-block');
  const cardHint = document.getElementById('card-hint');
  const summary = document.getElementById('summary');

  let plan = null;
  let card = null;        // Square card instance, paid plans only
  let cardReady = false;
  // One key per page load so a double-submit cannot double-charge.
  let attemptKey = idempotencyKey();

  /* ------------------------------------------------------- load plan --- */
  let data, cfg;
  try {
    const both = await Promise.all([
      api('/api/billing/plans', { auth: false }),
      api('/api/billing/config', { auth: false })
    ]);
    data = both[0];
    cfg = both[1];
  } catch (e) {
    showAlert(msg, 'error', 'Could not load plans: ' + e.message);
    document.getElementById('plan-lede').textContent = '';
    return;
  }

  // Falls back to the cheapest purchasable plan; contact-sales plans are last.
  const fallback = (data.plans.find(function (p) {
    return !p.contact_sales;
  }) || data.plans[0] || {}).key || '';
  const wanted = (qs('plan') || fallback).toLowerCase();
  plan = data.plans.find(function (p) { return p.key === wanted; });
  if (!plan) {
    showAlert(msg, 'error', 'That plan does not exist. Pick one from the pricing page.');
    document.getElementById('plan-lede').innerHTML =
      '<a href="/#pricing">Back to pricing</a>';
    return;
  }

  if (plan.contact_sales) {
    const mail = data.sales_email
      ? 'mailto:' + encodeURIComponent(data.sales_email) +
        '?subject=' + encodeURIComponent('Enterprise plan inquiry')
      : '/#pricing';
    showAlert(msg, 'warn',
      'Enterprise is sold through sales. Email us for a custom quote.');
    document.getElementById('plan-lede').innerHTML =
      '<a href="' + mail + '">Contact sales</a> · <a href="/#pricing">Back to pricing</a>';
    summary.hidden = true;
    cardBlock.hidden = true;
    btn.disabled = true;
    btn.textContent = 'Contact sales';
    return;
  }

  const isFree = plan.price_cents === 0;

  document.getElementById('plan-lede').textContent = isFree
    ? 'Start with ' + fmt.int(plan.credits) + ' free credits. No card required.'
    : 'You are buying the ' + plan.name + ' plan — ' +
      fmt.int(plan.credits) + ' credits for ' + plan.price_display + '.';

  summary.hidden = false;
  document.getElementById('sum-plan').textContent = plan.name;
  document.getElementById('sum-credits').textContent = fmt.int(plan.credits) + ' credits';
  document.getElementById('sum-total').textContent = isFree ? 'Free' : plan.price_display;

  btn.textContent = isFree
    ? 'Create free account'
    : 'Pay ' + plan.price_display + ' and create account';

  /* ------------------------------------------------------ square card --- */
  if (isFree) {
    btn.disabled = false;
  } else if (!cfg.square.enabled) {
    showAlert(msg, 'warn',
      'Card payments are not configured on this server, so this plan cannot be ' +
      'purchased yet. You can still start on the free plan.');
    cardBlock.hidden = true;
    btn.disabled = true;
  } else {
    cardBlock.hidden = false;
    try {
      card = await mountSquareCard(cfg.square, 'card-container');
      cardReady = true;
      cardHint.textContent = cfg.square.environment === 'sandbox'
        ? 'Sandbox mode — use Square test card 4111 1111 1111 1111, any future expiry, CVV 111.'
        : 'Secured by Square.';
      btn.disabled = false;
    } catch (e) {
      showAlert(msg, 'error', e.message);
      cardHint.textContent = 'Card field unavailable.';
    }
  }

  /* ---------------------------------------------------------- submit --- */
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    showAlert(msg, null, null);

    const payload = {
      username: document.getElementById('username').value.trim(),
      email: document.getElementById('email').value.trim(),
      password: document.getElementById('password').value,
      company_name: document.getElementById('company').value.trim() || null,
      contact_person: document.getElementById('contact').value.trim() || null,
      plan_key: plan.key,
      idempotency_key: attemptKey
    };

    if (payload.username.length < 3) {
      showAlert(msg, 'error', 'Username must be at least 3 characters.'); return;
    }
    if (!payload.email) { showAlert(msg, 'error', 'Enter your email address.'); return; }
    if (payload.password.length < 8) {
      showAlert(msg, 'error', 'Password must be at least 8 characters.'); return;
    }

    const done = busy(btn, isFree ? 'Creating account…' : 'Processing payment…');

    try {
      if (!isFree) {
        if (!cardReady) throw new Error('The card field is not ready yet.');
        payload.source_id = await tokenizeCard(card);
      }

      const result = await api('/api/billing/signup', {
        method: 'POST',
        auth: false,
        body: payload
      });

      Auth.save(result.auth);
      location.replace('/dashboard?welcome=1');
    } catch (err) {
      done();
      showAlert(msg, 'error', err.message);
      // A consumed token cannot be reused, and the next attempt is a new
      // charge, so both need to be fresh.
      attemptKey = idempotencyKey();
      if (!isFree && cardReady) {
        cardHint.textContent = 'Re-enter the card details to try again.';
      }
    }
  });
})();
