# Web UI, credit billing, and Square payments

Everything runs on the **same port as the API** (default `2010`). No separate
frontend server, no build step, no templating engine.

## Pages

| URL | Who | What |
|---|---|---|
| `/` | public | Landing page with the pricing section |
| `/signup?plan=<key>` | public | Self-serve signup + card checkout |
| `/login` | customers | Login; admins are redirected to the console |
| `/dashboard` | customers | Credits, usage, ledger, top-ups, API token |
| `/superadmin` | admins | Admin sign-in (amber theme, `noindex`) |
| `/superadmin/console` | admins | Users, revenue, credit grants, account actions |
| `/docs` | — | Existing Swagger UI, untouched |

`GET /` used to return the JSON banner `{"message": "Resume Parser API is
running"}`. That moved to **`/api/status`**. `/health` is unchanged, so existing
health checks keep working.

## How a customer gets access

1. Lands on `/`, picks a plan from the pricing section.
2. `/signup?plan=basic` collects account details and a card.
3. The card is tokenized **inside Square's iframe** — the browser sends a
   single-use `source_id` token to us, never a card number.
4. `POST /api/billing/signup` charges Square **first**. Only on a completed
   payment is the account created and credits granted, so a declined card never
   leaves a half-made user behind.
5. The response includes a JWT, so the browser lands on `/dashboard` already
   logged in with `role: user`.

The free plan skips steps 3–4 and grants its credits directly.

## Credits

Credits are the meter. Each metered call costs a fixed amount, set in `.env`:

| Operation | Credits | Env var |
|---|---|---|
| Parse resume (PDF) | 1 | `CREDIT_COST_PARSE_RESUME` |
| Parse resume (text) | 1 | `CREDIT_COST_PARSE_RESUME_TEXT` |
| Analyze GitHub profile | 2 | `CREDIT_COST_GITHUB_ANALYSIS` |
| Rank candidates | 2 | `CREDIT_COST_RANKED_CANDIDATES` |
| Generate candidate test | 3 | `CREDIT_COST_GENERATE_TEST` |
| Evaluate test submission | 3 | `CREDIT_COST_EVALUATE_TEST` |

Plans live in one place — `CREDIT_PLANS` in `credits.py`. Edit prices, credit
amounts, names and feature bullets there; the landing page, signup page and
top-up modal all read from it via `/api/billing/plans`.

### How charging works

- **Deducted up front**, inside a single MongoDB `find_one_and_update` whose
  filter carries the `credits_balance >= cost` guard. The database applies the
  decrement only when the funds are there, so concurrent calls cannot
  overspend a balance. Verified: 8 simultaneous charges against a balance of 1
  yield exactly one success.
- **Refunded automatically** by `CreditRefundMiddleware` when the handler
  returns 4xx/5xx or raises. A failed call costs nothing.
- **Admins are never charged.**
- **Out of credits** returns `402 Payment Required` naming the cost, your
  balance, and the operation.
- Every movement is appended to the `credit_transactions` ledger: purchases,
  debits, refunds, signup grants and admin grants.

### Double-charge protection

Two independent layers:

- Every checkout sends an `idempotency_key`; Square returns the original
  payment for a repeated key rather than charging again.
- The ledger has a **unique sparse index** on `square_payment_id`, so a
  replayed payment is refused before credits are granted. Verified.

## Square setup

Get these from the Square Developer dashboard and put them in `.env`:

```
SQUARE_ENVIRONMENT=sandbox        # production when you go live
SQUARE_ACCESS_TOKEN=              # secret, server only
SQUARE_APPLICATION_ID=            # public, used by the browser SDK
SQUARE_LOCATION_ID=               # public
SQUARE_CURRENCY=USD
```

Until all three IDs are set, `/api/billing/plans` reports
`payments_enabled: false` and the UI says so plainly: the pricing page shows a
notice, paid signup is blocked, and the top-up modal tells the user to ask an
admin for a manual grant. The free plan works regardless.

Sandbox test card: `4111 1111 1111 1111`, any future expiry, CVV `111`.

Only `SQUARE_ACCESS_TOKEN` is secret. The application and location IDs are
public by design — the browser SDK needs them — and are served from
`/api/billing/config`. The access token is never sent to the browser.

## New endpoints

Public:
- `GET /api/billing/plans` — plans + credit costs
- `GET /api/billing/config` — public Square config
- `POST /api/billing/signup` — create account, charge card, grant credits

Authenticated:
- `GET /api/billing/me` — balance, spend, plan
- `GET /api/billing/transactions` — own ledger
- `POST /api/billing/topup` — buy another bundle

Admin only (all behind `get_current_admin`):
- `POST /api/billing/admin/users/{user_id}/credits` — grant credits
- `GET /api/billing/admin/revenue` — revenue summary
- `GET /api/billing/admin/transactions` — ledger across all accounts
- `GET /api/billing/admin/credit-costs` — configured costs

## Auth model

The browser keeps the JWT in `localStorage` and sends it as
`Authorization: Bearer`. The API is unchanged — same tokens, same
`HTTPBearer`, so existing API clients are unaffected.

Page-level auth checks in JS only decide what to render. **Every** piece of
data is gated server-side; `/superadmin/console` loaded by a non-admin renders
an empty console and 403s on every request.

## Files

New: `credits.py`, `payments.py`, `billing_routes.py`, `web_routes.py`,
`web/*.html`, `web/static/*`.

Changed: `main.py` (routers, static mount, credit dependencies, `/` → UI),
`config.py` (Square + credit settings), `models.py` (billing models, credit
fields), `database.py` (credit methods, ledger collection),
`auth.py` (login returns role and balance).
