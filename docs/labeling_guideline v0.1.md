# SmartSpend — Category Labeling Guideline (v0.1)

**Purpose:** This document operationalizes the 11-category taxonomy so that labeling stays consistent across data sources, across time, and across whoever does the annotation (you, and later, anyone helping with the OOD/multilingual slice). Every ambiguous transaction should be resolvable by checking this doc, not by gut feeling. Update the version number whenever you change a rule, and note the change in a changelog at the bottom — this becomes part of your dataset documentation deliverable.

---

## 1. Category Definitions

For each category: **scope**, **includes**, **excludes**, and 3–4 example transaction strings (mixed German/English, realistic noisy format).

### Groceries
- **Scope:** Food and household staples bought for home consumption.
- **Includes:** Supermarkets, discounters, butchers, bakeries, farmers markets.
- **Excludes:** Prepared/ready-to-eat food meant for immediate consumption (→ Restaurants).
- **Examples:** `REWE-SAG-7342`, `EDEKA 4471 FFM`, `ALDI SUED 0091`, `LIDL DE FRANKFURT`

### Restaurants
- **Scope:** Prepared food and drink, eaten in or taken away.
- **Includes:** Restaurants, cafés, bars, fast food, food delivery platforms.
- **Excludes:** Groceries bought to cook at home; alcohol bought at a supermarket (→ Groceries).
- **Examples:** `MCDONALDS FRANKFURT`, `LIEFERANDO*ORDER8821`, `STARBUCKS STORE 1234`, `DOENER KEBAB HAUS`

### Entertainment
- **Scope:** Leisure and media consumption, not travel-related.
- **Includes:** Streaming subscriptions, cinema, concerts, games, gyms/hobby memberships.
- **Excludes:** Sit-down dining as part of an outing (→ Restaurants unless it's a bundled ticket, e.g. cinema+snack combo → Entertainment).
- **Examples:** `NETFLIX.COM`, `CINESTAR FFM`, `SPOTIFY AB`, `FITX FITNESS MTL`

### Transportation
- **Scope:** Routine local/regional movement.
- **Includes:** Public transit, ride-hailing, fuel, parking, car maintenance, tolls.
- **Excludes:** Flights, hotels, long-distance/holiday travel (→ Travel).
- **Examples:** `RMV TICKET FFM`, `UBER *TRIP HELP.UBER.COM`, `SHELL TANKSTELLE`, `APCOA PARKING`

### Shopping
- **Scope:** Non-food retail goods, general merchandise.
- **Includes:** Clothing, electronics, home goods, online marketplaces (when the item isn't identifiable as another category).
- **Excludes:** Groceries; anything clearly medical (→ Healthcare).
- **Examples:** `AMAZON.DE MKTP`, `ZALANDO PAYMENTS`, `MEDIAMARKT FFM`, `IKEA DEUTSCHLAND`

### Utilities
- **Scope:** Recurring home-operating running costs (services consumed, not the roof over your head).
- **Includes:** Electricity, gas, water, internet, mobile phone plans, waste collection fees.
- **Excludes:** Rent or mortgage payments (→ Housing).
- **Examples:** `MAINOVA AG FFM`, `TELEKOM DEUTSCHLAND`, `VODAFONE GMBH`, `ENTEGA STROM`

### Housing
- **Scope:** Cost of occupying a home.
- **Includes:** Rent payments, mortgage payments, property management fees, HOA-style fees.
- **Excludes:** Utilities consumed inside the home (→ Utilities); one-off home-goods purchases (→ Shopping).
- **Examples:** `GREYSTAR RENT PMT PPD ID`, `MTG PMT PENFED CU`, `HAUSVERWALTUNG SCHMIDT MIETE`

### Insurance
- **Scope:** Insurance premium payments.
- **Includes:** Health, liability, car, home, life insurance premiums.
- **Excludes:** Actual medical treatment costs (→ Healthcare).
- **Examples:** `ALLIANZ VERSICHERUNG`, `TK KRANKENKASSE BEITRAG`, `HUK-COBURG`, `AXA VERSICHERUNG AG`

### Healthcare
- **Scope:** Direct medical spending.
- **Includes:** Pharmacies, doctor co-pays, dental, optical.
- **Excludes:** Insurance premiums (→ Insurance); wellness/gym (→ Entertainment).
- **Examples:** `APOTHEKE AM DOM`, `ZAHNARZT DR MUELLER`, `DM DROGERIE MARKT`, `SANACORP PHARMA`

### Education
- **Scope:** Learning-related spending.
- **Includes:** Tuition, course platforms, books for study, exam fees.
- **Excludes:** General bookstore purchases with no clear study purpose (→ Shopping, use judgment).
- **Examples:** `FRA-UAS SEMESTERBEITRAG`, `UDEMY ONLINE COURSE`, `THALIA BUCHHANDLUNG`, `COURSERA INC`

### Travel
- **Scope:** Long-distance and holiday-related spending.
- **Includes:** Flights, hotels, holiday booking platforms, travel insurance for a trip, car rental abroad.
- **Excludes:** Daily local commuting (→ Transportation).
- **Examples:** `LUFTHANSA AG`, `BOOKING.COM HOTEL`, `DB FERNVERKEHR AG`, `AIRBNB PAYMENTS`

### Income
- **Scope:** Money received, not spent. Kept separate from the 11 spending categories so downstream analytics don't conflate "how much came in" with "how it was spent."
- **Includes:** Salary/wage deposits, government direct deposits, refunds treated as standalone credits (see §3 for the refund-vs-original-category distinction).
- **Excludes:** Nothing spending-related belongs here at all.
- **Examples:** `US GOVERNMENT DIR DEP PPD ID`, `WAGE ARBEITGEBER GMBH GEHALT`, `GEHALTSZAHLUNG SEPTEMBER`

### Other
- **Scope:** True catch-all — use sparingly and only when no category fits after checking this whole guideline.
- **Includes:** Bank fees, ATM/account fees, cash withdrawals, transfers between own accounts, unidentifiable merchant codes.
- **Excludes:** Nothing that can be reasonably assigned elsewhere — "Other" should never exceed a small fraction of the dataset; if it does, that's a sign the taxonomy needs a category added.
- **Examples:** `GEBUEHREN KONTOFUEHRUNG`, `BARGELDAUSZAHLUNG GA AUTOMAT`, `UEBERTRAG EIGENE KONTEN`, `ATM OPERATOR FEE`, `ANNUAL FEE`

---

## 2. Confusable Pairs — Explicit Tie-Break Rules

| Pair | Rule |
|---|---|
| Restaurants vs. Groceries | If the merchant primarily sells food to be consumed *on the spot or immediately after purchase*, it's Restaurants. If it sells ingredients/products, it's Groceries. Supermarket bakeries selling a single coffee-to-go: still Groceries unless it's a standalone café brand. |
| Shopping vs. Entertainment | If the purchase is a physical/digital *good* (electronics, clothes), Shopping. If it's an *experience or access* (subscription, ticket, session), Entertainment. |
| Insurance vs. Healthcare | Recurring premium = Insurance. One-off treatment/prescription = Healthcare. |
| Transportation vs. Travel | Local/regional and recurring = Transportation. Long-distance, holiday, or booking-platform-mediated = Travel. |
| Utilities vs. Other | If it's a recognizable recurring home-operating cost, Utilities. If it's a bank/account mechanic (fee, transfer, withdrawal), Other. |
| Housing vs. Utilities | Payment secures the home itself (rent/mortgage) = Housing. Payment is for a service consumed inside it (power, water, internet) = Utilities. |
| Income vs. Other | Money coming in = Income, always — regardless of source (salary, government, refund). Money going out for bank mechanics = Other. |

---

## 3. Handling Special Cases

- **Refunds/reversals:** Label with the *original* category the refund corresponds to, not "Other" — note the transaction as a refund in a separate boolean column if your dataset schema supports it.
- **Recurring subscriptions with ambiguous merchant names** (e.g., generic payment processor strings like `PAYPAL *XYZ123`): if the underlying merchant can't be identified from the string alone, label as Other and flag for exclusion from clean training examples, since this exact ambiguity is realistic noise you want represented but shouldn't force a false-confident label.
- **Multi-item transactions:** Label by the *dominant* purchase type where identifiable; otherwise Other.
- **Confidence flag:** For borderline calls, add a `label_confidence: low/medium/high` field. This lets you later filter for a "clean" high-confidence training subset versus the full noisy set, and gives you material for the error-analysis chapter.

---

## 4. Annotation Process Notes

- Label in batches of ~50–100 at a time; re-read this guideline before each batch to avoid rule drift over a long session.
- For the self-labeled OOD/German slice: apply the exact same rules — do not create looser rules just because it's a smaller hand-labeled set.
- If a genuinely new pattern shows up that isn't covered here, add it to this document immediately (with an example), don't just decide silently — this is what "documented dataset" means for your deliverable.

---

## Changelog
- v0.1 — Initial version, 11 categories, core tie-break rules established.
- v0.2 — Added Housing and Income categories after inspecting real data from both training sources (US Bank v2, Mitulshah). Taxonomy is now 13 categories: Groceries, Restaurants, Entertainment, Transportation, Shopping, Utilities, Housing, Insurance, Healthcare, Education, Travel, Income, Other. Resolved the earlier open question on rent handling.