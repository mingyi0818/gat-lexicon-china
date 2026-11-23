# Human validation protocol for the GAT lexicon

Status: protocol and blank coding sample prepared; independent human coding not yet completed.

## Purpose

Validate whether the 37-term automated lexicon identifies tourism-related policy attention in provincial government work-report clauses, and quantify false positives caused by generic or ambiguous terms.

## Sampling

- File: `results/human_validation_sample_blank.csv`.
- Population: clauses of at least eight Chinese characters from all available provincial reports in 2003–2019.
- Fixed seed: 20261115.
- Stratification: 200 clauses containing at least one lexicon term and 200 clauses containing none, shuffled before coding.
- The machine label is retained for audit but should be hidden from coders during coding.

## Required coders

Two independent human coders who can read Chinese policy documents. Neither coder should see the other coder's labels. AI-generated labels must not be represented as human judgments.

## Coding questions

For each clause, code:

1. `tourism_relevant`: 1 if the clause substantively concerns tourism, tourists, tourism facilities, destinations, tourism-related cultural integration, or tourism-sector outcomes; otherwise 0.
2. `policy_action`: 1 if the clause contains a government action, commitment, target, allocation, regulation, construction, support, promotion, or evaluation related to tourism; otherwise 0.
3. `ambiguous`: record in `notes` if a lexical hit has a plausible non-tourism use, such as generic reception, hotel, or scenic-language use without a tourism-policy object.

## Adjudication

After independent coding:

- Calculate raw agreement and Cohen's kappa for both binary questions.
- Resolve disagreements through discussion; if unresolved, use a third human adjudicator.
- Populate the adjudicated columns only after independent labels are frozen.

## Metrics

Against adjudicated `tourism_relevant`:

- precision, recall, specificity, negative predictive value, F1;
- 95% bootstrap confidence intervals with clause-level resampling;
- false-positive rate by matched lexicon term/group;
- separate metrics for generic, tourism-format, designation/accommodation, and industry/outcome term groups.

Against adjudicated `policy_action`:

- report the share of machine-positive clauses that express a policy action;
- do not reinterpret lexical salience as policy effort if this share is low.

## Decision rules

- No universal numeric pass threshold is imposed after seeing the data.
- If precision or agreement is weak, rename the construct “tourism-related rhetorical salience” and narrow all policy-effort claims.
- All validation outcomes, including adverse outcomes, must be reported.

