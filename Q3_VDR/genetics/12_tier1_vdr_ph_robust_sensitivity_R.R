# Source companion of 12_tier1_vdr_ph_robust_sensitivity_R.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Use Rscript or an R notebook kernel.

# %% Cell 2
# ============================================================
# Block 0. Setup and output policy
# ============================================================
TIER1_SNPS <- c("rs1544410", "rs2228570", "rs731236", "rs7975232")
MODEL_PREP_OUTPUT_DIR_NAME <- "Q3_VDR_v9_model_prep_outputs"
PH_OUTPUT_DIR_NAME <- "Q3_VDR_tier1_ph_sensitivity_outputs_R"

OVERWRITE_OUTPUTS <- tolower(Sys.getenv("Q3_VDR_OVERWRITE_OUTPUTS", "true")) %in% c("1", "true", "yes", "y")

infer_default_data_dir <- function() {
  cwd <- getwd()
  candidates <- c(
    cwd,
    file.path(cwd, "data"),
    "/home/jupyter/workspace/VDR",
    "/home/jupyter/workspace/VDR/data",
    "/home/jupyter/workspace/Q3_VDR",
    "/home/jupyter/workspace/Q3_VDR/data"
  )
  for (p in candidates) {
    if (dir.exists(p)) return(p)
  }
  cwd
}

DATA_DIR <- Sys.getenv("Q3_VDR_DATA_DIR", infer_default_data_dir())
MODEL_PREP_DIR <- Sys.getenv("Q3_VDR_MODEL_OUTPUT_DIR", file.path(DATA_DIR, MODEL_PREP_OUTPUT_DIR_NAME))
PH_DIR <- Sys.getenv("Q3_VDR_PH_SENSITIVITY_OUTPUT_DIR", file.path(DATA_DIR, PH_OUTPUT_DIR_NAME))
dir.create(PH_DIR, recursive = TRUE, showWarnings = FALSE)

save_csv <- function(df, filename) {
  path <- file.path(PH_DIR, filename)
  existed <- file.exists(path)
  if (existed && !OVERWRITE_OUTPUTS) stop(sprintf("Output exists and overwrite is disabled: %s", path))
  write.csv(df, path, row.names = FALSE, na = "")
  action <- if (existed) "overwritten" else "created"
  message(sprintf("%s: %s rows=%s cols=%s", action, path, nrow(df), ncol(df)))
  path
}

save_text <- function(lines, filename) {
  path <- file.path(PH_DIR, filename)
  existed <- file.exists(path)
  if (existed && !OVERWRITE_OUTPUTS) stop(sprintf("Output exists and overwrite is disabled: %s", path))
  writeLines(lines, con = path, useBytes = TRUE)
  action <- if (existed) "overwritten" else "created"
  message(sprintf("%s: %s", action, path))
  path
}

pkg_status <- data.frame(
  package = c("survival", "stats", "grDevices", "graphics"),
  available = c(requireNamespace("survival", quietly = TRUE), TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)
if (!pkg_status$available[pkg_status$package == "survival"]) {
  stop("R package 'survival' is required for this notebook but is not available.")
}
library(survival)

run_config <- data.frame(
  setting = c(
    "DATA_DIR", "MODEL_PREP_DIR", "PH_DIR", "OVERWRITE_OUTPUTS",
    "TIER1_SNPS", "scope", "model_families", "heavy_tvc_policy", "output_policy"
  ),
  value = c(
    DATA_DIR, MODEL_PREP_DIR, PH_DIR, as.character(OVERWRITE_OUTPUTS),
    paste(TIER1_SNPS, collapse = ";"),
    "Tier 1 VDR SNP x measured 25OHD CVD PH-robust sensitivities only; no MR/GRS/pathway expansion.",
    "primary Cox refit; age-stratified Cox; 2y/5y truncated Cox; 0-2/2-5/>5y window-specific Cox; Poisson rate with offset",
    "coxph(tt=...) TVC model skipped by default after Workbench kernel restart; window-specific Cox is the feasible time-varying sensitivity.",
    "overwrite same-name files in PH_DIR; do not delete unrelated files"
  ),
  stringsAsFactors = FALSE
)

save_csv(run_config, "00_ph_sensitivity_run_configuration.csv")
save_csv(pkg_status, "00_ph_sensitivity_package_status.csv")
cat("Reading model-prep inputs from:", MODEL_PREP_DIR, "\n")
cat("Writing PH sensitivity outputs to:", PH_DIR, "\n")

# %% Cell 3
# ============================================================
# Block 1. Load CVD model-prep dataset and enforce Tier 1 scope
# ============================================================
read_csv_required <- function(path, label) {
  if (!file.exists(path)) stop(sprintf("Missing required input for %s: %s", label, path))
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  message(sprintf("Loaded %s: rows=%s cols=%s from %s", label, nrow(df), ncol(df), path))
  df
}

cvd_path <- file.path(MODEL_PREP_DIR, "02_cvd_model_dataset_v9.csv")
cvd <- read_csv_required(cvd_path, "CVD model-prep dataset")

tier1_carrier_cols <- paste0(TIER1_SNPS, "_carrier")
missing_tier1 <- setdiff(tier1_carrier_cols, names(cvd))
if (length(missing_tier1) > 0) {
  stop(sprintf("Missing Tier 1 carrier columns in CVD model dataset: %s", paste(missing_tier1, collapse = ";")))
}

numeric_cols <- c(
  "vitd_main", "log_vitd", "vitd_lt20", "vitd_per10", "age_at_index", "female", "winter", "bmi_full",
  "follow_up_years", "event_mi_or_stroke", "baseline_diabetes", "baseline_hypertension",
  "baseline_hyperlipidemia", tier1_carrier_cols
)
for (col in intersect(numeric_cols, names(cvd))) {
  cvd[[col]] <- suppressWarnings(as.numeric(cvd[[col]]))
}

for (rsid in TIER1_SNPS) {
  carrier_col <- paste0(rsid, "_carrier")
  ix_bin <- paste0(rsid, "_x_vitd_lt20")
  cvd[[carrier_col]] <- as.integer(cvd[[carrier_col]])
  cvd[[ix_bin]] <- cvd$vitd_lt20 * cvd[[carrier_col]]
}

cvd$age_band <- cut(
  cvd$age_at_index,
  breaks = c(-Inf, 50, 65, 75, Inf),
  labels = c("<50", "50-64", "65-74", "75+"),
  right = FALSE
)

primary_cc_cols <- c(
  "follow_up_years", "event_mi_or_stroke", "vitd_lt20", "age_at_index",
  "female", "winter", "bmi_full", tier1_carrier_cols[1],
  paste0(TIER1_SNPS[1], "_x_vitd_lt20")
)
primary_cc_index <- complete.cases(cvd[, primary_cc_cols, drop = FALSE]) &
  !is.na(cvd$follow_up_years) & cvd$follow_up_years > 0

input_summary <- data.frame(
  metric = c(
    "cvd_rows",
    "events",
    "follow_up_positive_n",
    "primary_complete_case_n",
    "primary_complete_case_events",
    "age_band_nonmissing_n"
  ),
  value = c(
    nrow(cvd),
    sum(cvd$event_mi_or_stroke, na.rm = TRUE),
    sum(!is.na(cvd$follow_up_years) & cvd$follow_up_years > 0),
    sum(primary_cc_index, na.rm = TRUE),
    sum(cvd$event_mi_or_stroke[primary_cc_index], na.rm = TRUE),
    sum(!is.na(cvd$age_band))
  ),
  stringsAsFactors = FALSE
)
save_csv(input_summary, "01_ph_sensitivity_input_summary.csv")

age_band_summary <- as.data.frame(table(cvd$age_band, useNA = "ifany"), stringsAsFactors = FALSE)
names(age_band_summary) <- c("age_band", "n")
save_csv(age_band_summary, "01b_ph_sensitivity_age_band_summary.csv")

cat("Loaded CVD rows:", nrow(cvd), "\n")
input_summary

# %% Cell 4
# ============================================================
# Block 2. Helper functions
# ============================================================
bind_rows_base <- function(rows) {
  rows <- rows[vapply(rows, function(x) is.data.frame(x) && nrow(x) > 0, logical(1))]
  if (length(rows) == 0) return(data.frame())
  all_cols <- unique(unlist(lapply(rows, names)))
  aligned <- lapply(rows, function(d) {
    missing_cols <- setdiff(all_cols, names(d))
    for (col in missing_cols) d[[col]] <- NA
    d[, all_cols, drop = FALSE]
  })
  do.call(rbind, aligned)
}

complete_case_data <- function(df, needed) {
  missing_terms <- setdiff(needed, names(df))
  if (length(missing_terms) > 0) {
    return(list(data = data.frame(), status = "missing_terms", detail = paste(missing_terms, collapse = ";")))
  }
  d <- df[stats::complete.cases(df[, needed, drop = FALSE]), , drop = FALSE]
  d <- d[!is.na(d$follow_up_years) & d$follow_up_years > 0, , drop = FALSE]
  if (nrow(d) == 0) return(list(data = d, status = "no_complete_cases", detail = ""))
  list(data = d, status = "ready", detail = "")
}

inv_xtx <- function(x) {
  tryCatch(solve(x), error = function(e) qr.solve(x))
}

effect_from_vcov <- function(label, coef_vec, vc, weights, scale_label = "HR") {
  terms <- names(weights)
  if (length(terms) == 0 || any(!(terms %in% names(coef_vec))) || any(!(terms %in% rownames(vc)))) {
    return(data.frame(
      estimand = label, estimate_scale = scale_label, coef = NA_real_, se = NA_real_,
      p = NA_real_, estimate = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_,
      status = "missing_terms", detail = paste(setdiff(terms, names(coef_vec)), collapse = ";"),
      stringsAsFactors = FALSE
    ))
  }
  w <- as.numeric(weights)
  names(w) <- terms
  beta <- sum(w * coef_vec[terms])
  sub_v <- vc[terms, terms, drop = FALSE]
  variance <- as.numeric(t(w) %*% sub_v %*% w)
  se <- ifelse(is.finite(variance) && variance >= 0, sqrt(variance), NA_real_)
  z <- beta / se
  p <- 2 * pnorm(abs(z), lower.tail = FALSE)
  data.frame(
    estimand = label,
    estimate_scale = scale_label,
    coef = beta,
    se = se,
    p = p,
    estimate = exp(beta),
    ci_lower = exp(beta - 1.96 * se),
    ci_upper = exp(beta + 1.96 * se),
    status = "pass",
    detail = "",
    stringsAsFactors = FALSE
  )
}

cox_terms_table <- function(fit) {
  s <- summary(fit)
  coefs <- as.data.frame(s$coefficients)
  coefs$term <- rownames(coefs)
  rownames(coefs) <- NULL
  data.frame(
    term = coefs$term,
    coef = coefs$coef,
    se = coefs$`se(coef)`,
    z = coefs$z,
    p = coefs$`Pr(>|z|)`,
    hr = exp(coefs$coef),
    hr_ci_lower = exp(coefs$coef - 1.96 * coefs$`se(coef)`),
    hr_ci_upper = exp(coefs$coef + 1.96 * coefs$`se(coef)`),
    stringsAsFactors = FALSE
  )
}

cox_interaction_effect <- function(fit, interaction_term, label = "interaction") {
  effect_from_vcov(label, coef(fit), vcov(fit), setNames(1, interaction_term), "HR_or_HR_ratio")
}

fit_cox_model <- function(df, rsid, model_family, mode = c("primary", "age_stratified", "truncated", "window"), tau = NA, window_start = NA, window_end = NA, window_label = NA) {
  mode <- match.arg(mode)
  carrier <- paste0(rsid, "_carrier")
  interaction <- paste0(rsid, "_x_vitd_lt20")
  needed <- c("follow_up_years", "event_mi_or_stroke", "vitd_lt20", carrier, interaction, "age_at_index", "female", "winter", "bmi_full")
  if (mode == "age_stratified") needed <- c(needed, "age_band")
  cc <- complete_case_data(df, needed)
  info <- list(status = cc$status, detail = cc$detail, n = nrow(cc$data), events = NA, backend = "survival::coxph")
  if (cc$status != "ready") return(list(fit = NULL, info = info, effect = data.frame(), terms = data.frame()))
  d <- cc$data

  if (mode == "truncated") {
    d$analysis_time <- pmin(d$follow_up_years, tau)
    d$analysis_event <- as.integer(d$event_mi_or_stroke == 1 & d$follow_up_years <= tau)
    d <- d[d$analysis_time > 0, , drop = FALSE]
    info$backend <- paste0("survival::coxph_truncated_", tau, "y")
  } else if (mode == "window") {
    if (is.infinite(window_end)) {
      keep <- d$follow_up_years > window_start
      d <- d[keep, , drop = FALSE]
      d$analysis_time <- d$follow_up_years - window_start
      d$analysis_event <- as.integer(d$event_mi_or_stroke == 1 & d$follow_up_years > window_start)
    } else {
      keep <- d$follow_up_years > window_start
      d <- d[keep, , drop = FALSE]
      d$analysis_time <- pmin(d$follow_up_years, window_end) - window_start
      d$analysis_event <- as.integer(d$event_mi_or_stroke == 1 & d$follow_up_years > window_start & d$follow_up_years <= window_end)
    }
    d <- d[d$analysis_time > 0, , drop = FALSE]
    info$backend <- paste0("survival::coxph_window_", window_label)
  } else {
    d$analysis_time <- d$follow_up_years
    d$analysis_event <- d$event_mi_or_stroke
    if (mode == "age_stratified") info$backend <- "survival::coxph_age_band_strata"
  }

  info$n <- nrow(d)
  info$events <- sum(d$analysis_event, na.rm = TRUE)
  if (info$n < 100 || info$events < 50) {
    info$status <- "too_few_events"
    info$detail <- ""
    return(list(fit = NULL, info = info, effect = data.frame(), terms = data.frame()))
  }

  rhs <- paste0("vitd_lt20 + ", carrier, " + ", interaction, " + age_at_index + female + winter + bmi_full")
  if (mode == "age_stratified") rhs <- paste0("vitd_lt20 + ", carrier, " + ", interaction, " + female + winter + bmi_full + strata(age_band)")
  f <- as.formula(paste0("Surv(analysis_time, analysis_event) ~ ", rhs))
  fit <- tryCatch(coxph(f, data = d, ties = "efron", x = FALSE, model = FALSE), error = function(e) e)
  if (inherits(fit, "error")) {
    info$status <- "fit_error"
    info$detail <- conditionMessage(fit)
    return(list(fit = NULL, info = info, effect = data.frame(), terms = data.frame()))
  }
  info$status <- "pass"
  eff <- cox_interaction_effect(fit, interaction, paste0(model_family, "_interaction_ratio"))
  terms <- cox_terms_table(fit)
  list(fit = fit, info = info, effect = eff, terms = terms)
}

glm_poisson_vcov_hc0 <- function(fit) {
  X <- model.matrix(fit)
  y <- model.response(model.frame(fit))
  mu <- fitted(fit)
  bread <- inv_xtx(crossprod(X, X * as.numeric(mu)))
  score <- X * as.numeric(y - mu)
  meat <- crossprod(score)
  bread %*% meat %*% bread
}

fit_poisson_rate <- function(df, rsid) {
  carrier <- paste0(rsid, "_carrier")
  interaction <- paste0(rsid, "_x_vitd_lt20")
  needed <- c("follow_up_years", "event_mi_or_stroke", "vitd_lt20", carrier, interaction, "age_at_index", "female", "winter", "bmi_full")
  cc <- complete_case_data(df, needed)
  info <- list(status = cc$status, detail = cc$detail, n = nrow(cc$data), events = NA, backend = "glm_poisson_offset_sandwich_hc0")
  if (cc$status != "ready") return(list(fit = NULL, info = info, effect = data.frame()))
  d <- cc$data
  info$events <- sum(d$event_mi_or_stroke, na.rm = TRUE)
  f <- as.formula(paste0("event_mi_or_stroke ~ vitd_lt20 + ", carrier, " + ", interaction, " + age_at_index + female + winter + bmi_full + offset(log(follow_up_years))"))
  fit <- tryCatch(glm(f, data = d, family = poisson(link = "log")), error = function(e) e)
  if (inherits(fit, "error")) {
    info$status <- "fit_error"
    info$detail <- conditionMessage(fit)
    return(list(fit = NULL, info = info, effect = data.frame()))
  }
  vc <- tryCatch(glm_poisson_vcov_hc0(fit), error = function(e) e)
  if (inherits(vc, "error")) {
    info$status <- "sandwich_vcov_error"
    info$detail <- conditionMessage(vc)
    return(list(fit = fit, info = info, effect = data.frame()))
  }
  eff <- tryCatch(
    effect_from_vcov("poisson_rate_offset_interaction_ratio", coef(fit), vc, setNames(1, interaction), "RR_or_RR_ratio"),
    error = function(e) e
  )
  if (inherits(eff, "error")) {
    info$status <- "effect_error"
    info$detail <- conditionMessage(eff)
    return(list(fit = fit, info = info, effect = data.frame()))
  }
  info$status <- "pass"
  list(fit = fit, info = info, effect = eff)
}

add_status <- function(model_family, rsid, info, horizon = NA, time_window = NA) {
  data.frame(
    model_family = model_family,
    rsid = rsid,
    status = info$status,
    detail = info$detail,
    n = info$n,
    events = info$events,
    backend = info$backend,
    horizon_years = horizon,
    time_window = time_window,
    stringsAsFactors = FALSE
  )
}

decorate_effect <- function(df, model_family, rsid, info, horizon = NA, time_window = NA) {
  if (nrow(df) == 0) return(df)
  df$model_family <- model_family
  df$rsid <- rsid
  df$n <- info$n
  df$events <- info$events
  df$backend <- info$backend
  df$horizon_years <- horizon
  df$time_window <- time_window
  df
}

# %% Cell 5
# ============================================================
# Block 3. Fit memory-safe PH sensitivity model families
# ============================================================
spec_rows <- data.frame(
  model_family = c(
    "primary_cox_refit",
    "age_band_stratified_cox",
    "truncated_horizon_cox",
    "piecewise_window_cox",
    "poisson_rate_offset",
    "time_varying_coefficient_cox_not_run"
  ),
  role = c(
    "primary model preserved for comparison",
    "PH sensitivity for age_at_index signal",
    "2y and 5y horizon-specific Cox estimates",
    "0-2, 2-5, and >5 year follow-up window estimates",
    "non-PH rate-ratio sensitivity using follow-up offset",
    "methodologically direct but skipped by default after kernel restart from coxph(tt=...) memory expansion"
  ),
  primary_interpretation = c("no", "sensitivity", "sensitivity", "sensitivity", "sensitivity", "not_run"),
  stringsAsFactors = FALSE
)
save_csv(spec_rows, "02_ph_sensitivity_model_specifications.csv")

all_status <- list()
comparison_rows <- list()
term_rows <- list()

for (rsid in TIER1_SNPS) {
  primary <- fit_cox_model(cvd, rsid, "primary_cox_refit", mode = "primary")
  all_status[[length(all_status) + 1]] <- add_status("primary_cox_refit", rsid, primary$info)
  comparison_rows[[length(comparison_rows) + 1]] <- decorate_effect(primary$effect, "primary_cox_refit", rsid, primary$info)
  if (nrow(primary$terms) > 0) {
    terms <- primary$terms
    terms$model_family <- "primary_cox_refit"
    terms$rsid <- rsid
    term_rows[[length(term_rows) + 1]] <- terms
  }

  age_strat <- fit_cox_model(cvd, rsid, "age_band_stratified_cox", mode = "age_stratified")
  all_status[[length(all_status) + 1]] <- add_status("age_band_stratified_cox", rsid, age_strat$info)
  comparison_rows[[length(comparison_rows) + 1]] <- decorate_effect(age_strat$effect, "age_band_stratified_cox", rsid, age_strat$info)
}

status_df <- bind_rows_base(all_status)
comparison_df <- bind_rows_base(comparison_rows)
terms_df <- bind_rows_base(term_rows)
save_csv(status_df, "03_ph_sensitivity_fit_status.csv")
save_csv(comparison_df, "04_ph_sensitivity_primary_age_comparison_partial.csv")
save_csv(terms_df, "07c_ph_sensitivity_selected_model_terms.csv")
head(status_df, 20)

# %% Cell 6
# ============================================================
# Block 4. Fit horizon/window Cox and Poisson rate sensitivity
# ============================================================
block4_status <- list()
block4_rows <- list()

for (rsid in TIER1_SNPS) {
  for (tau in c(2, 5)) {
    trunc_fit <- fit_cox_model(cvd, rsid, paste0("truncated_horizon_cox_", tau, "y"), mode = "truncated", tau = tau)
    fam <- "truncated_horizon_cox"
    block4_status[[length(block4_status) + 1]] <- add_status(fam, rsid, trunc_fit$info, horizon = tau)
    block4_rows[[length(block4_rows) + 1]] <- decorate_effect(trunc_fit$effect, fam, rsid, trunc_fit$info, horizon = tau)
  }

  window_specs <- data.frame(
    window_start = c(0, 2, 5),
    window_end = c(2, 5, Inf),
    window_label = c("0-2y", "2-5y", ">5y"),
    stringsAsFactors = FALSE
  )
  for (k in seq_len(nrow(window_specs))) {
    wfit <- fit_cox_model(
      cvd, rsid, paste0("piecewise_window_cox_", window_specs$window_label[k]),
      mode = "window",
      window_start = window_specs$window_start[k],
      window_end = window_specs$window_end[k],
      window_label = window_specs$window_label[k]
    )
    fam <- "piecewise_window_cox"
    block4_status[[length(block4_status) + 1]] <- add_status(fam, rsid, wfit$info, time_window = window_specs$window_label[k])
    block4_rows[[length(block4_rows) + 1]] <- decorate_effect(wfit$effect, fam, rsid, wfit$info, time_window = window_specs$window_label[k])
  }

  pois <- fit_poisson_rate(cvd, rsid)
  block4_status[[length(block4_status) + 1]] <- add_status("poisson_rate_offset", rsid, pois$info)
  block4_rows[[length(block4_rows) + 1]] <- decorate_effect(pois$effect, "poisson_rate_offset", rsid, pois$info)
}

block4_status_df <- bind_rows_base(block4_status)
block4_comparison_df <- bind_rows_base(block4_rows)

status_df <- bind_rows_base(list(status_df, block4_status_df))
comparison_df <- bind_rows_base(list(comparison_df, block4_comparison_df))

tvc_skipped_status <- data.frame(
  model_family = "time_varying_coefficient_cox_not_run",
  rsid = TIER1_SNPS,
  status = "skipped_default_memory_guardrail",
  detail = "coxph(tt=...) caused Workbench kernel restart in prior version; use window-specific Cox as feasible time-varying sensitivity",
  n = NA,
  events = NA,
  backend = "not_run",
  horizon_years = NA,
  time_window = NA,
  stringsAsFactors = FALSE
)
status_df <- bind_rows_base(list(status_df, tvc_skipped_status))

save_csv(status_df, "03_ph_sensitivity_fit_status.csv")
save_csv(comparison_df, "08_ph_sensitivity_compact_comparison_table.csv")
head(comparison_df, 40)

# %% Cell 7
# ============================================================
# Block 5. Compact figure, interpretation note, and manifest
# ============================================================
comparison <- comparison_df
if (nrow(comparison) > 0) {
  comparison$model_role <- ifelse(comparison$model_family == "primary_cox_refit", "primary_refit", "ph_robust_sensitivity")
  comparison$ci_crosses_null <- with(comparison, ifelse(is.finite(ci_lower) & is.finite(ci_upper), ci_lower <= 1 & ci_upper >= 1, NA))
  comparison$nominal_p_lt_0_05 <- ifelse(is.finite(comparison$p), comparison$p < 0.05, NA)
  comparison$display_estimate <- sprintf("%.3f (%.3f, %.3f)", comparison$estimate, comparison$ci_lower, comparison$ci_upper)
  comparison$display_p <- ifelse(is.finite(comparison$p), sprintf("%.3g", comparison$p), "")
  comparison <- comparison[order(comparison$rsid, comparison$model_family, comparison$horizon_years, comparison$time_window), , drop = FALSE]
}
save_csv(comparison, "08_ph_sensitivity_compact_comparison_table.csv")

interpretation_flags <- data.frame(
  item = c(
    "primary_model_preserved",
    "heavy_tvc_model",
    "ph_violation_response",
    "main_estimand",
    "genetic_coding",
    "mr_causal_language",
    "comparison_rule"
  ),
  status = c(
    "yes",
    "skipped_default_memory_guardrail",
    "memory_safe_sensitivity_models_added",
    "effect_modification",
    "dominant_ALT_carrier",
    "prohibited",
    "compare_direction_and_ci_not_single_p_value"
  ),
  note = c(
    "Original Cox PH model remains primary and is refit only for side-by-side comparison.",
    "coxph(tt=...) caused a Workbench kernel restart in the prior version; it is not run by default.",
    "Age-stratified Cox, truncated-horizon Cox, window-specific Cox, and Poisson rate-offset sensitivity models are added.",
    "The target remains VDR carrier-status modification of the measured 25OHD-CVD association.",
    "Carrier status is not additive genotype dosage or externally weighted genetic instrument.",
    "Do not describe these analyses as MR or causal proof.",
    "Robustness is judged by whether interaction estimates remain near null across PH-robust alternatives."
  ),
  stringsAsFactors = FALSE
)
save_csv(interpretation_flags, "09_ph_sensitivity_interpretation_flags.csv")

make_figure_row <- function(filename, figure_type, status, detail = "", n_rows = NA) {
  data.frame(
    output_file = filename,
    path = file.path(PH_DIR, filename),
    figure_type = figure_type,
    status = status,
    detail = detail,
    n_rows = n_rows,
    stringsAsFactors = FALSE
  )
}

plot_comparison_forest <- function(df) {
  filename <- "21_ph_sensitivity_compact_interaction_forest.png"
  if (nrow(df) == 0) return(make_figure_row(filename, "ph_sensitivity_interaction_forest", "no_data", "", 0))
  d <- df[is.finite(df$estimate) & is.finite(df$ci_lower) & is.finite(df$ci_upper) & df$ci_lower > 0 & df$ci_upper > 0, , drop = FALSE]
  if (nrow(d) == 0) return(make_figure_row(filename, "ph_sensitivity_interaction_forest", "no_plottable_rows", "", 0))
  d$label <- ifelse(
    d$model_family == "truncated_horizon_cox",
    paste0(d$rsid, " | Cox truncated ", d$horizon_years, "y"),
    ifelse(
      d$model_family == "piecewise_window_cox",
      paste0(d$rsid, " | Cox window ", d$time_window),
      ifelse(
        d$model_family == "poisson_rate_offset",
        paste0(d$rsid, " | Poisson rate"),
        paste0(d$rsid, " | ", d$model_family)
      )
    )
  )
  d <- d[order(d$rsid, d$model_family, d$horizon_years, d$time_window), , drop = FALSE]
  path <- file.path(PH_DIR, filename)
  status <- "created"
  detail <- ""
  tryCatch({
    grDevices::png(path, width = 2300, height = max(1200, 240 + 68 * nrow(d)), res = 170)
    on.exit(grDevices::dev.off(), add = TRUE)
    par(mar = c(5, 13, 4, 2))
    y <- seq_len(nrow(d))
    log_range <- range(log(c(d$ci_lower, d$ci_upper, 1)), finite = TRUE)
    pad <- max(0.12, diff(log_range) * 0.08)
    xlim <- exp(log_range + c(-pad, pad))
    plot(d$estimate, y, log = "x", xlim = xlim, yaxt = "n", pch = 19,
         xlab = "Interaction estimate ratio, log scale", ylab = "",
         main = "Tier 1 VDR x vitamin D deficiency: PH-robust sensitivity comparison")
    segments(d$ci_lower, y, d$ci_upper, y, lwd = 1.2)
    abline(v = 1, lty = 2, col = "gray45")
    axis(2, at = y, labels = d$label, las = 1, cex.axis = 0.58)
  }, error = function(e) {
    status <<- "plot_error"
    detail <<- conditionMessage(e)
  })
  make_figure_row(filename, "ph_sensitivity_interaction_forest", status, detail, nrow(d))
}

fig_manifest <- plot_comparison_forest(comparison)
save_csv(fig_manifest, "20_ph_sensitivity_figure_manifest.csv")

primary_subset <- comparison[comparison$model_family == "primary_cox_refit", , drop = FALSE]
sensitivity_subset <- comparison[comparison$model_family != "primary_cox_refit", , drop = FALSE]
all_primary_cross <- if (nrow(primary_subset) > 0) all(primary_subset$ci_crosses_null %in% TRUE) else NA
all_sensitivity_cross <- if (nrow(sensitivity_subset) > 0) all(sensitivity_subset$ci_crosses_null %in% TRUE) else NA
any_sensitivity_nominal <- if (nrow(sensitivity_subset) > 0) any(sensitivity_subset$nominal_p_lt_0_05 %in% TRUE) else NA

note_lines <- c(
  "# Q3 VDR Tier 1 PH-Robust Sensitivity Interpretation",
  "",
  paste0("Date: ", format(Sys.Date(), "%Y-%m-%d")),
  "",
  "## Scope",
  "",
  "This note extends the Tier 1 VDR carrier-status x measured 25(OH)D CVD analysis after proportional-hazards diagnostics flagged the primary Cox models. The original Cox model remains the primary model. These analyses are sensitivity analyses, not MR, not an instrumental-variable analysis, and not causal proof.",
  "",
  "Tier 1 SNPs: rs1544410, rs2228570, rs731236, rs7975232.",
  "",
  "## Workbench Stability Change",
  "",
  "The previous coxph(tt=...) time-varying coefficient implementation caused a Workbench kernel restart before fitted-model outputs were written. This version skips that heavy model by default and uses memory-safe time-window alternatives.",
  "",
  "## Sensitivity Models Added",
  "",
  "1. Age-band stratified Cox model: addresses repeated PH signal for age_at_index by allowing separate baseline hazards across prespecified age bands.",
  "2. Truncated-horizon Cox models: estimates interaction ratios within 2-year and 5-year follow-up horizons.",
  "3. Piecewise/window-specific Cox models: estimates interaction ratios separately for 0-2, 2-5, and >5 years of follow-up.",
  "4. Poisson rate model with log follow-up offset and sandwich variance: provides a non-PH rate-ratio sensitivity.",
  "",
  "## Automated Summary Flags",
  "",
  paste0("- All primary Cox refit interaction CIs cross 1: ", all_primary_cross),
  paste0("- All PH-sensitivity interaction CIs cross 1: ", all_sensitivity_cross),
  paste0("- Any PH-sensitivity nominal interaction P < 0.05: ", any_sensitivity_nominal),
  "",
  "## Interpretation Template",
  "",
  "If the compact comparison table shows interaction estimates centered near 1 with confidence intervals crossing 1 across the PH-robust sensitivity families, the Tier 1 conclusion should remain: there is no robust evidence that classic VDR ALT-carrier status modifies the measured vitamin D deficiency-CVD association.",
  "",
  "If isolated PH-robust sensitivity rows show nominal P < 0.05, treat them as sensitivity signals requiring diagnostic review, not as a replacement for the primary null interaction result.",
  "",
  "## Methods References",
  "",
  "- Cox DR. Regression Models and Life-Tables. JRSS-B. 1972. DOI: https://doi.org/10.1111/j.2517-6161.1972.tb00899.x",
  "- Schoenfeld D. Partial residuals for the proportional hazards regression model. Biometrika. 1982. DOI: https://doi.org/10.1093/biomet/69.1.239",
  "- Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. Biometrika. 1994. DOI: https://doi.org/10.1093/biomet/81.3.515",
  "- Zou G. A modified Poisson regression approach to prospective studies with binary data. Am J Epidemiol. 2004. PMID: 15033648. PubMed: https://pubmed.ncbi.nlm.nih.gov/15033648/",
  "- Knol MJ, VanderWeele TJ. Recommendations for presenting analyses of effect modification and interaction. Int J Epidemiol. 2012. PMID: 22253321. PubMed: https://pubmed.ncbi.nlm.nih.gov/22253321/"
)
save_text(note_lines, "10_ph_sensitivity_updated_interpretation_note.md")

manifest_files <- list.files(PH_DIR, pattern = "[.](csv|png|pdf|md)$", full.names = TRUE)
manifest <- data.frame(
  output_file = basename(manifest_files),
  path = manifest_files,
  size_bytes = file.info(manifest_files)$size,
  stringsAsFactors = FALSE
)
save_csv(manifest, "99_ph_sensitivity_manifest.csv")

cat("DONE\n")
cat("PH sensitivity outputs are in:", PH_DIR, "\n")
cat("Upload these first: 03_ph_sensitivity_fit_status.csv, 08_ph_sensitivity_compact_comparison_table.csv, 10_ph_sensitivity_updated_interpretation_note.md, 20_ph_sensitivity_figure_manifest.csv, 21_ph_sensitivity_compact_interaction_forest.png, 99_ph_sensitivity_manifest.csv\n")
head(comparison, 40)
