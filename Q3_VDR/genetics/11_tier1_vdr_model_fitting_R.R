# Source companion of 11_tier1_vdr_model_fitting_R.ipynb
# Retains notebook cell order. Run in an authorized AoU environment.
# Use Rscript or an R notebook kernel.

# %% Cell 2
# ============================================================
# Block 0. Setup and output policy
# ============================================================
TIER1_SNPS <- c("rs1544410", "rs2228570", "rs731236", "rs7975232")
REBUILD_OUTPUT_DIR_NAME <- "Q3_VDR_v9_rebuild_outputs"
MODEL_PREP_OUTPUT_DIR_NAME <- "Q3_VDR_v9_model_prep_outputs"
TIER1_OUTPUT_DIR_NAME <- "Q3_VDR_tier1_model_outputs_R"

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
    if (dir.exists(p)) {
      return(p)
    }
  }
  cwd
}

DATA_DIR <- Sys.getenv("Q3_VDR_DATA_DIR", infer_default_data_dir())
MODEL_PREP_DIR <- Sys.getenv("Q3_VDR_MODEL_OUTPUT_DIR", file.path(DATA_DIR, MODEL_PREP_OUTPUT_DIR_NAME))
REBUILD_DIR <- Sys.getenv("Q3_VDR_REBUILD_OUTPUT_DIR", file.path(DATA_DIR, REBUILD_OUTPUT_DIR_NAME))
TIER1_DIR <- Sys.getenv("Q3_VDR_TIER1_R_MODEL_OUTPUT_DIR", file.path(DATA_DIR, TIER1_OUTPUT_DIR_NAME))
dir.create(TIER1_DIR, recursive = TRUE, showWarnings = FALSE)

save_csv <- function(df, filename) {
  path <- file.path(TIER1_DIR, filename)
  existed <- file.exists(path)
  if (existed && !OVERWRITE_OUTPUTS) {
    stop(sprintf("Output exists and overwrite is disabled: %s", path))
  }
  write.csv(df, path, row.names = FALSE, na = "")
  action <- if (existed) "overwritten" else "created"
  message(sprintf("%s: %s rows=%s cols=%s", action, path, nrow(df), ncol(df)))
  path
}

pkg_status <- data.frame(
  package = c("survival", "stats"),
  available = c(requireNamespace("survival", quietly = TRUE), TRUE),
  stringsAsFactors = FALSE
)
if (!pkg_status$available[pkg_status$package == "survival"]) {
  stop("R package 'survival' is required for coxph but is not available.")
}
library(survival)

run_config <- data.frame(
  setting = c(
    "DATA_DIR", "MODEL_PREP_DIR", "REBUILD_DIR", "TIER1_DIR",
    "OVERWRITE_OUTPUTS", "TIER1_SNPS", "model_engine", "scope_freeze", "output_policy"
  ),
  value = c(
    DATA_DIR, MODEL_PREP_DIR, REBUILD_DIR, TIER1_DIR,
    as.character(OVERWRITE_OUTPUTS), paste(TIER1_SNPS, collapse = ";"),
    "R survival::coxph + stats::lm HC3",
    "Tier 1 VDR SNP x measured 25OHD only; no Tier 2, Xiao/Wang, GRS, or MR.",
    "overwrite same-name CSV files in TIER1_DIR; do not delete unrelated files"
  ),
  stringsAsFactors = FALSE
)
save_csv(run_config, "00_tier1_R_model_run_configuration.csv")
save_csv(pkg_status, "00_tier1_R_package_status.csv")

cat("Reading model-prep inputs from:", MODEL_PREP_DIR, "\n")
cat("Writing Tier 1 R model outputs to:", TIER1_DIR, "\n")


# %% Cell 3
# ============================================================
# Block 1. Load model-prep datasets and enforce Tier 1 scope
# ============================================================
read_csv_required <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("Missing required input for %s: %s", label, path))
  }
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  message(sprintf("Loaded %s: rows=%s cols=%s from %s", label, nrow(df), ncol(df), path))
  df
}

read_csv_optional <- function(path, label) {
  if (!file.exists(path)) {
    message(sprintf("Missing optional input for %s: %s", label, path))
    return(data.frame())
  }
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  message(sprintf("Loaded %s: rows=%s cols=%s from %s", label, nrow(df), ncol(df), path))
  df
}

cvd <- read_csv_required(file.path(MODEL_PREP_DIR, "02_cvd_model_dataset_v9.csv"), "CVD model-prep dataset")
crp <- read_csv_optional(file.path(MODEL_PREP_DIR, "03_crp_model_dataset_v9.csv"), "CRP model-prep dataset")
candidate_plan <- read_csv_optional(file.path(MODEL_PREP_DIR, "01_candidate_analysis_plan_v9.csv"), "candidate analysis plan")
prep_summary <- read_csv_optional(file.path(MODEL_PREP_DIR, "04_model_dataset_summary_v9.csv"), "model dataset summary")

tier1_carrier_cols <- paste0(TIER1_SNPS, "_carrier")
missing_tier1 <- setdiff(tier1_carrier_cols, names(cvd))
if (length(missing_tier1) > 0) {
  stop(sprintf("Missing Tier 1 carrier columns in CVD model dataset: %s", paste(missing_tier1, collapse = ";")))
}

present_carrier_cols <- names(cvd)[grepl("^rs.*_carrier$", names(cvd))]
non_tier1_carrier_cols_present <- setdiff(present_carrier_cols, tier1_carrier_cols)
scope_status <- data.frame(
  item = c(
    "tier1_snps",
    "tier1_carrier_columns_present",
    "non_tier1_carrier_columns_present_but_not_used",
    "tier2_xiao_grs_mr_scope"
  ),
  value = c(
    paste(TIER1_SNPS, collapse = ";"),
    paste(tier1_carrier_cols, collapse = ";"),
    paste(non_tier1_carrier_cols_present, collapse = ";"),
    "not used in this notebook"
  ),
  status = c("included", "yes", "deferred", "excluded"),
  stringsAsFactors = FALSE
)
save_csv(scope_status, "01_tier1_R_scope_status.csv")

numeric_cols <- c(
  "vitd_main", "log_vitd", "vitd_lt20", "vitd_per10", "age_at_index", "female", "winter", "bmi_full",
  "follow_up_years", "event_mi_or_stroke", "baseline_diabetes", "baseline_hypertension",
  "baseline_hyperlipidemia", "crp_mgL", "log_crp", tier1_carrier_cols
)

coerce_numeric <- function(df, cols) {
  for (col in intersect(cols, names(df))) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
  df
}

cvd <- coerce_numeric(cvd, numeric_cols)
if (nrow(crp) > 0) {
  crp <- coerce_numeric(crp, numeric_cols)
}

for (rsid in TIER1_SNPS) {
  carrier_col <- paste0(rsid, "_carrier")
  ix_bin <- paste0(rsid, "_x_vitd_lt20")
  ix_log <- paste0(rsid, "_x_log_vitd")
  # Preserve NA carrier values if any; unknown genotype calls are not non-carriers.
  cvd[[carrier_col]] <- as.integer(cvd[[carrier_col]])
  cvd[[ix_bin]] <- cvd$vitd_lt20 * cvd[[carrier_col]]
  cvd[[ix_log]] <- cvd$log_vitd * cvd[[carrier_col]]
  if (nrow(crp) > 0 && carrier_col %in% names(crp)) {
    # Preserve NA carrier values if any; complete-case filtering will exclude them.
    crp[[carrier_col]] <- as.integer(crp[[carrier_col]])
    crp[[ix_bin]] <- crp$vitd_lt20 * crp[[carrier_col]]
    crp[[ix_log]] <- crp$log_vitd * crp[[carrier_col]]
  }
}

dataset_summary <- data.frame(
  dataset = c("cvd", "crp"),
  rows = c(nrow(cvd), nrow(crp)),
  events_mi_or_stroke = c(
    if ("event_mi_or_stroke" %in% names(cvd)) sum(cvd$event_mi_or_stroke, na.rm = TRUE) else NA,
    if (nrow(crp) > 0 && "event_mi_or_stroke" %in% names(crp)) sum(crp$event_mi_or_stroke, na.rm = TRUE) else NA
  ),
  columns = c(ncol(cvd), ncol(crp)),
  stringsAsFactors = FALSE
)
save_csv(dataset_summary, "02_tier1_R_input_dataset_summary.csv")
scope_status


# %% Cell 4
# ============================================================
# Block 2. Modeling helper functions
# ============================================================
holm_adjust <- function(p) {
  out <- rep(NA_real_, length(p))
  ok <- which(!is.na(p))
  if (length(ok) > 0) {
    out[ok] <- p.adjust(p[ok], method = "holm")
  }
  out
}

bind_rows_base <- function(rows) {
  rows <- rows[vapply(rows, function(x) is.data.frame(x) && nrow(x) > 0, logical(1))]
  if (length(rows) == 0) {
    return(data.frame())
  }
  all_cols <- unique(unlist(lapply(rows, names)))
  aligned <- lapply(rows, function(d) {
    missing_cols <- setdiff(all_cols, names(d))
    for (col in missing_cols) {
      d[[col]] <- NA
    }
    d[, all_cols, drop = FALSE]
  })
  do.call(rbind, aligned)
}

build_formula <- function(outcome, numeric_terms, categorical_terms = character(), survival = FALSE) {
  numeric_terms <- numeric_terms[!is.na(numeric_terms) & nzchar(numeric_terms)]
  categorical_terms <- categorical_terms[!is.na(categorical_terms) & nzchar(categorical_terms)]
  categorical_rhs <- if (length(categorical_terms) > 0) paste0("factor(", categorical_terms, ")") else character(0)
  rhs <- c(numeric_terms, categorical_rhs)
  if (survival) {
    as.formula(paste0("Surv(follow_up_years, event_mi_or_stroke) ~ ", paste(rhs, collapse = " + ")))
  } else {
    as.formula(paste0(outcome, " ~ ", paste(rhs, collapse = " + ")))
  }
}

complete_case_data <- function(df, needed) {
  needed <- unique(needed[!is.na(needed) & nzchar(needed)])
  missing_terms <- setdiff(needed, names(df))
  if (length(missing_terms) > 0) {
    return(list(data = data.frame(), status = "missing_terms", detail = paste(missing_terms, collapse = ";")))
  }
  d <- df[stats::complete.cases(df[, needed, drop = FALSE]), , drop = FALSE]
  if (nrow(d) == 0) {
    return(list(data = d, status = "no_complete_cases", detail = ""))
  }
  list(data = d, status = "ready", detail = "")
}

cox_result_table <- function(fit) {
  s <- summary(fit)
  coefs <- as.data.frame(s$coefficients)
  coefs$term <- rownames(coefs)
  rownames(coefs) <- NULL
  out <- data.frame(
    term = coefs$term,
    coef = coefs$coef,
    se = coefs$`se(coef)`,
    z_or_t = coefs$z,
    p = coefs$`Pr(>|z|)`,
    stringsAsFactors = FALSE
  )
  out$ci_lower_coef <- out$coef - 1.96 * out$se
  out$ci_upper_coef <- out$coef + 1.96 * out$se
  out$hr <- exp(out$coef)
  out$hr_ci_lower <- exp(out$ci_lower_coef)
  out$hr_ci_upper <- exp(out$ci_upper_coef)
  out
}

vcov_hc3 <- function(fit) {
  X <- model.matrix(fit)
  e <- residuals(fit)
  h <- hatvalues(fit)
  w <- as.numeric((e / pmax(1 - h, .Machine$double.eps))^2)
  xtx <- crossprod(X)
  bread <- tryCatch(solve(xtx), error = function(e) qr.solve(xtx))
  meat <- crossprod(X, X * w)
  bread %*% meat %*% bread
}

lm_hc3_result_table <- function(fit) {
  beta <- coef(fit)
  vc <- vcov_hc3(fit)
  se <- sqrt(diag(vc))
  tval <- beta / se
  pval <- 2 * pt(abs(tval), df = df.residual(fit), lower.tail = FALSE)
  out <- data.frame(
    term = names(beta),
    coef = as.numeric(beta),
    se = as.numeric(se),
    z_or_t = as.numeric(tval),
    p = as.numeric(pval),
    stringsAsFactors = FALSE
  )
  out <- out[out$term != "(Intercept)", , drop = FALSE]
  out$ci_lower_coef <- out$coef - 1.96 * out$se
  out$ci_upper_coef <- out$coef + 1.96 * out$se
  out$exp_coef <- exp(out$coef)
  out$percent_change <- (exp(out$coef) - 1) * 100
  list(table = out, vcov = vc)
}

fit_cox_model <- function(df, spec) {
  needed <- unique(c("follow_up_years", "event_mi_or_stroke", spec$numeric_terms, spec$categorical_terms))
  cc <- complete_case_data(df, needed)
  info <- list(status = cc$status, detail = cc$detail, n = nrow(cc$data), events = NA, n_predictors = NA, backend = "survival::coxph")
  if (cc$status != "ready") {
    return(list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL, info = info))
  }
  d <- cc$data
  info$events <- sum(d$event_mi_or_stroke, na.rm = TRUE)
  if (info$events < 20) {
    info$status <- "too_few_events"
    return(list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL, info = info))
  }
  f <- build_formula(NULL, spec$numeric_terms, spec$categorical_terms, survival = TRUE)
  fit <- tryCatch(coxph(f, data = d, ties = "efron", x = TRUE, model = TRUE), error = function(e) e)
  if (inherits(fit, "error")) {
    info$status <- "fit_error"
    info$detail <- conditionMessage(fit)
    return(list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL, info = info))
  }
  result <- cox_result_table(fit)
  info$status <- "pass"
  info$n_predictors <- length(coef(fit))
  list(result = result, vcov = vcov(fit), fit = fit, info = info)
}

fit_lm_model <- function(df, spec) {
  needed <- unique(c(spec$outcome, spec$numeric_terms, spec$categorical_terms))
  cc <- complete_case_data(df, needed)
  info <- list(status = cc$status, detail = cc$detail, n = nrow(cc$data), events = NA, n_predictors = NA, backend = "stats::lm_hc3")
  if (cc$status != "ready") {
    return(list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL, info = info))
  }
  d <- cc$data
  f <- build_formula(spec$outcome, spec$numeric_terms, spec$categorical_terms, survival = FALSE)
  fit <- tryCatch(lm(f, data = d), error = function(e) e)
  if (inherits(fit, "error")) {
    info$status <- "fit_error"
    info$detail <- conditionMessage(fit)
    return(list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL, info = info))
  }
  hc3 <- lm_hc3_result_table(fit)
  info$status <- "pass"
  info$n_predictors <- length(coef(fit)) - 1
  info$r_squared <- summary(fit)$r.squared
  list(result = hc3$table, vcov = hc3$vcov, fit = fit, info = info)
}

linear_combo <- function(label, weights, coef_vec, vc) {
  terms <- names(weights)
  if (length(terms) == 0 || any(!(terms %in% names(coef_vec))) || any(!(terms %in% rownames(vc)))) {
    return(data.frame(contrast = label, status = "missing_terms", stringsAsFactors = FALSE))
  }
  w <- as.numeric(weights)
  names(w) <- terms
  beta <- sum(w * coef_vec[terms])
  sub_v <- vc[terms, terms, drop = FALSE]
  variance <- as.numeric(t(w) %*% sub_v %*% w)
  se <- ifelse(variance >= 0, sqrt(variance), NA_real_)
  data.frame(
    contrast = label,
    coef = beta,
    se = se,
    ci_lower_coef = beta - 1.96 * se,
    ci_upper_coef = beta + 1.96 * se,
    hr_or_exp_coef = exp(beta),
    ci_lower = exp(beta - 1.96 * se),
    ci_upper = exp(beta + 1.96 * se),
    status = "pass",
    stringsAsFactors = FALSE
  )
}


# %% Cell 5
# ============================================================
# Block 3. Define Tier 1-only model specifications
# ============================================================
specs <- list()
idx <- 1
add_spec <- function(analysis_family, analysis_role, model_name, rsid, exposure, outcome = NA, numeric_terms, categorical_terms = character()) {
  carrier <- paste0(rsid, "_carrier")
  interaction <- if (exposure == "vitd_lt20") paste0(rsid, "_x_vitd_lt20") else paste0(rsid, "_x_log_vitd")
  specs[[idx <<- idx]] <<- list(
    analysis_family = analysis_family,
    analysis_role = analysis_role,
    model_name = model_name,
    rsid = rsid,
    exposure = exposure,
    outcome = outcome,
    carrier = carrier,
    interaction = interaction,
    numeric_terms = numeric_terms,
    categorical_terms = categorical_terms
  )
  idx <<- idx + 1
}

for (rsid in TIER1_SNPS) {
  carrier <- paste0(rsid, "_carrier")
  ix_bin <- paste0(rsid, "_x_vitd_lt20")
  ix_log <- paste0(rsid, "_x_log_vitd")
  add_spec("cvd_cox", "primary", "cvd_binary_core_with_bmi", rsid, "vitd_lt20",
           numeric_terms = c("vitd_lt20", carrier, ix_bin, "age_at_index", "female", "winter", "bmi_full"))
  add_spec("cvd_cox", "missingness_sensitivity", "cvd_binary_core_no_bmi", rsid, "vitd_lt20",
           numeric_terms = c("vitd_lt20", carrier, ix_bin, "age_at_index", "female", "winter"))
  add_spec("cvd_cox", "continuous_vitd_sensitivity", "cvd_logvitd_core_with_bmi", rsid, "log_vitd",
           numeric_terms = c("log_vitd", carrier, ix_log, "age_at_index", "female", "winter", "bmi_full"))
  add_spec("cvd_cox", "extended_covariate_sensitivity", "cvd_binary_extended_with_bmi_race_clinical", rsid, "vitd_lt20",
           numeric_terms = c("vitd_lt20", carrier, ix_bin, "age_at_index", "female", "winter", "bmi_full", "baseline_diabetes", "baseline_hypertension", "baseline_hyperlipidemia"),
           categorical_terms = c("race_eth_group"))
  add_spec("crp_ols", "crp_extension", "crp_loglog_core_with_bmi", rsid, "log_vitd", outcome = "log_crp",
           numeric_terms = c("log_vitd", carrier, ix_log, "age_at_index", "female", "winter", "bmi_full"))
}

spec_rows <- do.call(rbind, lapply(specs, function(s) {
  data.frame(
    analysis_family = s$analysis_family,
    analysis_role = s$analysis_role,
    model_name = s$model_name,
    rsid = s$rsid,
    exposure = s$exposure,
    outcome = ifelse(is.na(s$outcome), "", s$outcome),
    carrier = s$carrier,
    interaction = s$interaction,
    numeric_terms = paste(s$numeric_terms, collapse = ";"),
    categorical_terms = paste(s$categorical_terms, collapse = ";"),
    scope = "Tier 1 only",
    stringsAsFactors = FALSE
  )
}))
save_csv(spec_rows, "03_tier1_R_model_specifications.csv")
head(spec_rows, 30)


# %% Cell 6
# ============================================================
# Block 4. Fit CVD Cox models and CRP OLS extensions
# ============================================================
all_results <- list()
all_status <- list()
joint_rows <- list()
ph_rows <- list()
figure_rows <- list()
r_i <- 1
s_i <- 1
j_i <- 1
ph_i <- 1
fig_i <- 1

save_zph_plot <- function(zph, spec) {
  filename <- paste0("24_tier1_R_cox_zph_", spec$rsid, "_", spec$model_name, ".pdf")
  path <- file.path(TIER1_DIR, filename)
  status <- "created"
  detail <- ""
  tryCatch({
    grDevices::pdf(path, width = 10, height = 8)
    on.exit(grDevices::dev.off(), add = TRUE)
    plot(zph)
  }, error = function(e) {
    status <<- "plot_error"
    detail <<- conditionMessage(e)
  })
  data.frame(
    output_file = filename,
    path = path,
    figure_type = "cox_zph_residual_diagnostics",
    rsid = spec$rsid,
    model_name = spec$model_name,
    status = status,
    detail = detail,
    n_rows = NA,
    stringsAsFactors = FALSE
  )
}

for (spec in specs) {
  if (spec$analysis_family == "cvd_cox") {
    fit_obj <- fit_cox_model(cvd, spec)
  } else if (spec$analysis_family == "crp_ols") {
    if (nrow(crp) == 0) {
      fit_obj <- list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL,
                      info = list(status = "crp_dataset_missing", detail = "", n = 0, events = NA, n_predictors = NA, backend = "stats::lm_hc3"))
    } else {
      fit_obj <- fit_lm_model(crp, spec)
    }
  } else {
    fit_obj <- list(result = data.frame(), vcov = matrix(numeric(0), 0, 0), fit = NULL,
                    info = list(status = "unknown_analysis_family", detail = spec$analysis_family, n = NA, events = NA, n_predictors = NA, backend = ""))
  }

  info <- fit_obj$info
  all_status[[s_i]] <- data.frame(
    analysis_family = spec$analysis_family,
    analysis_role = spec$analysis_role,
    model_name = spec$model_name,
    rsid = spec$rsid,
    status = info$status,
    detail = info$detail,
    n = info$n,
    events = info$events,
    n_predictors = info$n_predictors,
    backend = info$backend,
    r_squared = ifelse(is.null(info$r_squared), NA, info$r_squared),
    stringsAsFactors = FALSE
  )
  s_i <- s_i + 1

  if (nrow(fit_obj$result) > 0) {
    res <- fit_obj$result
    res$analysis_family <- spec$analysis_family
    res$analysis_role <- spec$analysis_role
    res$model_name <- spec$model_name
    res$rsid <- spec$rsid
    res$n <- info$n
    res$events <- info$events
    res$backend <- info$backend
    res$exposure <- spec$exposure
    res$carrier <- spec$carrier
    res$interaction <- spec$interaction
    all_results[[r_i]] <- res
    r_i <- r_i + 1

    coef_vec <- coef(fit_obj$fit)
    vc <- fit_obj$vcov
    if (spec$analysis_family == "cvd_cox" && spec$exposure == "vitd_lt20") {
      contrasts <- list(
        linear_combo("vitamin_D_deficiency_HR_among_noncarriers", c(vitd_lt20 = 1), coef_vec, vc),
        linear_combo("vitamin_D_deficiency_HR_among_carriers", setNames(c(1, 1), c("vitd_lt20", spec$interaction)), coef_vec, vc),
        linear_combo("carrier_HR_among_nondeficient", setNames(1, spec$carrier), coef_vec, vc),
        linear_combo("carrier_HR_among_deficient", setNames(c(1, 1), c(spec$carrier, spec$interaction)), coef_vec, vc),
        linear_combo("multiplicative_interaction_ratio", setNames(1, spec$interaction), coef_vec, vc)
      )
      for (ct in contrasts) {
        ct$analysis_family <- spec$analysis_family
        ct$analysis_role <- spec$analysis_role
        ct$model_name <- spec$model_name
        ct$rsid <- spec$rsid
        ct$n <- info$n
        ct$events <- info$events
        joint_rows[[j_i]] <- ct
        j_i <- j_i + 1
      }
    } else if (spec$analysis_family == "crp_ols") {
      contrasts <- list(
        linear_combo("log_vitd_slope_among_noncarriers", c(log_vitd = 1), coef_vec, vc),
        linear_combo("log_vitd_slope_among_carriers", setNames(c(1, 1), c("log_vitd", spec$interaction)), coef_vec, vc),
        linear_combo("carrier_difference_at_log_vitd_zero", setNames(1, spec$carrier), coef_vec, vc),
        linear_combo("log_vitd_x_carrier_interaction", setNames(1, spec$interaction), coef_vec, vc)
      )
      for (ct in contrasts) {
        ct$analysis_family <- spec$analysis_family
        ct$analysis_role <- spec$analysis_role
        ct$model_name <- spec$model_name
        ct$rsid <- spec$rsid
        ct$n <- info$n
        ct$events <- NA
        joint_rows[[j_i]] <- ct
        j_i <- j_i + 1
      }
    }

    if (spec$analysis_family == "cvd_cox" && !is.null(fit_obj$fit)) {
      zph <- tryCatch(cox.zph(fit_obj$fit), error = function(e) e)
      if (!inherits(zph, "error")) {
        zph_table <- as.data.frame(zph$table)
        zph_table$term <- rownames(zph_table)
        rownames(zph_table) <- NULL
        names(zph_table) <- make.names(names(zph_table))
        zph_table$analysis_family <- spec$analysis_family
        zph_table$analysis_role <- spec$analysis_role
        zph_table$model_name <- spec$model_name
        zph_table$rsid <- spec$rsid
        ph_rows[[ph_i]] <- zph_table
        ph_i <- ph_i + 1
        if (spec$analysis_role == "primary") {
          figure_rows[[fig_i]] <- save_zph_plot(zph, spec)
          fig_i <- fig_i + 1
        }
      } else {
        ph_rows[[ph_i]] <- data.frame(
          analysis_family = spec$analysis_family,
          analysis_role = spec$analysis_role,
          model_name = spec$model_name,
          rsid = spec$rsid,
          term = "cox.zph_error",
          detail = conditionMessage(zph),
          stringsAsFactors = FALSE
        )
        ph_i <- ph_i + 1
      }
    }
  }
}

results_df <- bind_rows_base(all_results)
status_df <- bind_rows_base(all_status)
joint_df <- bind_rows_base(joint_rows)
ph_df <- bind_rows_base(ph_rows)
figure_df <- bind_rows_base(figure_rows)

save_csv(status_df, "04_tier1_R_model_fit_status.csv")
save_csv(results_df, "05_tier1_R_all_model_term_results.csv")
save_csv(joint_df, "06_tier1_R_joint_and_stratum_contrasts.csv")
save_csv(ph_df, "14_tier1_R_cox_ph_assumption_zph.csv")

primary <- if (nrow(results_df) > 0) {
  subset(results_df, analysis_family == "cvd_cox" & analysis_role == "primary")
} else data.frame()
sensitivity <- if (nrow(results_df) > 0) {
  subset(results_df, analysis_family == "cvd_cox" & analysis_role != "primary")
} else data.frame()
crp_results <- if (nrow(results_df) > 0) {
  subset(results_df, analysis_family == "crp_ols")
} else data.frame()

save_csv(primary, "07_tier1_R_cvd_cox_primary_term_results.csv")
save_csv(sensitivity, "08_tier1_R_cvd_cox_sensitivity_term_results.csv")
save_csv(crp_results, "09_tier1_R_crp_lm_term_results.csv")

head(status_df, 30)


# %% Cell 7
# ============================================================
# Block 5. Primary interaction p-value adjustment and reporting tables
# ============================================================
if (nrow(primary) > 0) {
  primary_interactions <- primary[primary$term == primary$interaction, , drop = FALSE]
  primary_interactions$p_bonferroni_4 <- pmin(as.numeric(primary_interactions$p) * length(TIER1_SNPS), 1)
  primary_interactions$p_holm_4 <- holm_adjust(as.numeric(primary_interactions$p))
  primary_interactions$multiple_testing_family <- "four Tier 1 VDR primary interaction tests"
} else {
  primary_interactions <- data.frame()
}

primary_joint <- if (nrow(joint_df) > 0) {
  subset(joint_df, analysis_family == "cvd_cox" & analysis_role == "primary")
} else data.frame()

make_figure_row <- function(filename, figure_type, status, detail = "", n_rows = NA) {
  data.frame(
    output_file = filename,
    path = file.path(TIER1_DIR, filename),
    figure_type = figure_type,
    rsid = "",
    model_name = "",
    status = status,
    detail = detail,
    n_rows = n_rows,
    stringsAsFactors = FALSE
  )
}

plot_forest <- function(df, filename, figure_type, label_col, estimate_col, lower_col, upper_col, title, xlab) {
  needed <- c(label_col, estimate_col, lower_col, upper_col)
  if (nrow(df) == 0 || any(!(needed %in% names(df)))) {
    return(make_figure_row(filename, figure_type, "no_data", "required columns missing or empty input", 0))
  }
  est <- suppressWarnings(as.numeric(df[[estimate_col]]))
  lo <- suppressWarnings(as.numeric(df[[lower_col]]))
  hi <- suppressWarnings(as.numeric(df[[upper_col]]))
  ok <- is.finite(est) & is.finite(lo) & is.finite(hi) & lo > 0 & hi > 0
  d <- df[ok, , drop = FALSE]
  est <- est[ok]
  lo <- lo[ok]
  hi <- hi[ok]
  if (nrow(d) == 0) {
    return(make_figure_row(filename, figure_type, "no_plottable_rows", "no finite positive HR/CI values", 0))
  }
  labels <- as.character(d[[label_col]])
  ord <- order(est)
  est <- est[ord]
  lo <- lo[ord]
  hi <- hi[ord]
  labels <- labels[ord]
  path <- file.path(TIER1_DIR, filename)
  status <- "created"
  detail <- ""
  tryCatch({
    grDevices::png(path, width = 2100, height = max(900, 260 + 95 * length(labels)), res = 160)
    on.exit(grDevices::dev.off(), add = TRUE)
    par(mar = c(5, 11, 4, 2))
    y <- seq_along(labels)
    log_range <- range(log(c(lo, hi, 1)), finite = TRUE)
    pad <- max(0.12, diff(log_range) * 0.08)
    xlim <- exp(log_range + c(-pad, pad))
    plot(est, y, log = "x", xlim = xlim, yaxt = "n", pch = 19,
         xlab = xlab, ylab = "", main = title)
    segments(lo, y, hi, y, lwd = 1.6)
    abline(v = 1, lty = 2, col = "gray45")
    axis(2, at = y, labels = labels, las = 1, cex.axis = 0.72)
  }, error = function(e) {
    status <<- "plot_error"
    detail <<- conditionMessage(e)
  })
  make_figure_row(filename, figure_type, status, detail, length(labels))
}

plot_ph_global <- function(ph_df) {
  filename <- "23_tier1_R_cox_ph_global_pvalues.png"
  if (nrow(ph_df) == 0 || !("p" %in% names(ph_df)) || !("term" %in% names(ph_df))) {
    return(make_figure_row(filename, "cox_ph_global_pvalue_plot", "no_data", "cox.zph table missing", 0))
  }
  d <- subset(ph_df, analysis_role == "primary" & term == "GLOBAL")
  d$p <- suppressWarnings(as.numeric(d$p))
  d <- d[is.finite(d$p), , drop = FALSE]
  if (nrow(d) == 0) {
    return(make_figure_row(filename, "cox_ph_global_pvalue_plot", "no_plottable_rows", "no finite GLOBAL p-values", 0))
  }
  path <- file.path(TIER1_DIR, filename)
  status <- "created"
  detail <- ""
  tryCatch({
    grDevices::png(path, width = 1500, height = 950, res = 160)
    on.exit(grDevices::dev.off(), add = TRUE)
    vals <- -log10(pmax(d$p, .Machine$double.xmin))
    par(mar = c(7, 5, 4, 2))
    barplot(vals, names.arg = d$rsid, las = 2, ylab = "-log10(P)",
            main = "Cox PH global tests: Tier 1 primary models")
    abline(h = -log10(0.05), lty = 2, col = "gray45")
  }, error = function(e) {
    status <<- "plot_error"
    detail <<- conditionMessage(e)
  })
  make_figure_row(filename, "cox_ph_global_pvalue_plot", status, detail, nrow(d))
}

figure_extra <- list()
figure_extra[[1]] <- plot_forest(
  primary_interactions,
  "21_tier1_R_primary_interaction_forest.png",
  "primary_interaction_hr_forest",
  "rsid", "hr", "hr_ci_lower", "hr_ci_upper",
  "VDR x vitamin D deficiency interaction HR ratios",
  "Multiplicative interaction HR ratio, log scale"
)

joint_plot <- primary_joint
if (nrow(joint_plot) > 0 && "contrast" %in% names(joint_plot)) {
  keep_contrasts <- c(
    "vitamin_D_deficiency_HR_among_noncarriers",
    "vitamin_D_deficiency_HR_among_carriers",
    "multiplicative_interaction_ratio"
  )
  joint_plot <- joint_plot[joint_plot$contrast %in% keep_contrasts, , drop = FALSE]
  contrast_label <- c(
    vitamin_D_deficiency_HR_among_noncarriers = "VitD<20 HR noncarrier",
    vitamin_D_deficiency_HR_among_carriers = "VitD<20 HR carrier",
    multiplicative_interaction_ratio = "interaction ratio"
  )
  joint_plot$short_contrast <- contrast_label[joint_plot$contrast]
  joint_plot$plot_label <- paste(joint_plot$rsid, joint_plot$short_contrast, sep = " | ")
}
figure_extra[[2]] <- plot_forest(
  joint_plot,
  "22_tier1_R_primary_joint_hr_forest.png",
  "primary_joint_hr_forest",
  "plot_label", "hr_or_exp_coef", "ci_lower", "ci_upper",
  "Tier 1 primary joint HR contrasts",
  "HR or interaction ratio, log scale"
)
figure_extra[[3]] <- plot_ph_global(ph_df)

figure_extra_df <- bind_rows_base(figure_extra)
figure_df <- bind_rows_base(list(figure_df, figure_extra_df))

complete_case_counts <- status_df[, c("analysis_family", "analysis_role", "model_name", "rsid", "n", "events", "status")]
names(complete_case_counts)[names(complete_case_counts) == "n"] <- "complete_case_n"

interpretation_flags <- data.frame(
  item = c("scope", "genetic_coding", "primary_estimand", "population_stratification", "multiple_testing", "bmi_missingness", "cox_diagnostics"),
  status = c("Tier 1 only", "dominant_ALT_carrier", "effect_modification", "not_fully_solved", "Tier 1 family only", "important", "cox.zph_output_created"),
  note = c(
    "Expanded VDR, Xiao/Wang pathway, weighted GRS, genotype extraction, and MR are deferred.",
    "Carrier status is not additive genotype dosage.",
    "Interaction is interpreted as effect modification in observational data, not causal proof.",
    "No genetic ancestry PCs are included; race/ethnicity is only used in extended sensitivity.",
    "Primary interaction p-values adjusted across four Tier 1 SNPs.",
    "BMI-adjusted primary model is complete-case; no-BMI model is missingness sensitivity.",
    "Schoenfeld residual tests are saved for Cox models."
  ),
  stringsAsFactors = FALSE
)

save_csv(primary_interactions, "10_tier1_R_primary_interaction_multiple_testing.csv")
save_csv(primary_joint, "11_tier1_R_cvd_primary_joint_hr_table.csv")
save_csv(complete_case_counts, "12_tier1_R_complete_case_counts.csv")
save_csv(interpretation_flags, "13_tier1_R_interpretation_flags.csv")
save_csv(figure_df, "20_tier1_R_figure_manifest.csv")

manifest_files <- list.files(TIER1_DIR, pattern = "[.](csv|png|pdf)$", full.names = TRUE)
manifest <- data.frame(
  output_file = basename(manifest_files),
  path = manifest_files,
  size_bytes = file.info(manifest_files)$size,
  stringsAsFactors = FALSE
)
save_csv(manifest, "99_tier1_R_model_manifest.csv")

cat("DONE\n")
cat("Tier 1 R model outputs are in:", TIER1_DIR, "\n")
cat("Upload these first: 04_tier1_R_model_fit_status.csv, 07_tier1_R_cvd_cox_primary_term_results.csv, 10_tier1_R_primary_interaction_multiple_testing.csv, 11_tier1_R_cvd_primary_joint_hr_table.csv, 12_tier1_R_complete_case_counts.csv, 14_tier1_R_cox_ph_assumption_zph.csv, 20_tier1_R_figure_manifest.csv, 21_tier1_R_primary_interaction_forest.png, 22_tier1_R_primary_joint_hr_forest.png, 23_tier1_R_cox_ph_global_pvalues.png, 24_tier1_R_cox_zph_<rsid>_cvd_binary_core_with_bmi.pdf, 99_tier1_R_model_manifest.csv\n")
primary_interactions

