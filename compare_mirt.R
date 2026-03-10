library(mirt)

cat("Loading Swedish SAT binary data...\n")
data <- read.csv("/tmp/swesat_binary.csv")
cat(sprintf("Data: %d respondents x %d items\n", nrow(data), ncol(data)))
cat("Items 1-80 = quantitative, items 81-160 = verbal\n\n")

# Confirmatory model: same structure as Python test
#   Factor 1 loads on items 1-80  (quantitative)
#   Factor 2 loads on items 81-160 (verbal)
#   Freely estimate the correlation between F1 and F2
# model_spec <- mirt.model("
#   F1 = 1-80
#   F2 = 81-160
#   COV = F1*F2
# ")

data <- data[, 1:80]
model_spec <- mirt.model("
  F1 = 1-40
  F2 = 41-80
  COV = F1*F2
")
model_spec <- mirt.model("
  F1 = 1-40
  F2 = 41-80
")

cat("Fitting confirmatory 2D 2PL model (this may take a while)...\n")
fit <- mirt(data, model_spec, itemtype = "2PL", verbose = TRUE)

cat("\n========================================\n")
cat("Model summary\n")
cat("========================================\n")
print(summary(fit, suppress = 0.0))

cat("\n========================================\n")
cat("Factor covariance / correlation matrix\n")
cat("========================================\n")
cov_mat <- vcov(fit)  # parameter vcov - not what we want
# Extract the latent factor correlation
s <- summary(fit)
cat("Latent factor correlation (F1, F2):\n")
print(s$fcor)

# Save item parameters
cat("\n========================================\n")
cat("Item parameter summary (first 5 and last 5 items)\n")
cat("========================================\n")
params <- coef(fit, simplify = TRUE, IRTpars = TRUE)$items
print(head(params, 5))
cat("...\n")
print(tail(params, 5))
