library(dplyr)
library(vbvs.concurrent)
library(refund.shiny)

train_test_cross_validation_split <- function(data, drug){
  drug_names <- data$subj
  drug_names_matrix <- matrix(drug_names)
  test_index <- which(drug_names_matrix == drug)
  train_index <- which(drug_names_matrix != drug)
  return (list(test_index, train_index))
}

CV_regressor <- function(base_path, data, kinase, v0, Kp, Kt, A, B) {
  base_path = paste(base_path, "kinase_", kinase, "_v0_", v0, "_Kp_", Kp, "_Kt_", Kt, "_A_", A, "_B_", B, "/", sep = "")
  dir.create(base_path, recursive = TRUE)

  y_test_predicted_final <- data$Y
  y_test_true_standartized_final <-data$Y

  kinase_str <- paste("Cov_", kinase, sep="")

  for (drug in unique(matrix(data$subj))) {
    print(paste0("Drug ", drug))
    split = train_test_cross_validation_split(data, drug)
    test_index <- split[[1]]
    train_index <- split[[2]]
    data_test <- data[test_index, c("subj", "Y", "time", kinase_str)]
    data_train <- data[train_index, c("subj", "Y", "time", kinase_str)]

    formula_str <- paste("Y ~ Cov_", kinase, " | time", sep="")
    print(formula_str)
    model <- as.formula(formula_str)

    fit_vbvs = vbvs_concurrent(model, id.var = "subj", Kp = Kp, Kt = Kt, data = data_train, t.min = 0.0, t.max = 10.0, standardized = FALSE, v0 = v0, A = A, B = B, Atheta = A, Btheta = B)

    y_test_predicted <- predict(fit_vbvs, data.new = data_test, standardized = FALSE)
    y_test_true_standartized <- standardize_variables(data.original = data_train, data.new = data_test, time.var = "time", LHS = "Y", trmstrings = c(kinase_str))$Y

    y_test_predicted_final[test_index] <- y_test_predicted
    y_test_true_standartized_final[test_index] <- y_test_true_standartized
  }

  y_test_predicted_final_path <- paste(base_path, "y_test_predicted_final.csv", sep = "")
  write.csv(y_test_predicted_final, y_test_predicted_final_path, row.names=FALSE)

  y_test_true_standartized_final_path <- paste(base_path, "y_test_true_standartized_final.csv", sep = "")
  write.csv(y_test_true_standartized_final, y_test_true_standartized_final_path, row.names=FALSE)

  return (list(y_test_predicted_final, y_test_true_standartized_final, fit_vbvs))
}

hyperparameters_config <- list(
  # list(v0 = 0.5, Kp = 1, Kt = 5, A = 0.5, B = 0.5),
  list(v0 = 0.1, Kp = 1, Kt = 4, A = 0.5, B = 0.5)
  # list(v0 = 10, Kp = 4, Kt = 7, A = 1, B = 6)
)

data <- read.csv("./fof/data/r/prepared_data.csv")

base_path <- "./fof/data/r/hyperparameter_tuning_cross_validation_one_kinase/"

for (hyperparameters in hyperparameters_config){
  print("ITERATION")
  print(hyperparameters)
  v0 <- hyperparameters$v0
  Kp <- hyperparameters$Kp
  Kt <- hyperparameters$Kt
  A <- hyperparameters$A
  B <- hyperparameters$B

  # kinase_list <- c(1, 252, 264, 281, 285, 294, 369)

  for (kinase in 1:369){
    final_result <- CV_regressor(base_path, data, kinase, v0, Kp, Kt, A, B)
    model <- final_result[[3]]
    print('GAMMA SUM')
    print(sum(model$gamma.pm))
    print('GAMMA > 0.01 count')
    print(sum(model$gamma.pm > 0.01))
  }
}