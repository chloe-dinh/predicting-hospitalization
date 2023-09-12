## Support Vector Machine 

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_svm_rbf_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("kernlab")

covid_svm_rbf_wf <- workflow() %>%
  add_recipe(covid_recipe) %>%
  add_model(covid_svm_rbf_spec %>% set_args(cost = tune()))

covid_svm_param_grid <- grid_regular(cost(range = c(-10, 5)), levels = 10)

covid_svm_auc_tune_res <- tune_grid(
  covid_svm_rbf_wf, 
  resamples = covid_folds, 
  grid = covid_svm_param_grid,
  metrics = metric_set(yardstick::roc_auc)
)

covid_best_svm_auc <- dplyr::arrange(collect_metrics(covid_svm_auc_tune_res), desc(mean))
head(covid_best_svm_auc)

covid_best_svm_complexity_auc <- select_best(covid_svm_auc_tune_res)

covid_svm_final_auc <- finalize_workflow(covid_svm_rbf_wf, covid_best_svm_complexity_auc)
covid_svm_final_fit_auc <- fit(covid_svm_final_auc, data = covid_train)

covid_svm_accuracy_tune_res <- tune_grid(
  covid_svm_rbf_wf, 
  resamples = covid_folds, 
  grid = covid_svm_param_grid,
  metrics = metric_set(accuracy)
)

covid_best_svm_accuracy <- dplyr::arrange(collect_metrics(covid_svm_accuracy_tune_res), desc(mean))
head(covid_best_svm_accuracy)

best_svm_complexity_accuracy <- select_best(covid_svm_accuracy_tune_res)

covid_svm_final_accuracy <- finalize_workflow(covid_svm_rbf_wf, best_svm_complexity_accuracy)
covid_svm_final_fit_accuracy <- fit(covid_svm_final_accuracy, data = covid_train)

save(covid_svm_auc_tune_res, covid_best_svm_auc, covid_svm_final_fit_auc, 
     covid_best_svm_accuracy, covid_svm_final_fit_accuracy,
     file = "/Users/chloe/Downloads/final-project/rda/covid_Support_Vector_Machine.rda")