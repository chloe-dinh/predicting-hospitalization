## Random Forest Model 

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_rand_forest_spec <- rand_forest(mtry = tune(), 
                                      trees = tune(), 
                                      min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

covid_rand_forest_wf <- workflow() %>%
  add_recipe(covid_recipe) %>%
  add_model(covid_rand_forest_spec)

covid_rf_param_grid <- grid_regular(mtry(range = c(2, 14)), 
                                    trees(range = c(2, 100)),
                                    min_n(range = c(10, 100)),
                                    levels = 6)


covid_rf_tune_res_auc <- tune_grid(
  covid_rand_forest_wf, 
  resamples = covid_folds, 
  grid = covid_rf_param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)


covid_best_rd_auc <- dplyr::arrange(collect_metrics(covid_rf_tune_res_auc), desc(mean))
head(covid_best_rd_auc)

best_rf_complexity_auc <- select_best(covid_rf_tune_res_auc)

covid_rf_final_auc <- finalize_workflow(covid_rand_forest_wf, best_rf_complexity_auc)
covid_rf_final_fit_auc <- fit(covid_rf_final_auc, data = covid_train)

covid_rf_tune_res_accuracy <- tune_grid(
  covid_rand_forest_wf, 
  resamples = covid_folds, 
  grid = covid_rf_param_grid, 
  metrics = metric_set(accuracy)
)

covid_best_rd_accuracy <- dplyr::arrange(collect_metrics(covid_rf_tune_res_accuracy), desc(mean))
head(covid_best_rd_accuracy)

best_rf_complexity_accuracy <- select_best(covid_rf_tune_res_accuracy)

covid_rf_final_accuracy <- finalize_workflow(covid_rand_forest_wf, best_rf_complexity_accuracy)
covid_rf_final_fit_accuracy <- fit(covid_rf_final_accuracy, data = covid_train)

save(covid_rf_tune_res_auc, covid_rf_final_fit_auc, covid_best_rd_auc,
     covid_rf_tune_res_accuracy, covid_rf_final_fit_accuracy, covid_best_rd_accuracy, 
     file = "/Users/chloe/Downloads/final-project/rda/covid_Random_Forest.rda")
