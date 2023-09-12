## Decision Tree Model

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_dec_tree_spec <- decision_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart")

covid_dec_tree_wf <- workflow() %>%
  add_recipe(covid_recipe) %>%
  add_model(covid_dec_tree_spec %>% set_args(cost_complexity = tune())) 

covid_dt_param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

covid_dt_tune_res <- tune_grid(
  covid_dec_tree_wf, 
  resamples = covid_folds, 
  grid = covid_dt_param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)

covid_best_pruned_tree <- dplyr::arrange(collect_metrics(covid_dt_tune_res), desc(mean))
covid_best_pruned_tree

covid_dt_best_complexity <- select_best(covid_dt_tune_res)

covid_dt_final <- finalize_workflow(covid_dec_tree_wf, covid_dt_best_complexity)
covid_dt_final_fit <- fit(covid_dt_final, data = covid_train)

save(covid_dt_tune_res, covid_dt_final_fit, 
     file = "/Users/chloe/Downloads/final-project/rda/covid_Decision_Tree.rda")