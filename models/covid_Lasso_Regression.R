## Lasso Regression Model

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_lasso_spec <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

covid_lasso_workflow <- workflow() %>% 
  add_recipe(covid_recipe) %>% 
  add_model(covid_lasso_spec)

lasso_pen_mix_grid <- grid_regular(penalty(range = c(-5, 5)), mixture(range = c(0,1)), levels = 10)
lasso_pen_mix_grid

covid_lasso_tune_res <- tune_grid(
  covid_lasso_workflow,
  resamples = covid_folds, 
  grid = lasso_pen_mix_grid
)

collect_metrics(covid_lasso_tune_res)
best_covid_lasso_penalty <- select_best(covid_lasso_tune_res, metric = "roc_auc")
best_covid_lasso_penalty

covid_lasso_final <- finalize_workflow(covid_lasso_workflow, best_covid_lasso_penalty)
covid_lasso_final_fit <- fit(covid_lasso_final, data = covid_train)

save(covid_lasso_tune_res, covid_lasso_final_fit, 
     file = "/Users/chloe/Downloads/final-project/rda/covid_Lasso_Regression.rda")
