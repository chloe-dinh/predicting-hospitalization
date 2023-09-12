## Logistical Regression Model 

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_log_reg <- logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")

covid_log_wkflow <- workflow() %>% 
  add_model(covid_log_reg) %>% 
  add_recipe(covid_recipe)

covid_log_fit <- fit(covid_log_wkflow, covid_train)
predict(covid_log_fit, new_data = covid_train, type="prob")

covid_log_kfold_fit <- fit_resamples(covid_log_wkflow, covid_folds)
collect_metrics(covid_log_kfold_fit)

save(covid_log_fit, covid_log_kfold_fit,
     file = "/Users/chloe/Downloads/final-project/rda/covid_Logistic_Regression.rda")