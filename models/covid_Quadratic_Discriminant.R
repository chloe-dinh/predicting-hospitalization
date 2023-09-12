## Quadratic Discriminant Analysis 

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

set.seed(3435)

covid_qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

covid_qda_wkflow <- workflow() %>% 
  add_model(covid_qda_mod) %>% 
  add_recipe(covid_recipe)

covid_qda_fit <- fit(covid_qda_wkflow, covid_train)
predict(covid_qda_fit, new_data = covid_train, type="prob")

covid_qda_kfold_fit <- fit_resamples(covid_qda_wkflow, covid_folds, control = control_grid(save_pred = TRUE))
collect_metrics(covid_qda_kfold_fit)

covid_roc_qda <- augment(covid_qda_fit, covid_train)

save(covid_qda_fit, covid_qda_kfold_fit, covid_roc_qda, 
     file = "/Users/chloe/Downloads/final-project/rda/covid_Quadratic_Discriminant.rda")