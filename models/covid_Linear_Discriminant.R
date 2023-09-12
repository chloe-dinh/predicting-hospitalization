## Linear Discriminant Analysis

load("/Users/chloe/Downloads/final-project/rda/covid_Model_Setup.rda")

library(parsnip)
library(discrim)

set.seed(3435)

covid_lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

covid_lda_wkflow <- workflow() %>% 
  add_model(covid_lda_mod) %>% 
  add_recipe(covid_recipe)

covid_lda_fit <- fit(covid_lda_wkflow, covid_train)
predict(covid_lda_fit, new_data = covid_train, type="prob")

covid_lda_kfold_fit <- fit_resamples(covid_lda_wkflow, covid_folds)
collect_metrics(covid_lda_kfold_fit)

save(covid_lda_fit, covid_lda_kfold_fit, 
     file = "/Users/chloe/Downloads/final-project/rda/covid_Linear_Discriminant.rda")