library(tidymodels)
rm(list =ls())
options(scipen = 999)
options(digits = 5)

###### Lectura de datos ######

datos = readr::read_csv('C:/Users/maaf1/OneDrive - lamolina.edu.pe/Escritorio/archive/cleaned_merged_heart_dataset.csv')

## Conversión a factor

datos$sex = factor(datos$sex, levels = c(0,1), labels = c('female','male'))
datos$cp = factor(datos$cp, levels = c(0,1,2,3))
datos$fbs = as.factor(datos$fbs)
datos$restecg = as.factor(datos$restecg)
datos$exang = as.factor(datos$exang)
datos$slope = as.factor(datos$slope)
datos$ca = as.factor(datos$ca)
datos$thal = as.factor(datos$thal)
datos$target = as.factor(datos$target)

###### Partición de datos ######

set.seed(2024)
datos_split = initial_split(datos, prop = .7, strata = target)
datos_split

datos_train = training(datos_split)
datos_test = testing(datos_split)
set.seed(2024)
datos_train = recipe(target~.,data = datos_train) %>% 
  step_impute_knn(all_predictors()) %>% 
  prep() %>% 
  bake(new_data = NULL)
set.seed(2024)
datos_test = recipe(target~.,data = datos_train) %>% 
  step_impute_knn(all_predictors()) %>% 
  prep() %>% 
  bake(new_data = datos_test)
  
##### Modelo de árboles de decisión ######

## Validación cruzada

datos_fold <- 
  vfold_cv(datos_train, v = 20, strata = target)

## NA

set.seed(2024)
datos_recipe_dt = 
  recipe(target~., datos_train) %>%
  step_impute_knn(all_predictors())

## Creación del modelo

des_tree = decision_tree() %>% 
  set_engine('rpart') %>% 
  set_mode('classification')

## Flujo de trabajo

work_flow_dt = 
  workflow() %>% 
  add_model(des_tree) %>%
  add_recipe(datos_recipe_dt)

modelo_final_dt=
  work_flow_dt %>% 
  fit_resamples(resamples = datos_fold) %>% 
  show_best(metric='accuracy')

modelo_final_dt_fit=
  work_flow_dt %>% 
  finalize_workflow(modelo_final_dt) %>% 
  fit(data = datos_train)

## Predicción

class_pred=
  modelo_final_dt_fit %>% 
  predict(new_data = datos_test)

resultados_dt = 
  datos_test %>% 
  select(target) %>% 
  bind_cols(class_pred)

resultados_dt %>%  
  conf_mat(truth = target,
           estimate = .pred_class)

resultados_dt %>%  
  accuracy(truth = target,
           estimate = .pred_class)

## Creación del modelo

des_tree = decision_tree(tree_depth = tune(),
                         min_n = tune(),
                         cost_complexity = tune()) %>% 
  set_engine('rpart') %>% 
  set_mode('classification')

## Flujo de trabajo

work_flow_dt = 
  workflow() %>% 
  add_model(des_tree) %>%
  add_recipe(datos_recipe_dt)


# Tuneo de hiperparametros

hiperparameter = 
  grid_random(extract_parameter_set_dials(des_tree),
              size = 700)

modelos_elegidos = tibble(cost_complexity = 
                            c(0.0000000001,6.70e-10,0.000000232,0.00000000309,0.000000998,0.00000978),
                          tree_depth = 
                            c(10,13,14,10,12,15),
                          min_n = c(2,2,3,2,3,4)
                          )

library(doParallel)
library(parallel)
registerDoParallel(cores = parallel::detectCores())
tictoc::tic()
resultados_tune = 
  work_flow_dt %>% 
  tune_grid(resamples = datos_fold, grid = modelos_elegidos, metrics = metric_set(roc_auc,sens,spec,accuracy))
tictoc::toc()
stopImplicitCluster()

# Selección del mejor modelo por Accuracy

best_model = 
  resultados_tune %>% 
  select_best(metric = 'accuracy')

best_model_fit = 
  work_flow_dt %>% 
  finalize_workflow(best_model) %>% 
  fit(data = datos_train)

set.seed(2024)
work_flow_dt %>%
  finalize_workflow(best_model) %>%
  last_fit(split = datos_split, metrics = metric_set(roc_auc,sens,spec,accuracy)) %>% 
  collect_metrics(summarize=T)

# Predicción

class_pred_tune = 
  best_model_fit %>% 
  predict(new_data = datos_test)

resultados_dt_tune = 
  datos_test %>% 
  select(target) %>% 
  bind_cols(class_pred_tune)

# Metricas

resultados_dt_tune %>%  
  conf_mat(truth = target,
           estimate = .pred_class)

resultados_dt_tune %>%  
  accuracy(truth = target,
           estimate = .pred_class)

####### Probando con nuevos datos de testeo #######

datos_2 = readr::read_csv('C:/Users/maaf1/OneDrive - lamolina.edu.pe/Escritorio/archive/raw_merged_heart_dataset.csv')

datos_2[datos_2 == "?"] <- NA


datos_2$sex = factor(datos_2$sex, levels = c(0,1), labels = c('female','male'))
datos_2$cp = factor(datos_2$cp, levels = c(0,1,2,3))
datos_2$fbs = as.factor(datos_2$fbs)
datos_2$restecg = as.factor(datos_2$restecg)
datos_2$exang = as.factor(datos_2$exang)
datos_2$slope = as.factor(datos_2$slope)
datos_2$ca = as.factor(datos_2$ca)
datos_2$thal = as.factor(datos_2$thal)
datos_2$target = as.factor(datos_2$target)
datos_2$trestbps = as.numeric(datos_2$trestbps)
datos_2$chol = as.numeric(datos_2$chol)
datos_2$thalachh = as.numeric(datos_2$thalachh)

str(datos_2)
resultados_nuevos = 
  best_model_fit %>% 
  predict(new_data = datos_2) %>% 
  bind_cols(target = datos_2$target)

resultados_nuevos %>% 
  conf_mat(truth = target,
           estimate = .pred_class)
resultados_nuevos %>% 
  accuracy(truth = target,
           estimate = .pred_class)

rpart.plot::rpart.plot(extract_fit_engine(best_model_fit),cex = 0.3)
