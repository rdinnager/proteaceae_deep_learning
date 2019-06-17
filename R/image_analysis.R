library(keras)

image_dir <- "data/prot_images"
train_dir <- "train_dir"
validation_dir <- "validation_dir"
test_dir <- "test_dir"

#' Load and merge imagesets

prot_dirs <- list.dirs("images", recursive = TRUE, full.names = FALSE)
hakea_dirs <- prot_dirs[grep("Hakea", prot_dirs)]
banksia_dirs <- prot_dirs[grep("Banksia", prot_dirs)]
grevillea_dirs <- prot_dirs[grep("Grevillea", prot_dirs)]

test_prop <- 0.2
validation_prop <- 0.2

dir.create(file.path(image_dir))
dir.create(file.path(image_dir, train_dir))
dir.create(file.path(image_dir, validation_dir))
dir.create(file.path(image_dir, test_dir))

dir.create(file.path(image_dir, train_dir, "train_hakea"))
dir.create(file.path(image_dir, train_dir, "train_banksia"))
dir.create(file.path(image_dir, train_dir, "train_grevillea"))

dir.create(file.path(image_dir, validation_dir, "validation_hakea"))
dir.create(file.path(image_dir, validation_dir, "validation_banksia"))
dir.create(file.path(image_dir, validation_dir, "validation_grevillea"))

dir.create(file.path(image_dir, test_dir, "test_hakea"))
dir.create(file.path(image_dir, test_dir, "test_banksia"))
dir.create(file.path(image_dir, test_dir, "test_grevillea"))

hakea_files <- list.files(file.path("images", hakea_dirs), recursive = TRUE, full.names = TRUE)
hakea_test_files <- sample(hakea_files, floor(test_prop * length(hakea_files)))
hakea_validation_files <- sample(hakea_files[!hakea_files %in% hakea_test_files], floor(validation_prop * length(hakea_files)))
hakea_train_files <- hakea_files[!hakea_files %in% c(hakea_test_files, hakea_validation_files)]
file.copy(hakea_test_files,
          file.path(image_dir, test_dir, "test_hakea", basename(hakea_test_files)))
file.copy(hakea_validation_files,
          file.path(image_dir, validation_dir, "validation_hakea", basename(hakea_validation_files)))
file.copy(hakea_train_files,
          file.path(image_dir, train_dir, "train_hakea", basename(hakea_train_files)))

banksia_files <- list.files(file.path("images", banksia_dirs), recursive = TRUE, full.names = TRUE)
banksia_test_files <- sample(banksia_files, floor(test_prop * length(banksia_files)))
banksia_validation_files <- sample(banksia_files[!banksia_files %in% banksia_test_files], floor(validation_prop * length(banksia_files)))
banksia_train_files <- banksia_files[!banksia_files %in% c(banksia_test_files, banksia_validation_files)]
file.copy(banksia_test_files,
          file.path(image_dir, test_dir, "test_banksia", basename(banksia_test_files)))
file.copy(banksia_validation_files,
          file.path(image_dir, validation_dir, "validation_banksia", basename(banksia_validation_files)))
file.copy(banksia_train_files,
          file.path(image_dir, train_dir, "train_banksia", basename(banksia_train_files)))

grevillea_files <- list.files(file.path("images", grevillea_dirs), recursive = TRUE, full.names = TRUE)
grevillea_test_files <- sample(grevillea_files, floor(test_prop * length(grevillea_files)))
grevillea_validation_files <- sample(grevillea_files[!grevillea_files %in% grevillea_test_files], floor(validation_prop * length(grevillea_files)))
grevillea_train_files <- grevillea_files[!grevillea_files %in% c(grevillea_test_files, grevillea_validation_files)]
file.copy(grevillea_test_files,
          file.path(image_dir, test_dir, "test_grevillea", basename(grevillea_test_files)))
file.copy(grevillea_validation_files,
          file.path(image_dir, validation_dir, "validation_grevillea", basename(grevillea_validation_files)))
file.copy(grevillea_train_files,
          file.path(image_dir, train_dir, "train_grevillea", basename(grevillea_train_files)))


#' Setup a simple convolutional network

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(299, 299, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", input_shape = c(299, 299, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", input_shape = c(299, 299, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", input_shape = c(299, 299, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

#' Make image generators

train_datagen <- image_data_generator(rescale = 1/255,
                                      rotation_range = 30,
                                      width_shift_range = 0.1,
                                      height_shift_range = 0.1,
                                      shear_range = 0.1,
                                      zoom_range = 0.1,
                                      horizontal_flip = TRUE)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  file.path(image_dir, train_dir),
  train_datagen,
  target_size = c(299, 299),
  batch_size = 32,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  file.path(image_dir, validation_dir),
  validation_datagen,
  target_size = c(299, 299),
  batch_size = 32,
  class_mode = "categorical"
)

test_generator <- flow_images_from_directory(
  file.path(image_dir, test_dir),
  test_datagen,
  target_size = c(299, 299),
  batch_size = 32,
  class_mode = "categorical",
  shuffle = FALSE
)

#' Test out image augmentation
augmentation_generator <- flow_images_from_directory(
  file.path(image_dir, train_dir),
  train_datagen,
  target_size = c(299, 299),
  batch_size = 1
)
op <- par(mfrow = c(4, 4), pty = "s", mar = c(1, 0, 1, 0))
for(i in 1:16) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[[1]][1, , , ]))
}
par(op)

history <- model %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 200,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
  )

history_2<- model %>%
  fit_generator(
    train_generator,
    steps_per_epoch = 200,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 100
  )

model %>%
  save_model_hdf5("proteaceae_classifier_round_1_simple_convnet.h5")

results <- model %>%
  evaluate_generator(test_generator, steps = 112)
results

preds <- model %>%
  predict_generator(test_generator, steps = 112, verbose = 1)

pred_mat <- cbind(apply(preds, 1, which.max), test_generator$classes + 1)
conf_mat <- table(pred_mat[ , 1], pred_mat[ , 2])

conf_mat / rowSums(conf_mat)
sum(diag(conf_mat)) / sum(conf_mat)

