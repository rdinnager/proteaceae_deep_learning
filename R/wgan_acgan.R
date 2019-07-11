library(reticulate)
library(keras)
library(tensorflow)
library(progress)
library(abind)
library(ggplot2)

#use_condaenv("r-tensorflow")
PIL <- import("PIL")
PIL$ImageFile$LOAD_TRUNCATED_IMAGES <- TRUE

# gpu_options <- tf$GPUOptions(per_process_gpu_memory_fraction = 0.3)
# # gpu_options <- tf$GPUOptions(allow_growth = TRUE)  # <- one or the other
# config <- tf$ConfigProto(gpu_options = gpu_options)
# k_set_session(tf$Session(config = config))

k_clear_session()

config = tensorflow::tf$ConfigProto(gpu_options = list(allow_growth = TRUE))
sess = tensorflow::tf$Session(config = config)
keras::k_set_session(session = sess)


#k_set_floatx("float16")
#k_set_epsilon(1e-4)

image_height <- 128 # Image height in pixels
image_width <- 128 # Image width in pixels
image_channels <- 3 # Number of color channels - here Red, Green and Blue
z_size <- 100 # Length of gaussian noise vector for generator input
batch_size <- 20
num_classes <- 3

epochs <- 1000
adam_lr <- 0.00005 
adam_beta_1 <- 0
D_ITERS <- 5

train_gen <- image_data_generator(preprocessing_function = function(x) (x - 127.5) / 127.5)
train_generator <- flow_images_from_directory(
  "data/images_small/train_dir",
  train_gen,
  target_size = c(image_height, image_width),
  batch_size = batch_size,
  class_mode = "sparse"
)

validation_generator <- flow_images_from_directory(
  "data/images_small/validation_dir",
  train_gen,
  target_size = c(image_height, image_width),
  batch_size = batch_size,
  class_mode = "sparse"
)

augmentation_generator <- flow_images_from_directory(
  "data/images_small/train_dir",
  train_gen,
  target_size = c(image_height, image_width),
  batch_size = 1
)
op <- par(mfrow = c(4, 4), pty = "s", mar = c(1, 0, 1, 0))
for(i in 1:16) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster((batch[[1]][1, , , ] + 1) / 2))
}
par(op)


## Start defining model. note: this is basically a straight port of the following python code to R: 
## https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html?r#wasserstein-gan

## define loss function
wasserstein_loss <- function(y_true, y_pred) {
  return(k_mean(y_true * y_pred))
}

## define weight clipping function
wgan_clip <- function(w) {
  k_clip(w, -0.01, 0.01)
}

create_discriminator <- function(image_height, image_width, image_channels, num_classes) {
  
  # weights are initlaized from normal distribution with below params
  weight_init <- initializer_random_normal(mean = 0, stddev = 0.02)

  discriminator_input <-layer_input(shape = c(image_height, image_width, image_channels), name = 'input_image')

  features <- discriminator_input %>%
    layer_conv_2d(filters = 32, kernel_size = 3, padding = 'same',
                  name='conv_1', kernel_initializer = weight_init#,
                  #kernel_constraint = wgan_clip
                  ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    layer_conv_2d(filters = 64, kernel_size = 3, padding = 'same',  
                  name='conv_2', kernel_initializer=weight_init,
                  #kernel_constraint = wgan_clip,
                  strides = 2) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    layer_conv_2d(filters = 128, kernel_size = 3, padding = 'same',  
                  name='conv_3', kernel_initializer=weight_init,
                  #kernel_constraint = wgan_clip,
                  strides = 2) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    layer_conv_2d(filters = 256, kernel_size = 3, padding = 'same',  
                  name='conv_4', kernel_initializer=weight_init#,
                  #kernel_constraint = wgan_clip
                  ) %>%
  layer_flatten()

  discriminator_output_is_fake  <- features %>%
    layer_dense(units = 1, name = "output_is_fake")

  discriminator_output_class = features %>%
    layer_dense(units = num_classes, activation = "softmax", name = "output_class")
    
  return(keras_model(discriminator_input, list(discriminator_output_is_fake, discriminator_output_class)))
}

create_generator <- function(z_size = 100, num_classes) {
  dict_len <- num_classes
  embedding_len <- z_size

  # weights are initlaized from normal distribution with below params
  weight_init <- initializer_random_normal(mean = 0, stddev = 0.02)

  # class#
  
  generator_input_class <- layer_input(shape = list(1), name = "generator_input_class")
  
  # encode class# to the same size as Z to use hadamard multiplication later on
  embedded_class <- generator_input_class %>%
    layer_embedding(dict_len, embedding_len, embeddings_initializer = "glorot_uniform") %>%
    layer_flatten(name = "embedded_class")

  # latent var
  generator_input_z <- layer_input(shape = list(z_size), name = "generator_input_z")

  # hadamard product
  h <- layer_multiply(c(generator_input_z, embedded_class), name = 'h')
  
  generator_output <- h %>%
    layer_dense(units = 1024) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 128 * image_height / 4 * image_width / 4) %>%
    layer_batch_normalization(momentum = 0.3) %>%
    layer_activation_leaky_relu() %>%
    layer_reshape(target_shape = c(image_height / 4, image_width / 4, 128)) %>%
    layer_upsampling_2d(size = 2) %>%
    layer_conv_2d(filters = 256, kernel_size = 5, padding = "same", kernel_initializer = weight_init) %>%
    layer_batch_normalization(momentum = 0.3) %>%
    layer_activation_leaky_relu() %>%
    layer_upsampling_2d(size = 2) %>%
    layer_conv_2d(filters = 128, kernel_size = 5, padding = "same", kernel_initializer = weight_init) %>%
    layer_batch_normalization(momentum = 0.3) %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filters = image_channels, kernel_size = 5, padding = "same",
                  activation = "tanh", name = "generator_output_image", kernel_initializer = weight_init)
    
  return(keras_model(list(generator_input_z, generator_input_class), generator_output))

}

discriminator <- create_discriminator(image_height, image_width, image_channels, num_classes)

# discriminator %>%
#   compile(optimizer = optimizer_adam(
#     beta_1 = adam_beta_1,
#     lr = adam_lr
#   ),
#   loss = list(wasserstein_loss, "sparse_categorical_crossentropy")
# )
discriminator %>%
  compile(optimizer = optimizer_rmsprop(
    lr = adam_lr
  ),
  loss = list(wasserstein_loss, "sparse_categorical_crossentropy")
  )

gan_input_z = layer_input(shape = list(z_size), name = 'gan_input_z')
gan_input_class = layer_input(shape = list(1), name = 'gan_input_class', dtype='int32')

generator <- create_generator(z_size, num_classes)

#freeze_weights(discriminator)

gan_output <- discriminator(generator(list(gan_input_z, gan_input_class)))

gan <- keras_model(list(gan_input_z, gan_input_class), gan_output)
# gan %>%
#   compile(optimizer = optimizer_adam(
#     beta_1 = adam_beta_1,
#     lr = adam_lr
#   ),
#   loss = list(wasserstein_loss, "sparse_categorical_crossentropy")
# )
gan %>%
  compile(optimizer = optimizer_rmsprop(
    lr = adam_lr
  ),
  loss = list(wasserstein_loss, "sparse_categorical_crossentropy")
  )


num_train <- train_generator$n
num_validation <- validation_generator$n 
iters <- 1
  
for(epoch in 1:epochs) {
  num_batches <- trunc(num_train/batch_size)
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
  epoch_gen_loss <- NULL
  epoch_disc_loss_fake <- NULL
  epoch_disc_loss_real <- NULL
  
  d_iters <- D_ITERS
  
  for(index in 1:num_batches){
    
    pb$tick()
    
    # if(iters %% 1000 < 5 | iters %% 500 == 0){ # 25 times in 1000, every 500th
    #   d_iters <- 100
    # } else{
    #   d_iters <- D_ITERS
    # }
  
    for(d_iter in 1:d_iters) {
      
      unfreeze_weights(discriminator)
      
      ## clip weights
      
      weights <- get_weights(discriminator)
      weights <- lapply(weights, function(x) {x[x > 0.01] <- 0.01; x[x < -0.01] <- -0.01; x})
      set_weights(discriminator, weights)
      
      real <- generator_next(train_generator)
      real_images <- real[[1]]
      real_classes <- real[[2]]
      
      rows <- nrow(real_images)
      
      disc_loss <- train_on_batch(discriminator, real_images, list(matrix(1, nrow = rows, ncol = 1),
                                                                   real_classes)) 
      epoch_disc_loss_real <- cbind(epoch_disc_loss_real, unlist(disc_loss))
      
      
      z <- matrix(rnorm(rows * z_size),
                  nrow = rows,
                  ncol = z_size)
      
      generated_classes <- as.integer(sample.int(num_classes, rows, replace = TRUE) - 1)
      generated_images <- generator %>% predict(list(z, generated_classes))
      
      disc_loss <- train_on_batch(discriminator, generated_images, list(matrix(-1, nrow = rows, ncol = 1),
                                                                   generated_classes)) 
      epoch_disc_loss_fake <- cbind(epoch_disc_loss_fake, unlist(disc_loss))
      
            #print("D!")
      
    }
    
    freeze_weights(discriminator)
    
    z <- matrix(rnorm(batch_size * z_size),
                nrow = batch_size,
                ncol = z_size)
    
    generated_classes <- as.integer(sample.int(num_classes, batch_size, replace = TRUE) - 1)
    #generated_images <- generator %>% predict(list(z, generated_classes))
    
    gen_loss <- train_on_batch(gan, list(z, generated_classes), list(matrix(1, nrow = batch_size, ncol = 1),
                                                                      generated_classes)) 
    epoch_gen_loss <- cbind(epoch_gen_loss, unlist(gen_loss))
    
    #print("G!")
    
    iters <- iters + 1  
    
    if(iters %% 100 == 0) {
      plot((-1*epoch_disc_loss_fake[1,] + epoch_disc_loss_real[1, ]) / 2, type = "l")
      
      z <- matrix(rnorm(batch_size * z_size),
                  nrow = batch_size,
                  ncol = z_size)
      
      generated_classes <- as.integer(sample.int(num_classes, rows, replace = TRUE) - 1)
      generated_images <- generator %>% predict(list(z, generated_classes))
      
      op <- par(mfrow = c(4, 4), pty = "s", mar = c(1, 0, 1, 0))
      for(i in 1:16) {
        plot(as.raster((generated_images[i, , , ] + 1) / 2))
        title(names(train_generator$class_indices)[generated_classes[i] + 1])
      }
      par(op)
    }
  }
  
  ### epoch report ####
  disc_loss <- (-1*epoch_disc_loss_fake[1,] + epoch_disc_loss_real[1, ]) / 2
  
  ## evaluate on validation data
  
  ## output images
  
  

  
}

############# WGAN-GP with eager execution attempt #1 ################

library(keras)
# make sure we use the tensorflow implementation of Keras
# this line has to be executed immediately after loading the library
use_implementation("tensorflow")

library(tensorflow)
# enable eager execution
# the argument device_policy is needed only when using a GPU
tfe_enable_eager_execution(device_policy = "silent")

