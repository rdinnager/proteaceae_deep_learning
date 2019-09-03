########### Score Matching Image Generator #############
## This is an implementation of density score estimation

library(keras)
use_implementation("tensorflow")

#k_set_floatx("float16")

library(tensorflow)

#gpu_options <- tf$GPUOptions(allow_growth = TRUE)  # <- one or the other
#config <- tf$ConfigProto(gpu_options = gpu_options)

#tfe_enable_eager_execution(device_policy = "silent", config = config)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)


image_height <- 64L # Image height in pixels
image_width <- 64L # Image width in pixels
image_channels <- 3 # Number of color channels - here Red, Green and Blue

epochs <- 1000
adam_lr <- 1e-4 
adam_beta1 <- 0.5
adam_beta2 <- 0.9

sigma_min <- 0.01
sigma_max <- 2
sigma_num <- 10L

batch_size <- 44L

image_files <- list.files("data/images_small/train_dir", recursive = TRUE, full.names = TRUE)
num_images <- length(image_files)

logit_transform <- function(image, lam = 1e-6) {
  image = lam + (1 - 2 * lam) * image
  tf$math$log(image) - tf$math$log1p(-image)
}

# image_file <- image_files[1]
load_an_image <- function(image_file, is_train) {
  
  image <- tf$io$read_file(image_file)
  image <- tf$image$decode_jpeg(image, channels = 3)
  
  input_image <- k_cast(image, tf$float32)
  
  if (is_train) {
    
    if (runif(1) > 0.5) {
      input_image <- tf$image$flip_left_right(input_image)
    }
    
    input_image <-
      tf$image$resize_images(input_image,
                             c(image_height, image_width),
                             align_corners = TRUE,
                             method = 2)
    
  } 
  
  input_image <-  (input_image / 256 * 255 + tf$random_uniform(k_int_shape(input_image))) / 256
  
  #logit_transform(input_image)
}

buffer_size <- length(image_files)
batches_per_epoch <- (buffer_size / batch_size) %>% round()

image_files <- sample(image_files)

image_files2 <- list(tf$convert_to_tensor(image_files))

train_dataset <- tensor_slices_dataset(image_files2) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_map(function(image) load_an_image(image, TRUE)) %>%
  dataset_batch(batch_size, drop_remainder = TRUE) %>%
  dataset_prefetch(-1)

iter <- make_iterator_one_shot(train_dataset)
test <- iterator_get_next(iter)
test


res_block <- function(filters,
                     size,
                     apply_dropout = FALSE,
                     name = "res_block") {
  
  keras_model_custom(name = NULL, function(self) {
    
    self$apply_dropout <- apply_dropout
    self$conv_1 <- layer_conv_2d(
      filters = filters,
      kernel_size = size,
      strides = 1,
      padding = "same",
      kernel_initializer = initializer_he_normal(),
      use_bias = FALSE
    )
    self$conv_2 <- layer_conv_2d(
      filters = filters,
      kernel_size = size,
      strides = 1,
      padding = "same",
      kernel_initializer = initializer_he_normal(),
      use_bias = FALSE
    )
    self$condinstancenorm <- layer_instance_norm_plus_params_as_input()
    
    if (self$apply_dropout) {
      self$dropout <- layer_dropout(rate = 0.5)
    }
    
    function(xs, mask = NULL, training = TRUE) {
      
      c(images, sigmas) %<-% xs
      x <- self$up_conv(x1) %>% self$condinstancenorm(training = training)
      if (self$apply_dropout) {
        x %>% self$dropout(training = training)
      }
      x %>% layer_activation("relu")
      concat <- k_concatenate(list(x, x2))
      concat
    }
  })
}




upsample <- function(filters,
                     size,
                     apply_dropout = FALSE,
                     name = "upsample") {
  
  keras_model_custom(name = NULL, function(self) {
    
    self$apply_dropout <- apply_dropout
    self$up_conv <- layer_conv_2d_transpose(
      filters = filters,
      kernel_size = size,
      strides = 2,
      padding = "same",
      kernel_initializer = initializer_random_normal(),
      use_bias = FALSE
    )
    self$condinstancenorm <- cont_cond_instance_layer_norm()
    if (self$apply_dropout) {
      self$dropout <- layer_dropout(rate = 0.5)
    }
    
    function(xs, mask = NULL, training = TRUE) {
      
      c(x1, x2) %<-% xs
      x <- self$up_conv(x1) %>% self$condinstancenorm(training = training)
      if (self$apply_dropout) {
        x %>% self$dropout(training = training)
      }
      x %>% layer_activation("relu")
      concat <- k_concatenate(list(x, x2))
      concat
    }
  })
}

scorenet <-
  function(image_height, image_width, image_channels, name = NULL) {
    
    
    keras_model_custom(name = name, function(self) {
      
      self$down1 <- 
        layer_conv_2d(
          filters = 32,
          kernel_size = c(5, 5),
          padding = "same"
        )
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$dropout1 <- layer_dropout(rate = 0.3)
      self$conv2 <-
        layer_conv_2d(
          filters = 64,
          kernel_size = c(5, 5),
          padding = "same",
          strides = c(2, 2)
        )
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$dropout2 <- layer_dropout(rate = 0.3)
      self$conv3 <- 
        layer_conv_2d(
          filters = 128, 
          kernel_size = c(5, 5), 
          padding = 'same',  
          strides = c(2, 2)
        )
      self$leaky_relu3 <- layer_activation_leaky_relu()
      self$dropout3 <- layer_dropout(rate = 0.3)
      self$conv4 <- 
        layer_conv_2d(
          filters = 256, 
          kernel_size = c(5, 5), 
          padding = 'same',
          strides = c(1, 1)
        )
      self$flatten <- layer_flatten()
      self$fc_critic <- layer_dense(units = 1)
      self$fc_classifier <- layer_dense(units = num_classes) ## don't apply softmax activation, will be taken care of in loss calculation
      
      function(inputs, mask = NULL, training = TRUE) {
        x <- inputs[[1]] %>% 
          self$conv1() %>%
          self$leaky_relu1() %>%
          self$dropout1(training = training) %>%
          self$conv2() %>%
          self$leaky_relu2() %>%
          self$dropout2(training = training) %>%
          self$conv3() %>%
          self$leaky_relu3() %>%
          self$dropout3(training = training) %>%
          self$conv4() %>%
          self$flatten() 
        
        list(x %>% self$fc_critic(), x %>% self$fc_classifier())
      }
    })
  }


sigmas <- seq(sigma_min, sigma_max, length.out = sigma_num) %>%
  matrix(nrow = sigma_num) %>%
  tf$convert_to_tensor(tf$float32)

for(epoch in epochs) {
  
  start <- Sys.time()
  total_gp <- 0
  total_loss_crit <- 0
  total_ac_loss <- 0
  seq_loss_crit <- NULL
  seq_gp <- NULL
  seq_ac_loss <- NULL
  iter <- make_iterator_one_shot(train_dataset)
  
  until_out_of_range({
    
    batch <- iterator_get_next(iter)
    
    noisy_batch <- tf$broadcast_to(batch, c(sigma_num, k_int_shape(batch))) + 
      tf$random_normal(c(sigma_num, k_int_shape(batch))) * tf$reshape(sigmas, c(sigma_num, 1L, 1L, 1L, 1L))
    
  })
  
}