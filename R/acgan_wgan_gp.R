########### WGAN-GP #############
library(keras)
use_implementation("tensorflow")

#k_set_floatx("float16")

library(tensorflow)

gpu_options <- tf$GPUOptions(allow_growth = TRUE)  # <- one or the other
config <- tf$ConfigProto(gpu_options = gpu_options)

tfe_enable_eager_execution(device_policy = "silent", config = config)

library(tfdatasets)


image_height <- 64L # Image height in pixels
image_width <- 64L # Image width in pixels
image_channels <- 3 # Number of color channels - here Red, Green and Blue
z_size <- 100 # Length of gaussian noise vector for generator input
num_classes <- 3

epochs <- 1000
adam_lr <- 1e-4 
adam_beta1 <- 0.5
adam_beta2 <- 0.9

n_critic <- 5

batch_size <- 44L

image_files <- list.files("data/images_small/train_dir", recursive = TRUE, full.names = TRUE)
num_images <- length(image_files)

# image_file <- image_files[1]
load_an_image <- function(image_file, cats, is_train) {
  
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
  
  input_image <- (input_image / 127.5) - 1
  
  cats <- k_cast(cats, tf$int32)
  
  list(input_image, cats)
}

buffer_size <- length(image_files)
batches_per_epoch <- (buffer_size / batch_size) %>% round()

gradient_penalty_weight <- 10
ac_weight_critic <- 1.0
ac_weight_generator <- 0.1

image_files <- sample(image_files)
labels <- ifelse(grepl("banksia", image_files), 0L,
                 ifelse(grepl("grevillea", image_files), 1L,
                        2L))
image_files2 <- list(tf$convert_to_tensor(image_files), tf$convert_to_tensor(labels))

train_dataset <- tensor_slices_dataset(image_files2) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_map(function(image, label) load_an_image(image, label, TRUE)) %>%
  dataset_batch(batch_size, drop_remainder = TRUE) %>%
  dataset_prefetch_to_device("/gpu:0")

iter <- make_iterator_one_shot(train_dataset)
iterator_get_next(iter)


critic <-
  function(image_height, image_width, image_channels, num_classes, name = NULL) {
    
    
    keras_model_custom(name = name, function(self) {
      
      self$conv1 <- 
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
        x <- inputs %>% 
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
  
  
generator <- 
  function(z_size = 100, num_classes, momentum = 0.3, name = NULL) {
    
  keras_model_custom(name = name, function(self) {
    
    self$fc1 <- layer_dense(units = 128 * image_height / 4 * image_width / 4, use_bias = FALSE)
    self$batchnorm1 <- layer_batch_normalization(momentum = momentum)
    self$leaky_relu1 <- layer_activation_leaky_relu()
    self$upsample1 <- layer_upsampling_2d(size = 2)
    self$conv1 <-
      layer_conv_2d(
        filters = 256, 
        kernel_size = c(5,5), 
        padding = "same",
        use_bias = FALSE
      )
    self$batchnorm2 <- layer_batch_normalization(momentum = momentum)
    self$leaky_relu2 <- layer_activation_leaky_relu()
    self$upsample2 <- layer_upsampling_2d(size = 2)
    self$conv2 <-
      layer_conv_2d(
        filters = 128, 
        kernel_size = c(5, 5), 
        padding = "same",
        use_bias = FALSE
      )
    self$batchnorm3 <- layer_batch_normalization(momentum = momentum)
    self$leaky_relu3 <- layer_activation_leaky_relu()
    self$conv3 <-
      layer_conv_2d(
        filters = image_channels, 
        kernel_size = c(5, 5), 
        padding = "same",
        activation = "tanh",
        use_bias = FALSE
      )
    self$embedding <- layer_embedding(input_dim = num_classes, output_dim = z_size, 
                                      embeddings_initializer = "glorot_uniform")
    self$flatten <- layer_flatten()
   
    
    function(inputs, mask = NULL, training = TRUE) {
      
      embedded_class <- inputs[[2]] %>%
        self$embedding() %>%
        self$flatten()
      
      h <- layer_multiply(c(inputs[[1]], embedded_class))
      
      h %>%
        self$fc1() %>%
        self$batchnorm1(training = training) %>%
        self$leaky_relu1() %>%
        k_reshape(shape = c(-1, image_height / 4, image_width / 4, 128)) %>%
        self$upsample1() %>%
        self$conv1() %>%
        self$batchnorm2(training = training) %>%
        self$leaky_relu2() %>%
        self$upsample2() %>%
        self$conv2() %>%
        self$batchnorm3(training = training) %>%
        self$leaky_relu3() %>%
        self$conv3()
    }
  })
  
  
  }

generator <- generator(z_size, num_classes)
critic <- critic(image_height, image_width, image_channels, num_classes)

l_norm <- function(gradi) {
  
  ## Gradient penalty
  # first get the gradients:
  #   assuming: - that y_pred has dimensions (batch_size, 1)
  #             - averaged_samples has dimensions (batch_size, nbr_features)
  # gradients afterwards has dimension (batch_size, nbr_features), basically
  # a list of nbr_features-dimensional gradient vectors
  #gradients <- tf$gradients(averaged_output, averaged_samples)[0]
  # compute the euclidean norm by squaring ...
  gradients_sqr <- tf$square(gradi)
  #   ... summing over the rows ...
  gradients_sqr_sum <- tf$reduce_sum(gradients_sqr,
                                     axis =  c(1L, 2L, 3L))
  #   ... and sqrt
  gradient_l2_norm <- tf$sqrt(gradients_sqr_sum)
  # compute lambda * (1 - ||grad||)^2 still for each single sample
  gradient_penalty <- gradient_penalty_weight * tf$square(gradient_l2_norm - 1) ## whoah error here, was 1 - norm, should be norm - 1!
  # return the mean as loss over all the batch samples
  grad_loss <- tf$reduce_mean(gradient_penalty)
  
  grad_loss
}

critic_optimizer <- tf$train$AdamOptimizer(1e-4, beta1 = 0.5, beta2 = 0.9)
generator_optimizer <- tf$train$AdamOptimizer(1e-4, beta1 = 0.5, beta2 = 0.9)

## function to take random weighted average of two tensors
random_weighted_average <- function(tens_1, tens_2) {
  shape <- tf$concat(list(tf$shape(tens_1)[1, drop = FALSE], 
                          tf$ones(tens_1$shape$ndims - 1, "int32")), 0L)
  
  alpha = tf$random_uniform(shape=shape, minval=0., maxval=1.)
  averaged_samples = tens_1 + alpha * (tens_2 - tens_1)
  averaged_samples$set_shape(tens_1$shape)
  averaged_samples
}

generate_and_save_images <- function(model, epoch, test_input, folder) {
  predictions <- model(test_input, training = FALSE)
  png(paste0(folder, "/images_epoch_", epoch, ".png"))
  par(mfcol = c(5, 5))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:25) {
    img <- as.array(predictions)[i, , , ]
    plot(as.raster((img + 1) / 2))
    title(paste0("class = ", as.array(test_input[[2]])[i]))
  }
  dev.off()
}

n_critic <- 5
folder <- "AC_pretrained"
num_epochs <- 3000
checkpoint_directory <- "checkpoints_AC_pretrained"
checkpoint_prefix <- file.path(checkpoint_directory, "AC_pretrained")

checkpoint <- tf$train$Checkpoint(generator = generator, critic = critic,
                                  generator_optimizer = generator_optimizer,
                                  critic_optimizer = critic_optimizer)
status <- checkpoint$restore(tf$train$latest_checkpoint(checkpoint_directory))

# pretrain_epochs <- 100
# 
# for(pretrain_epoch in (1:pretrain_epochs)) {
#   start <- Sys.time()
#   total_ac_loss <- 0
#   total_ac_acc <- 0
#   iter <- make_iterator_one_shot(train_dataset)
#   until_out_of_range({
#     batch <- iterator_get_next(iter)
#     with(tf$GradientTape() %as% crit_tape_ac, {
#       crit_real_output <- critic(batch[[1]], training = TRUE)
#       crit_ac_loss <- tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels = 
#                                                                                       batch[[2]],
#                                                                                     logits =
#                                                                                       crit_real_output[[2]]))
#     })
#     gradients_of_critic_ac <-
#       crit_tape_ac$gradient(crit_ac_loss, critic$variables)
#     
#     critic_optimizer$apply_gradients(purrr::transpose(
#       list(gradients_of_critic_ac, critic$variables)
#     ))
#     
#     #print("Critic!")
#     
#     total_ac_loss <- total_ac_loss + crit_ac_loss
#     
#   })
#   
#   cat("Time for epoch ", pretrain_epoch, ": ", Sys.time() - start, "\n")
#   cat("AC loss: ", total_ac_loss$numpy() / batches_per_epoch, "\n\n")
#   
#   if(pretrain_epoch %% 10 == 0) {
#     checkpoint$save(file_prefix = checkpoint_prefix)
#   }
# }

########## Start GAN training #################

for (epoch in (1591:num_epochs)) {
  start <- Sys.time()
  total_gp <- 0
  total_loss_crit <- 0
  total_ac_loss <- 0
  seq_loss_crit <- NULL
  seq_gp <- NULL
  seq_ac_loss <- NULL
  iter <- make_iterator_one_shot(train_dataset)
  
  until_out_of_range({
    for(i in 1:n_critic) {
      batch <- iterator_get_next(iter)
      with(tf$GradientTape() %as% crit_tape, {
        
        noise <- k_random_normal(c(dim(batch[[1]])[1], z_size))
        generated_labels <- tf$random_uniform(c(batch_size, 1L), dtype = "int32", maxval = 3L)
        generated_images <- generator(list(noise, generated_labels))
        crit_real_output <- critic(batch[[1]], training = TRUE)
        crit_generated_output <-
          critic(generated_images, training = TRUE)
        
        averaged_samples <- random_weighted_average(batch[[1]], generated_images)
        
        with(tf$GradientTape() %as% grad, {
          
          grad$watch(averaged_samples)
          averaged_output <- critic(averaged_samples, training = TRUE)
          
        })
        crit_tape$watch(averaged_samples)
        grads <- grad$gradient(averaged_output[[1]], averaged_samples)
        
        
        w_loss <- tf$reduce_mean(crit_generated_output[[1]]) - tf$reduce_mean(crit_real_output[[1]])
        
        grad_penalty <- l_norm(grads)
        
        crit_ac_loss <- tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels = 
                                                                         tf$concat(list(batch[[2]], tf$squeeze(generated_labels)), 0L),
                                                                       logits =
                                                                         tf$concat(list(crit_real_output[[2]], crit_generated_output[[2]]), 0L)))
        
        crit_loss <- w_loss + gradient_penalty_weight * grad_penalty + ac_weight_critic * crit_ac_loss
        
      })
      
      gradients_of_critic <-
        crit_tape$gradient(crit_loss, critic$variables)
      
      critic_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_critic, critic$variables)
      ))
      
      #print("Critic!")
      
      
      total_loss_crit <- total_loss_crit + (-w_loss)
      total_gp <- total_gp + grad_penalty
      total_ac_loss <- total_ac_loss + crit_ac_loss
      
      seq_loss_crit <- c(seq_loss_crit, -(-w_loss))
      seq_gp <- c(seq_gp, grad_penalty)
      seq_ac_loss <- c(seq_ac_loss, crit_ac_loss)
      
    }
    
    batch <- iterator_get_next(iter)
    
    with(tf$GradientTape() %as% gen_tape, { with(tf$GradientTape() %as% crit_tape, {
      
      noise <- k_random_normal(c(dim(batch[[1]])[1], z_size))
      generated_labels <- tf$random_uniform(c(batch_size, 1L), dtype = "int32", maxval = 3L)
      generated_images <- generator(list(noise, generated_labels))
      crit_real_output <- critic(batch[[1]], training = TRUE)
      crit_generated_output <-
        critic(generated_images, training = TRUE)
      
      averaged_samples <- random_weighted_average(batch[[1]], generated_images)
      
      with(tf$GradientTape() %as% grad, {
        
        grad$watch(averaged_samples)
        averaged_output <- critic(averaged_samples, training = TRUE)
        
      })
      crit_tape$watch(averaged_samples)
      grads <- grad$gradient(averaged_output, averaged_samples)
      
      
      w_loss <- tf$reduce_mean(crit_generated_output[[1]]) - tf$reduce_mean(crit_real_output[[1]])
      
      grad_penalty <- l_norm(grads)
      
      crit_ac_loss <- tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels = 
                                                                                      tf$concat(list(batch[[2]], tf$squeeze(generated_labels)), 0L),
                                                                                    logits =
                                                                                      tf$concat(list(crit_real_output[[2]], crit_generated_output[[2]]), 0L)))
      
      crit_loss <- w_loss + gradient_penalty_weight * grad_penalty + ac_weight_critic * crit_ac_loss  
      
      w_loss_gen <- -tf$reduce_mean(crit_generated_output[[1]])
      
      gen_ac_loss <- tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels = 
                                                                                      tf$squeeze(generated_labels),
                                                                                    logits =
                                                                                      crit_generated_output[[2]]))
      gen_loss <- w_loss_gen + ac_weight_generator * gen_ac_loss
    }) })
    
    gradients_of_generator <-
      gen_tape$gradient(gen_loss, generator$variables)
    gradients_of_critic <-
      crit_tape$gradient(crit_loss, critic$variables)
    
    generator_optimizer$apply_gradients(purrr::transpose(
      list(gradients_of_generator, generator$variables)
    ))
    critic_optimizer$apply_gradients(purrr::transpose(
      list(gradients_of_critic, critic$variables)
    ))
    
    #print("Critic!")
    #print("Generator!")
    
    total_loss_crit <- total_loss_crit + (-w_loss)
    total_gp <- total_gp + grad_penalty
    total_ac_loss <- total_ac_loss + crit_ac_loss
    
    seq_loss_crit <- c(seq_loss_crit, -(-w_loss))
    seq_gp <- c(seq_gp, grad_penalty)
    seq_ac_loss <- c(seq_ac_loss, crit_ac_loss)
    
    
  })
  
  cat("Time for epoch ", epoch, ": ", Sys.time() - start, "\n")
  cat("Gradient Penalty: ", total_gp$numpy() / batches_per_epoch, "\n")
  cat("Critic loss: ", total_loss_crit$numpy() / batches_per_epoch, "\n")
  cat("AC loss: ", total_ac_loss$numpy() / batches_per_epoch, "\n\n")
  
  if(epoch %% 100 == 0) {
    checkpoint$save(file_prefix = checkpoint_prefix)
    
  }
  if(epoch %% 10 == 0) {
    generate_and_save_images(generator,
                             epoch,
                             list(k_random_normal(c(25L, z_size)), 
                                  tf$random_uniform(c(25L, 1L), dtype = "int32", maxval = 3L)),
                             folder)
  }

}

save_model_weights_hdf5(generator, "saved_models/generator_1590_epochs_weights.hd5")
save_model_weights_hdf5(critic, "saved_models/critic_1590_epochs_weights.hd5")

generator2 <- generator

load_model_weights_hdf5(generator, "saved_models/generator_1590_epochs_weights.hd5")

##### Explore latent space ##########

checkpoint_directory <- "checkpoints_AC_pretrained"
checkpoint_prefix <- file.path(checkpoint_directory, "AC_pretrained")

checkpoint <- tf$train$Checkpoint(generator = generator, critic = critic,
                                  generator_optimizer = generator_optimizer,
                                  critic_optimizer = critic_optimizer)
status <- checkpoint$restore(tf$train$latest_checkpoint(checkpoint_directory))

noise <- k_random_normal(c(25, z_size))
random_labels <- matrix(0L, nrow = 25, ncol = 1)
random_images <- generator(list(noise, random_labels))

par(mfcol = c(5, 5))
par(mar = c(0, 0, 0, 0),
    xaxs = 'i',
    yaxs = 'i')
for (i in 1:25) {
  img <- as.array(random_images)[i, , , ]
  plot(as.raster((img + 1) / 2))
  title(paste0("image no. = ", i))
}

save_latents <- as.matrix(noise)[c(2, 22, 3, 12),]

x <- save_latents
interpolate_points <- function(x, num_interp = 75, loop_back = TRUE) {
  num_gaps <- nrow(x) - 1
  intermediates <- list()
  for (i in seq_len(num_gaps)) {
    slope = x[i + 1, ] - x[i, ]
    intermediates[[i]] <- t(x[i, ] + sapply(seq(0, 1, length.out = num_interp), function(x) x * slope))
  }
  if(loop_back) {
    slope = x[1, ] - x[num_gaps + 1, ]
    intermediates[[num_gaps + 1]] <- t(x[num_gaps + 1, ] + sapply(seq(0, 1, length.out = num_interp), function(x) x * slope))
  }
  interpolation <- do.call(rbind, intermediates)
}

interp_latent <- interpolate_points(save_latents)

image_set <- generator(list(tf$convert_to_tensor(interp_latent, "float32"), matrix(0L, nrow = nrow(interp_latent), ncol = 1)))

library(imager)
image_array <- aperm(as.array(image_set), c(3, 2, 1, 4))
test <- as.cimg(image_array)
play(test, loop = TRUE)

save.video(test, "test.mpeg")


noise <- k_random_normal(c(25, z_size))
random_labels <- matrix(1L, nrow = 25, ncol = 1)
random_images <- generator(list(noise, random_labels))

par(mfcol = c(5, 5))
par(mar = c(0.5, 0.5, 0.5, 0.5),
    xaxs = 'i',
    yaxs = 'i')
for (i in 1:25) {
  img <- as.array(random_images)[i, , , ]
  plot(as.raster((img + 1) / 2))
  title(paste0("image no. = ", i))
}

save_latents <- list()
save_latents[[1]] <- as.matrix(noise)[24, ]
save_latents[[2]] <- as.matrix(noise)[14, ]
save_latents[[3]] <- as.matrix(noise)[23, ]
save_latents[[4]] <- as.matrix(noise)[13, ]
save_latents[[5]] <- as.matrix(noise)[6, ]
save_latents[[6]] <- as.matrix(noise)[9, ]
save_latents[[7]] <- as.matrix(noise)[22, ]
save_latents[[8]] <- as.matrix(noise)[1, ]

save_latents_mat <- do.call(rbind, save_latents)

latent_interp <- interpolate_points(save_latents_mat, num_interp = 50)

with(tf$device("cpu:0"), {
  image_set <- generator(list(tf$convert_to_tensor(latent_interp, "float32"), matrix(0L, nrow = nrow(latent_interp), ncol = 1)))  
})

image_array <- aperm(as.array(image_set) * 127.5 + 127.5, c(3, 2, 1, 4))
test <- as.cimg(image_array)
play(test, loop = TRUE, normalise = FALSE)

save.video(test, "test2.mpeg")

#### Array figure latent interpolation ##############

interpolate_points <- function(x, num_interp = 75, loop_back = TRUE) {
  num_gaps <- nrow(x) - 1
  intermediates <- list()
  for (i in seq_len(num_gaps)) {
    slope = x[i + 1, ] - x[i, ]
    intermediates[[i]] <- t(x[i, ] + sapply(seq(0, 1, length.out = num_interp), function(x) x * slope))
  }
  if(loop_back) {
    slope = x[1, ] - x[num_gaps + 1, ]
    intermediates[[num_gaps + 1]] <- t(x[num_gaps + 1, ] + sapply(seq(0, 1, length.out = num_interp), function(x) x * slope))
  }
  interpolation <- do.call(rbind, intermediates)
}

noise_list <- replicate(9, as.matrix(k_random_normal(c(4, z_size))), simplify = FALSE)

interp_list <- lapply(noise_list, interpolate_points, num_interp = 40)

num_rows <- nrow(interp_list[[1]])

label_list <- list()
label_list[1:3] <- replicate(3, matrix(2L, nrow = num_rows, ncol = 1), simplify = FALSE)
label_list[4:6] <- replicate(3, matrix(0L, nrow = num_rows, ncol = 1), simplify = FALSE)
label_list[7:9] <- replicate(3, matrix(1L, nrow = num_rows, ncol = 1), simplify = FALSE)

library(stringr)
fold <- "animations/temp_pngs_1"
## loop through and construct plots
for(i in 1:num_rows) {
  noise_mat <- t(sapply(interp_list, function(x) x[i, ]))
  label_mat <- matrix(sapply(label_list, function(x) x[i, ]), ncol = 1)
  image_gen <- generator(list(tf$convert_to_tensor(noise_mat, "float32"), label_mat))
  png(file.path(fold, paste0("image_", str_pad(i, 3, pad = "0"), ".png")), height = 64*3, width = 64*3)
  par(mfcol = c(3, 3))
  par(mar = c(0.0, 0.0, 0.0, 0.0),
      xaxs = 'i',
      yaxs = 'i')
  for (j in 1:9) {
    img <- as.array(image_gen)[j, , , ]
    plot(as.raster((img + 1) / 2))
  }
  dev.off()
}

library(magick)
pngs <- image_read(list.files(fold, full.names = TRUE))

animation <- image_animate(pngs, fps = 20)
#image_write(animation, "animations/proteaceae_animation_2.gif")

image_write_gif(animation, "animations/proteaceae_animation_6.gif", delay = 0.1)
