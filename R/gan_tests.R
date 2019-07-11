#' Practice GANs in tensorflow eager using simpler images.
########### Simple Vanilla GAN #############
library(keras)
use_implementation("tensorflow")

library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)

np <- import("numpy")

kuzushiji <- np$load("kmnist-train-imgs.npz")
kuzushiji <- kuzushiji$get("arr_0")

train_images <- kuzushiji %>% 
  k_expand_dims() %>%
  k_cast(dtype = "float32")

# normalize images to [-1, 1] because the generator uses tanh activation
train_images <- (train_images - 127.5) / 127.5

buffer_size <- 60000
batch_size <- 256
batches_per_epoch <- (buffer_size / batch_size) %>% round()

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

## Generator
generator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      
      self$fc1 <- layer_dense(units = 7 * 7 * 64, use_bias = FALSE)
      self$batchnorm1 <- layer_batch_normalization()
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$conv1 <-
        layer_conv_2d_transpose(
          filters = 64,
          kernel_size = c(5, 5),
          strides = c(1, 1),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm2 <- layer_batch_normalization()
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$conv2 <-
        layer_conv_2d_transpose(
          filters = 32,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm3 <- layer_batch_normalization()
      self$leaky_relu3 <- layer_activation_leaky_relu()
      self$conv3 <-
        layer_conv_2d_transpose(
          filters = 1,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE,
          activation = "tanh"
        )
      
      function(inputs, mask = NULL, training = TRUE) {
        self$fc1(inputs) %>%
          self$batchnorm1(training = training) %>%
          self$leaky_relu1() %>%
          k_reshape(shape = c(-1, 7, 7, 64)) %>%
          self$conv1() %>%
          self$batchnorm2(training = training) %>%
          self$leaky_relu2() %>%
          self$conv2() %>%
          self$batchnorm3(training = training) %>%
          self$leaky_relu3() %>%
          self$conv3()
      }
    })
  }

## Discriminator
discriminator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      
      self$conv1 <- layer_conv_2d(
        filters = 64,
        kernel_size = c(5, 5),
        strides = c(2, 2),
        padding = "same"
      )
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$dropout <- layer_dropout(rate = 0.3)
      self$conv2 <-
        layer_conv_2d(
          filters = 128,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same"
        )
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$flatten <- layer_flatten()
      self$fc1 <- layer_dense(units = 1)
      
      function(inputs, mask = NULL, training = TRUE) {
        inputs %>% self$conv1() %>%
          self$leaky_relu1() %>%
          self$dropout(training = training) %>%
          self$conv2() %>%
          self$leaky_relu2() %>%
          self$flatten() %>%
          self$fc1()
      }
    })
  }

generator <- generator()
discriminator <- discriminator()

# https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun
generator$call = tf$contrib$eager$defun(generator$call)
discriminator$call = tf$contrib$eager$defun(discriminator$call)

discriminator_loss <- function(real_output, generated_output) {
  real_loss <- tf$losses$sigmoid_cross_entropy(
    multi_class_labels = k_zeros_like(real_output),
    logits = real_output)
  generated_loss <- tf$losses$sigmoid_cross_entropy(
    multi_class_labels = k_ones_like(generated_output),
    logits = generated_output)
  real_loss + generated_loss
}

generator_loss <- function(generated_output) {
  tf$losses$sigmoid_cross_entropy(
    tf$zeros_like(generated_output),
    generated_output)
}

discriminator_optimizer <- tf$train$AdamOptimizer(1e-4)
generator_optimizer <- tf$train$AdamOptimizer(1e-4)


generate_and_save_images <- function(model, epoch, test_input, folder) {
  predictions <- model(test_input, training = FALSE)
  png(paste0(folder, "/images_epoch_", epoch, ".png"))
  par(mfcol = c(5, 5))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:25) {
    img <- predictions[i, , , 1]
    img <- t(apply(img, 2, rev))
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((0:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}

noise_dim <- 100

train <- function(train_dataset, epochs, noise_dim, folder) {
  for (epoch in seq_len(num_epochs)) {
    start <- Sys.time()
    total_loss_gen <- 0
    total_loss_disc <- 0
    iter <- make_iterator_one_shot(train_dataset)
    
    until_out_of_range({
      batch <- iterator_get_next(iter)
      noise <- k_random_normal(c(batch_size, noise_dim))
      with(tf$GradientTape() %as% gen_tape, { with(tf$GradientTape() %as% disc_tape, {
        generated_images <- generator(noise)
        disc_real_output <- discriminator(batch, training = TRUE)
        disc_generated_output <-
          discriminator(generated_images, training = TRUE)
        gen_loss <- generator_loss(disc_generated_output)
        disc_loss <-
          discriminator_loss(disc_real_output, disc_generated_output)
      }) })
      
      gradients_of_generator <-
        gen_tape$gradient(gen_loss, generator$variables)
      gradients_of_discriminator <-
        disc_tape$gradient(disc_loss, discriminator$variables)
      
      generator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_generator, generator$variables)
      ))
      discriminator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_discriminator, discriminator$variables)
      ))
      
      total_loss_gen <- total_loss_gen + gen_loss
      total_loss_disc <- total_loss_disc + disc_loss
      
    })
    
    cat("Time for epoch ", epoch, ": ", Sys.time() - start, "\n")
    cat("Generator loss: ", total_loss_gen$numpy() / batches_per_epoch, "\n")
    cat("Discriminator loss: ", total_loss_disc$numpy() / batches_per_epoch, "\n\n")
    if (epoch %% 10 == 0)
      generate_and_save_images(generator,
                               epoch,
                               k_random_normal(c(25, noise_dim)),
                               folder)
    
  }
}

num_epochs <- 500

train(train_dataset, num_epochs, noise_dim, "vanilla_gan_test")



########### WGAN-GP #############
library(keras)
use_implementation("tensorflow")

library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

library(tfdatasets)

np <- import("numpy")

kuzushiji <- np$load("kmnist-train-imgs.npz")
kuzushiji <- kuzushiji$get("arr_0")

train_images <- kuzushiji %>% 
  k_expand_dims() %>%
  k_cast(dtype = "float32")

# normalize images to [-1, 1] because the generator uses tanh activation
train_images <- (train_images - 127.5) / 127.5

buffer_size <- 60000
batch_size <- 256
batches_per_epoch <- (buffer_size / batch_size) %>% round()
gradient_penalty_weight <- 10

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

## Generator
generator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      
      self$fc1 <- layer_dense(units = 7 * 7 * 64, use_bias = FALSE)
      self$batchnorm1 <- layer_batch_normalization()
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$conv1 <-
        layer_conv_2d_transpose(
          filters = 64,
          kernel_size = c(5, 5),
          strides = c(1, 1),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm2 <- layer_batch_normalization()
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$conv2 <-
        layer_conv_2d_transpose(
          filters = 32,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm3 <- layer_batch_normalization()
      self$leaky_relu3 <- layer_activation_leaky_relu()
      self$conv3 <-
        layer_conv_2d_transpose(
          filters = 1,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE,
          activation = "tanh"
        )
      
      function(inputs, mask = NULL, training = TRUE) {
        self$fc1(inputs) %>%
          self$batchnorm1(training = training) %>%
          self$leaky_relu1() %>%
          k_reshape(shape = c(-1, 7, 7, 64)) %>%
          self$conv1() %>%
          self$batchnorm2(training = training) %>%
          self$leaky_relu2() %>%
          self$conv2() %>%
          self$batchnorm3(training = training) %>%
          self$leaky_relu3() %>%
          self$conv3()
      }
    })
  }

## Discriminator
critic <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {
      
      self$conv1 <- layer_conv_2d(
        filters = 64,
        kernel_size = c(5, 5),
        strides = c(2, 2),
        padding = "same"
      )
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$dropout <- layer_dropout(rate = 0.3)
      self$conv2 <-
        layer_conv_2d(
          filters = 128,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same"
        )
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$flatten <- layer_flatten()
      self$fc1 <- layer_dense(units = 1)
      
      function(inputs, mask = NULL, training = TRUE) {
        inputs %>% self$conv1() %>%
          self$leaky_relu1() %>%
          self$dropout(training = training) %>%
          self$conv2() %>%
          self$leaky_relu2() %>%
          self$flatten() %>%
          self$fc1()
      }
    })
  }

generator <- generator()
critic <- critic()

# https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun
#generator$call = tf$contrib$eager$defun(generator$call)
#critic$call = tf$contrib$eager$defun(critic$call)

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
    img <- predictions[i, , , 1]
    img <- t(apply(img, 2, rev))
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((0:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}

noise_dim <- 100
n_critic <- 5

train <- function(train_dataset, epochs, noise_dim, folder) {
  for (epoch in seq_len(num_epochs)) {
    start <- Sys.time()
    total_gp <- 0
    total_loss_crit <- 0
    seq_loss_crit <- NULL
    seq_gp <- NULL
    iter <- make_iterator_one_shot(train_dataset)
    
    until_out_of_range({
      for(i in 1:n_critic) {
        batch <- iterator_get_next(iter)
        with(tf$GradientTape() %as% crit_tape, {
          
          noise <- k_random_normal(c(dim(batch)[1], noise_dim))
          generated_images <- generator(noise)
          crit_real_output <- critic(batch, training = TRUE)
          crit_generated_output <-
            critic(generated_images, training = TRUE)
          
          averaged_samples <- random_weighted_average(batch, generated_images)
          
          with(tf$GradientTape() %as% grad, {
            
            averaged_samples <- random_weighted_average(batch, generated_images)
            grad$watch(averaged_samples)
            averaged_output <- critic(averaged_samples, training = TRUE)
            
          })
          crit_tape$watch(averaged_samples)
          grads <- grad$gradient(averaged_output, averaged_samples)
          
         
          w_loss <- tf$reduce_mean(crit_generated_output) - tf$reduce_mean(crit_real_output)
          
          grad_penalty <- l_norm(grads)
          
          crit_loss <- w_loss + gradient_penalty_weight * grad_penalty
          
        })
        
        gradients_of_critic <-
          crit_tape$gradient(crit_loss, critic$variables)
        
        critic_optimizer$apply_gradients(purrr::transpose(
          list(gradients_of_critic, critic$variables)
        ))
        
        
        total_loss_crit <- total_loss_crit + (-w_loss)
        total_gp <- total_gp + grad_penalty
        
        seq_loss_crit <- c(seq_loss_crit, -(-w_loss))
        seq_gp <- c(seq_gp, grad_penalty)
        
      }
      
      batch <- iterator_get_next(iter)
      noise <- k_random_normal(c(dim(batch)[1], noise_dim))
      with(tf$GradientTape() %as% gen_tape, { with(tf$GradientTape() %as% crit_tape, {
        
        generated_images <- generator(noise)
        crit_real_output <- critic(batch, training = TRUE)
        crit_generated_output <-
          critic(generated_images, training = TRUE)
        
        with(tf$GradientTape() %as% grad, {
          
          averaged_samples <- random_weighted_average(batch, generated_images)
          averaged_output <- critic(averaged_samples, training = TRUE)
          grad$watch(averaged_samples)
        })
        
        grads <- grad$gradient(averaged_output, averaged_samples)
        
        w_loss <- tf$reduce_mean(crit_generated_output) - tf$reduce_mean(crit_real_output)
        
        grad_penalty <- l_norm(grads)
        
        crit_loss <- w_loss + gradient_penalty_weight * grad_penalty
        
        gen_loss <- -tf$reduce_mean(crit_generated_output)
        
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
      
      total_loss_crit <- total_loss_crit + (-w_loss)
      total_gp <- total_gp + grad_penalty
      
      seq_loss_crit <- c(seq_loss_crit, -(-w_loss))
      seq_gp <- c(seq_gp, grad_penalty)
      
      
    })
    
    
    
    
    
    
    
    
    cat("Time for epoch ", epoch, ": ", Sys.time() - start, "\n")
    cat("Gradient Penalty: ", total_gp$numpy() / batches_per_epoch, "\n")
    cat("Critic loss: ", total_loss_crit$numpy() / batches_per_epoch, "\n\n")
    if (epoch %% 10 == 0)
      generate_and_save_images(generator,
                               epoch,
                               k_random_normal(c(25, noise_dim)),
                               folder)
    
  }
}

num_epochs <- 170*4

train(train_dataset, num_epochs, noise_dim, "wgan_gp_test")


### Testy tests ####

x <- tf$constant(3.0)
with (tf$GradientTape() %as% g, {
  g$watch(x)
  with (tf$GradientTape() %as% gg, {
    gg$watch(x)
    y <- x * x
  })
  dy_dx = gg$gradient(y, x)     # Will compute to 6.0
})
d2y_dx2 = g$gradient(dy_dx, x)  # Will compute to 2.0

iter <- make_iterator_one_shot(train_dataset)
batch <- iterator_get_next(iter)
with(tf$GradientTape() %as% crit_tape, {
  
  
  noise <- k_random_normal(c(dim(batch)[1], noise_dim))
  
  generated_images <- generator(noise)
  crit_real_output <- critic(batch, training = TRUE)
  crit_generated_output <-
    critic(generated_images, training = TRUE)
  
  shape <- tf$concat(list(tf$shape(batch)[1, drop = FALSE], 
                          tf$ones(batch$shape$ndims - 1, "int32")), 0L)

  alpha = tf$random_uniform(shape=shape, minval=0., maxval=1.)
  averaged_samples = batch + alpha * (generated_images - batch)
  averaged_samples$set_shape(batch$shape)
  
  with(tf$GradientTape() %as% grad, {
    
    
    grad$watch(averaged_samples)
    averaged_output <- critic(averaged_samples, training = TRUE)
    
  })
  
  crit_tape$watch(averaged_output)
  #crit_tape$watch(averaged_samples)
  crit_tape$watch(averaged_samples)
  grads <- grad$gradient(averaged_output, averaged_samples)
  
  
  w_loss <- tf$reduce_mean(crit_generated_output) - tf$reduce_mean(crit_real_output)
  
  gradients_sqr <- tf$square(grads)
  #   ... summing over the rows ...
  gradients_sqr_sum <- tf$reduce_sum(gradients_sqr,
                                     axis =  c(1L, 2L, 3L))
  #   ... and sqrt
  gradient_l2_norm <- tf$sqrt(gradients_sqr_sum)
  # compute lambda * (1 - ||grad||)^2 still for each single sample
  gradient_penalty <- tf$square(gradient_l2_norm - 1) ## whoah error here, was 1 - norm, should be norm - 1!
  # return the mean as loss over all the batch samples
  grad_penalty <- tf$reduce_mean(gradient_penalty)
  #grad_penalty <- tf$reduce_mean(grads)
  
  crit_loss <- w_loss + gradient_penalty_weight * grad_penalty
  

  
})

crit_tape$gradient(grad_penalty, critic$variables)

crit_tape$gradient(crit_loss, critic$variables)[[5]]
crit_tape$gradient(w_loss, critic$variables)[[5]]

crit_tape$gradient(crit_loss, critic$variables)[[1]]
crit_tape$gradient(w_loss, critic$variables)[[1]]

crit_tape$gradient(averaged_output, averaged_samples)[[1]]
grad$gradient(averaged_output, averaged_samples)[[1]]


batch <- iterator_get_next(iter)
with(tf$GradientTape(persistent = TRUE) %as% crit_tape, {
  
  
  noise <- k_random_normal(c(dim(batch)[1], noise_dim))
  
  generated_images <- generator(noise)
  crit_real_output <- critic(batch, training = TRUE)
  crit_generated_output <-
    critic(generated_images, training = TRUE)
  
  shape <- tf$concat(list(tf$shape(batch)[1, drop = FALSE], 
                          tf$ones(batch$shape$ndims - 1, "int32")), 0L)
  
  alpha = tf$random_uniform(shape=shape, minval=0., maxval=1.)
  averaged_samples = batch + alpha * (generated_images - batch)
  averaged_samples$set_shape(batch$shape)
  
  crit_tape$watch(averaged_samples)
  averaged_output <- critic(averaged_samples, training = TRUE)
  
  
  
  grads <- crit_tape$gradient(averaged_output, averaged_samples)
  

  
  w_loss <- tf$reduce_mean(crit_generated_output) - tf$reduce_mean(crit_real_output)
  
  gradients_sqr <- tf$square(grads)
  #   ... summing over the rows ...
  gradients_sqr_sum <- tf$reduce_sum(gradients_sqr,
                                     axis =  c(1L, 2L, 3L))
  #   ... and sqrt
  gradient_l2_norm <- tf$sqrt(gradients_sqr_sum)
  # compute lambda * (1 - ||grad||)^2 still for each single sample
  gradient_penalty <- tf$square(gradient_l2_norm - 1) ## whoah error here, was 1 - norm, should be norm - 1!
  # return the mean as loss over all the batch samples
  grad_penalty <- tf$reduce_mean(gradient_penalty)
  
  crit_loss <- w_loss + gradient_penalty_weight * grad_penalty
  

  
})

crit_tape$gradient(w_loss, critic$variables, unconnected_gradients = "zero")[[5]]
crit_tape$gradient(crit_loss, critic$variables, unconnected_gradients = "zero")[[5]]

crit_tape$gradient(grad_penalty, critic$variables, unconnected_gradients = "none")