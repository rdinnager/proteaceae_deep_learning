library(imager)
all_images <- list.files("images", full.names = TRUE, recursive = TRUE)

proc_folder <- "images_for_processing"
dir.create(proc_folder)
dir.create(file.path(proc_folder, "Banksia"))
dir.create(file.path(proc_folder, "Hakea"))
dir.create(file.path(proc_folder, "Grevillea"))

banksia_images <- grep("Banksia", all_images, value = TRUE)
hakea_images <- grep("Hakea", all_images, value = TRUE)
grevillea_images <- grep("Grevillea", all_images, value = TRUE)

file.copy(banksia_images, file.path(proc_folder, "Banksia", basename(banksia_images)))
file.copy(hakea_images, file.path(proc_folder, "Hakea", basename(hakea_images)))
file.copy(grevillea_images, file.path(proc_folder, "Grevillea", basename(grevillea_images)))


##### Crop images ##########
library(imager)
library(magick)
proc_images <- list.files("images_for_processing", full.names = TRUE, recursive = TRUE)

for(i in 1:length(proc_images)) {
  img <- try(image_read(proc_images[i]) %>% magick2cimg())
  if(class(img) != "try-error") {
    w_to_h <- width(img) / height(img)
    if(w_to_h < 1) {
      cutit <- (height(img) - width(img)) / 2
      img <- imsub(img, y %inr% c(cutit, height - cutit))
    } else {
      cutit <- (width(img) - height(img)) / 2
      img <- imsub(img, x %inr% c(cutit, width - cutit))
    }
    img <- resize(img, 128, 128, interpolation_type = 6)
    save.image(img, gsub("images_for_processing", "images_cropped", proc_images[i]), quality = 1)
  }
  print(paste0(i, " of ", length(proc_images)))
}



image_dir <- "data/images_small"
train_dir <- "train_dir"
validation_dir <- "validation_dir"
test_dir <- "test_dir"

#' Load and merge imagesets

prot_dirs <- list.dirs("images_cropped", recursive = TRUE, full.names = FALSE)
hakea_dirs <- prot_dirs[grep("Hakea", prot_dirs)]
banksia_dirs <- prot_dirs[grep("Banksia", prot_dirs)]
grevillea_dirs <- prot_dirs[grep("Grevillea", prot_dirs)]

test_prop <- 0.05
validation_prop <- 0.1

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

hakea_files <- list.files(file.path("images_cropped", hakea_dirs), recursive = TRUE, full.names = TRUE)
hakea_test_files <- sample(hakea_files, floor(test_prop * length(hakea_files)))
hakea_validation_files <- sample(hakea_files[!hakea_files %in% hakea_test_files], floor(validation_prop * length(hakea_files)))
hakea_train_files <- hakea_files[!hakea_files %in% c(hakea_test_files, hakea_validation_files)]
file.copy(hakea_test_files,
          file.path(image_dir, test_dir, "test_hakea", basename(hakea_test_files)))
file.copy(hakea_validation_files,
          file.path(image_dir, validation_dir, "validation_hakea", basename(hakea_validation_files)))
file.copy(hakea_train_files,
          file.path(image_dir, train_dir, "train_hakea", basename(hakea_train_files)))

banksia_files <- list.files(file.path("images_cropped", banksia_dirs), recursive = TRUE, full.names = TRUE)
banksia_test_files <- sample(banksia_files, floor(test_prop * length(banksia_files)))
banksia_validation_files <- sample(banksia_files[!banksia_files %in% banksia_test_files], floor(validation_prop * length(banksia_files)))
banksia_train_files <- banksia_files[!banksia_files %in% c(banksia_test_files, banksia_validation_files)]
file.copy(banksia_test_files,
          file.path(image_dir, test_dir, "test_banksia", basename(banksia_test_files)))
file.copy(banksia_validation_files,
          file.path(image_dir, validation_dir, "validation_banksia", basename(banksia_validation_files)))
file.copy(banksia_train_files,
          file.path(image_dir, train_dir, "train_banksia", basename(banksia_train_files)))

grevillea_files <- list.files(file.path("images_cropped", grevillea_dirs), recursive = TRUE, full.names = TRUE)
grevillea_test_files <- sample(grevillea_files, floor(test_prop * length(grevillea_files)))
grevillea_validation_files <- sample(grevillea_files[!grevillea_files %in% grevillea_test_files], floor(validation_prop * length(grevillea_files)))
grevillea_train_files <- grevillea_files[!grevillea_files %in% c(grevillea_test_files, grevillea_validation_files)]
file.copy(grevillea_test_files,
          file.path(image_dir, test_dir, "test_grevillea", basename(grevillea_test_files)))
file.copy(grevillea_validation_files,
          file.path(image_dir, validation_dir, "validation_grevillea", basename(grevillea_validation_files)))
file.copy(grevillea_train_files,
          file.path(image_dir, train_dir, "train_grevillea", basename(grevillea_train_files)))
