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