#' Try and find hakea images on inaturalist. Filter them for flower pics later
library(rinat)
library(dplyr)
library(readr)

hakea_obs <- get_inat_obs(taxon_name = "Hakea", maxresults = 5000)
banksia_obs <- get_inat_obs(taxon_name = "Banksia", maxresults = 5000)
grevillea_obs <- get_inat_obs(taxon_name = "Grevillea", maxresults = 5000)

#' We will just use research grade observations for now.

prot_obs <- bind_rows(hakea_obs, banksia_obs, grevillea_obs)

prot_obs <- prot_obs %>%
  dplyr::filter(quality_grade != "needs_id")

#' What species do we have?
table(prot_obs$scientific_name)

write_rds("data/image_data.rds")

#' We will download all of these for now and later delete any that don't have flowers in them.
dir.create("inat_prot_photos")
for(i in 1:nrow(prot_obs)) {
  folder_name <- paste0("inat_prot_photos/", gsub(" ", "_", prot_obs$scientific_name[i], fixed = TRUE))
  if(!dir.exists(folder_name)) {
    dir.create(folder_name)
  }
  file_name <- gsub("?", "_", basename(prot_obs$image_url[i]), fixed = TRUE)
  if(!file.exists(paste0(folder_name, "/", file_name, ".jpg")))
  try(download.file(prot_obs$image_url[i], paste0(folder_name, "/", file_name, ".jpg"), mode = "wb", quiet = TRUE))
  Sys.sleep(2)
  message("Downloaded image # ", i, " of ", nrow(prot_obs), ": ", prot_obs$scientific_name[i])
}

#' Get more from ALA
library(ALA4R)
library(readr)
library(dplyr)

banksia_occ <- occurrences("genus:Banksia", download_reason_id = 7, fields = c("all_image_url", "species", "taxon_name"))
write_rds(banksia_occ, "data/ALA_Banksia.rds")
banksia_images <- banksia_occ$data %>%
  dplyr::filter(allImageUrl != "")
banksia_image_info <- image_info(banksia_images$allImageUrl)
write_rds(banksia_image_info, "data/ALA_Banksia_image_info.rds")

hakea_occ <- occurrences("genus:Hakea", download_reason_id = 7, fields = c("all_image_url", "species", "taxon_name"))
write_rds(hakea_occ, "data/ALA_Hakea.rds")
hakea_images <- hakea_occ$data %>%
  dplyr::filter(allImageUrl != "")
hakea_image_info <- image_info(hakea_images$allImageUrl)
write_rds(hakea_image_info, "data/ALA_Hakea_image_info.rds")

grevillea_occ <- occurrences("genus:Grevillea", download_reason_id = 7, fields = c("all_image_url", "species", "taxon_name"))
write_rds(grevillea_occ, "data/ALA_Grevillea.rds")
grevillea_images <- grevillea_occ$data %>%
  dplyr::filter(allImageUrl != "")
grevillea_image_info <- image_info(grevillea_images$allImageUrl)
write_rds(grevillea_image_info, "data/ALA_Grevillea_image_info.rds")


library(readr)
library(dplyr)
ala_dir <- "ala_prot_photos/"
## Download Banksia images
banksia_occ <- read_rds("data/ALA_Banksia.rds")
banksia_images <- banksia_occ$data %>%
  dplyr::filter(allImageUrl != "")
banksia_image_info <- read_rds("data/ALA_Banksia_image_info.rds")
for(i in 1:nrow(banksia_image_info)) {
  if(banksia_images$species[i] != "") {
    save_dir <- paste0(ala_dir, gsub(" ", "_", banksia_images$species[i]))
  } else {
    save_dir <- paste0(ala_dir, banksia_images$scientificName[i])
  }
  dir.create(save_dir, recursive = TRUE)
  new_file <- paste0(save_dir, "/", banksia_image_info$imageIdentifier[i], ".", sub("image/", "", banksia_image_info$mimeType[i]))
  if(!file.exists(new_file)) {
    Sys.sleep(2)
    try(download.file(banksia_image_info$imageURL[i], new_file, mode = "wb", quiet = TRUE))
  }
  message("Downloaded image # ", i, " of ", nrow(banksia_image_info), ": ", banksia_images$species[i])
}

## Download Hakea images
hakea_occ <- read_rds("data/ALA_Hakea.rds")
hakea_images <- hakea_occ$data %>%
  dplyr::filter(allImageUrl != "")
hakea_image_info <- read_rds("data/ALA_Hakea_image_info.rds")
for(i in 1:nrow(hakea_image_info)) {
  if(hakea_images$species[i] != "") {
    save_dir <- paste0(ala_dir, gsub(" ", "_", hakea_images$species[i]))
  } else {
    save_dir <- paste0(ala_dir, hakea_images$scientificName[i])
  }
  dir.create(save_dir, recursive = TRUE)
  new_file <- paste0(save_dir, "/", hakea_image_info$imageIdentifier[i], ".", sub("image/", "", hakea_image_info$mimeType[i]))
  if(!file.exists(new_file)) {
    Sys.sleep(2)
    try(download.file(hakea_image_info$imageURL[i], new_file, mode = "wb", quiet = TRUE))
  }
  message("Downloaded image # ", i, " of ", nrow(hakea_image_info), ": ", hakea_images$species[i])
}


## Download Grevillea images
grevillea_occ <- read_rds("data/ALA_Grevillea.rds")
grevillea_images <- grevillea_occ$data %>%
  dplyr::filter(allImageUrl != "")
grevillea_image_info <- read_rds("data/ALA_Grevillea_image_info.rds")
for(i in 1:nrow(grevillea_image_info)) {
  if(grevillea_images$species[i] != "") {
    save_dir <- paste0(ala_dir, gsub(" ", "_", grevillea_images$species[i]))
  } else {
    save_dir <- paste0(ala_dir, grevillea_images$scientificName[i])
  }
  dir.create(save_dir, recursive = TRUE)
  new_file <- paste0(save_dir, "/", grevillea_image_info$imageIdentifier[i], ".", sub("image/", "", grevillea_image_info$mimeType[i]))
  if(!file.exists(new_file)) {
    Sys.sleep(2)
    try(download.file(grevillea_image_info$imageURL[i], new_file, mode = "wb", quiet = TRUE))
  }
  message("Downloaded image # ", i, " of ", nrow(grevillea_image_info), ": ", grevillea_images$species[i])
}


## Make shiny app to annotate images
library(shiny)
library(taipan)

questions <- taipanQuestions(
  scene = div(radioButtons("flowers", label = "Is there at least one flower?",
                           choices = list("Yes", "No"), selected = "No"),
              radioButtons("seeds", label = "Is there at least one seed / pod / cone?",
                           choices = list("Yes", "No"), selected = "No"),
              radioButtons("zoom", label = "What zoom level?",
                           choices = list("Close-up: plant organs",
                                          "Intermediate Zoom",
                                          "Wide Shot: whole plant",
                                          "Wide Shot: landscape",
                                          "Drawing",
                                          "Specimen"), 
                           selected = "Close-up: plant organs"),
              radioButtons("non-plants", label = "Is there any non-plants?",
                           choices = list("Yes", "No"), selected = "No")),
  selection = div(radioButtons("flower_or_seed", label = "Is the selection the ...?",
                               choices = list("Good example of a flower",
                                              "Good example of a seed",
                                              "Whole plant"), selected = "Good example of a flower"))
)

image_files <- list.files("images", full.names = TRUE, recursive = TRUE)

buildTaipan(
  questions = questions,
  images = image_files,
  "taipan"
)

runApp("taipan")