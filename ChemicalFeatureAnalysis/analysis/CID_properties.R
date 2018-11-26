library(ggplot2)
library(cowplot)
library(broom)
library(userfriendlyscience)

setwd("/Users/ijmiller2/Desktop/UW_2018/For_Jesse/drug-class-ijm/data/")

chemical_properties_df <- read.table("CID_properties_nr.csv", sep=",", header = TRUE)

p1 <- ggplot(aes(drug_class, MolecularWeight, fill=drug_class), data = chemical_properties_df) +
  geom_violin(trim = FALSE)  + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme(legend.position="none") + coord_flip()

p2 <- ggplot(aes(drug_class, XLogP, fill=drug_class), data = chemical_properties_df) +
  geom_violin(trim = FALSE)  + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme(legend.position="none") + coord_flip()

p3 <- ggplot(aes(drug_class, HBondAcceptorCount, fill=drug_class), data = chemical_properties_df) +
  geom_violin(trim = FALSE)  + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme(legend.position="none") + coord_flip()

p4 <- ggplot(aes(drug_class, HBondDonorCount, fill=drug_class), data = chemical_properties_df) +
  geom_violin(trim = FALSE)  + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme(legend.position="none") + coord_flip()
 
p <- plot_grid(p1, p2, p3, p4, labels = c("A","B","C","D"), align='hv')
p