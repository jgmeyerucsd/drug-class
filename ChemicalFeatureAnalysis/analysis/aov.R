library(ggplot2)
library(cowplot)
library(broom)
library(userfriendlyscience)

setwd("/Users/ijmiller2/Desktop/UW_2018/For_Jesse/drug-class-ijm/data/")

chemical_properties_df <- read.table("CID_properties_nr.csv", sep=",", header = TRUE)

#Molecular Weight
mw_a <- aov(MolecularWeight ~ drug_class, chemical_properties_df)
mw_posthoc <- TukeyHSD(mw_a)
mw_posthoc_df = tidy(mw_posthoc)
#posthoc_df = posthoc_df[order(mw_posthoc$adj.p.value),]

new_df = as.data.frame(posthoc_df$comparison)
new_df$mw = mw_posthoc_df$adj.p.value

#XLogP
xlogp_a <- aov(XLogP ~ drug_class, chemical_properties_df)
xlogp_posthoc <- TukeyHSD(xlogp_a)
xlogp_posthoc_df = tidy(xlogp_posthoc)

new_df$xlogp = xlogp_posthoc_df$adj.p.value

#HBondDonor
HBondDonorCount_a <- aov(HBondDonorCount ~ drug_class, chemical_properties_df)
HBondDonorCount_posthoc <- TukeyHSD(HBondDonorCount_a)
HBondDonorCount_posthoc_df = tidy(HBondDonorCount_posthoc)

new_df$HBondDonorCount = HBondDonorCount_posthoc_df$adj.p.value

#HBondAcceptor
HBondAcceptorCount_a <- aov(HBondAcceptorCount ~ drug_class, chemical_properties_df)
HBondAcceptorCount_posthoc <- TukeyHSD(HBondAcceptorCount_a)
HBondAcceptorCount_posthoc_df = tidy(HBondAcceptorCount_posthoc)

new_df$HBondAcceptorCount = HBondAcceptorCount_posthoc_df$adj.p.value

write.table(new_df, file = "aov_posthoc_pvalues.tab", quote = FALSE, sep = "\t", row.names = FALSE)