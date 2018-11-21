library(userfriendlyscience)
library(broom)

chemical_properties_df <- read.table("CID_properties_nr.csv", sep=",", header = TRUE)

#See docs here: https://rpubs.com/aaronsc32/games-howell-test
games_howell_pvals <- function(features, group, feature_name){
  y = features
  x = group
  a = oneway(x=x, y=y, posthoc = 'games-howell', pvalueDigits = 5)
  comparisons = rownames(a$intermediate$posthoc)
  p_values = a$intermediate$posthoc$p
  #column_name = paste(feature_name, "p_values",sep="_")
  column_name = feature_name
  out_df = data.frame(comparisons, p_values)
  colnames(out_df)[2] <- column_name
  return(out_df)
}

feature = "MolecularWeight"
p_value_df <- 
games_howell_pvals(chemical_properties_df[[feature]],chemical_properties_df$drug_class, feature)

feature = "HBondAcceptorCount"
p_value_df[feature] <-games_howell_pvals(chemical_properties_df[[feature]],chemical_properties_df$drug_class, feature)[feature]

feature = "HBondDonorCount"
p_value_df[feature] <- games_howell_pvals(chemical_properties_df[[feature]],chemical_properties_df$drug_class, feature)[feature]

feature = "XLogP"
no_na_df = na.omit(chemical_properties_df)
p_value_df[feature] <- games_howell_pvals(no_na_df[[feature]],no_na_df$drug_class, feature)[feature]

p_value_df

write.table(p_value_df, file = "games_howell_posthoc_pvalues.tab", quote = FALSE, sep = "\t", row.names = FALSE)
