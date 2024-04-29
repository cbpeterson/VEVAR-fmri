library(ggraph)
library(igraph)
library(R.matlab)

setwd("~/Dropbox/Study/Rice/Research/Projects/VEVAR/Code_paper")

roi = 10

colnames <- readMat("ROI_names.mat")
col_names <- unlist(colnames)
col_names2 = gsub(" ", "", col_names)[1:roi]

grid.col <- rep("grey", length(col_names2))
names(grid.col) <- col_names2

labels = c(names(grid.col[6:10]), rev(names(grid.col[1:5])))
angles = c((90 - 36*seq(1:10))[1:5],(270 - 36*seq(1:10))[6:10])

edge_associations = read.csv("Results_g1.csv",header = F,sep = ',')
rownames(edge_associations) = colnames(edge_associations) = col_names2
edge_associations = edge_associations[labels,labels]
edge_associations = t(edge_associations)
mygraph = graph_from_adjacency_matrix(as.matrix(edge_associations))

node_labels = c()
for(i in colnames(edge_associations)){
  node_labels = c(node_labels,paste(" ",paste(i,sep = "")," "))
}
node_labels = gsub("_R","",node_labels)
node_labels = gsub("_L","",node_labels)

edge_associations2 = read.csv("Results_g2.csv",header = F,sep = ',')
rownames(edge_associations2) = colnames(edge_associations2) = col_names2
edge_associations2 = edge_associations2[labels,labels]
edge_associations2 = t(edge_associations2)
mygraph2 = graph_from_adjacency_matrix(as.matrix(edge_associations2))

edge.colors = rep(1,length(E(mygraph)))
edge.colors[do.call(paste0,as.data.frame(ends(mygraph,E(mygraph)))) %in% do.call(paste0,as.data.frame(ends(mygraph2,E(mygraph2))))] = 2

pdf("network_group1.pdf")
ggraph(mygraph,layout = 'linear', circular = TRUE) + 
  geom_edge_arc(aes(colour = factor(edge.colors)), edge_alpha=1, edge_width=0.3, fold=FALSE,arrow = arrow(type = "closed",length = unit(2, 'mm'))) +
  geom_node_point(aes(size=(.1 + apply(edge_associations,1,sum))^1.2), alpha=0.4) +
  geom_node_text(aes(label = node_labels), angle = angles, hjust=c(rep(0,5),rep(1,5)), size=2.75) +
  theme_void() + 
  xlim(-1.35,1.35) + 
  ylim(-1.35,1.35) +
  theme(
    legend.position="none",
    plot.margin=unit(c(0.0,0.0,0.0,0.0), "null"),
    panel.spacing=unit(c(0.0,0.0,0.0,0.0), "null"),
    plot.title = element_text(hjust = 0.5)
  ) +  
  ggtitle("Group 1") + 
  scale_edge_color_manual(values = c('black','red'))
dev.off()

edge.colors = rep(1,length(E(mygraph2)))
edge.colors[do.call(paste0,as.data.frame(ends(mygraph2,E(mygraph2)))) %in% do.call(paste0,as.data.frame(ends(mygraph,E(mygraph))))] = 2

pdf("network_group2.pdf")
ggraph(mygraph2,layout = 'linear', circular = TRUE) + 
  geom_edge_arc(aes(colour = factor(edge.colors)), edge_alpha=1, edge_width=0.3, fold=FALSE,arrow = arrow(type = "closed",length = unit(2, 'mm'))) +
  geom_node_point(aes(size=(.1 + apply(edge_associations2,1,sum))^1.2), alpha=0.4) +
  geom_node_text(aes(label = node_labels), angle = angles, hjust=c(rep(0,5),rep(1,5)), size=2.75) +
  theme_void() + 
  xlim(-1.35,1.35) + 
  ylim(-1.35,1.35) +
  theme(
    legend.position="none",
    plot.margin=unit(c(0.0,0.0,0.0,0.0), "null"),
    panel.spacing=unit(c(0.0,0.0,0.0,0.0), "null"),
    plot.title = element_text(hjust = 0.5)
  ) +  
  ggtitle("Group 2") + 
  scale_edge_color_manual(values = c('black','red'))
dev.off()