# **************************************************************************
# View learning performance of running LSPI on the Mountain Car
# plots depend of various CSV files

# usage in R
# > source("$PATH_TO_SCRIPTS/view_results.R")
# > grid.draw(view_result())

# Load needed libraries
library(ggplot2)
library(grid)
library(gridExtra)

nb_test <- 10
action_names <- c("Left", "-", "Right")

## Prepare a figure with subplots
# 1st row : evolution of performance of greedy policy vs LSPI iterations
# 2nd row : details about the last policy computed (orbits, ValueFunction, Greedy Action)
#
view_results <- function()
{
  # Plot with global performance offline sampling
  data_global_offline <- load_headerfile( "global_offline.csv" )
  fig_global_offline <- plot_global_learning( data_global_offline ) #, "Offline " )
  fig_global_offline <- fig_global_offline + labs( title="Testing policies. Offline.")

  # Plot all trajectories of last policy testing
  test_filename <- paste0("test_offline_lspi_", nb_test-1, ".csv")
  data_alltraj <- load_headerfile( test_filename )
  fig_alltraj <- plot_trajectories( data_alltraj )
  fig_alltraj <- fig_alltraj +labs( title="All test orbits" )

  # Plot ValueFunction and Greedy Policy
  qval_filename <- paste0("qval_offline_lspi_", nb_test-1, ".csv")
  data_qval <- load_headerfile( qval_filename )

  fig_qval <- plot_qval( data_qval )
  fig_qval <- fig_qval + labs(title="Value")
  fig_policy <- plot_policy( data_qval )
  fig_policy <- fig_policy + labs(title="Greedy Policy")

  fig_iteration <- arrangeGrob( fig_alltraj, fig_qval, fig_policy,
                              layout_matrix=matrix(c(1,2,3), nrow=1, byrow=TRUE),
                              top=paste0("Policy and Value, iteration=", nb_test-1))

  fig_full <- arrangeGrob( fig_global_offline, fig_iteration,
                          layout_matrix=matrix(c(1,2), nrow=2, byrow=TRUE))

  return (fig_full) # to be shown using grid.draw()

}

## make plot of quantiles of episode length vs lspi iteration
#
# data : it len
plot_global_learning <- function( data )
{
  # prepare data for mean+max/min
  d_stat <- do.call(data.frame,
                    aggregate( len ~ it, data,
                              FUN=function(x) quantile(x, probs=c(0, 0.25, 0.5, 0.75, 1.0))
                              ))

  p <- ggplot(d_stat, aes(x=it))
  p1 <- p + geom_ribbon( aes(ymin=len.25., ymax=len.75.), alpha=0.3) +
    geom_line( aes(y=len.50.)) +
    geom_line(aes(y=len.100.), linetype=2) +
    geom_line(aes(y=len.0.), linetype=2)
  p1 <- p1 + labs(x="iterations", y="episode length (quantiles)")

  return (p1)
}

## make plot of all trajectories of a test of policy
#
# data : ep s.pos s.vel next_s.pos next_s.vel
plot_trajectories <- function( data )
{
  p <- ggplot(data, aes(x=s.pos, y=s.vel,
                        xend=next_s.pos, yend=next_s.vel,
                        color=as.factor(ep)))
  p <- p + geom_segment( show.legend=FALSE )
  p <- p + labs(x="Pos", y="Vel", color="Episode nb")
  p <- p + xlim(-1.2, 0.6) + ylim(-0.07, 0.07)

  return (p)
}

## make plots of Value function and BestAction according to Pos x Vel
#
# data : s.pos s.vel qval
plot_qval <- function(data)
{
  p <- ggplot(data, aes(x=s.pos, y=s.vel))
  pq <- p + geom_raster( aes(fill=qval))
  pq <- pq + scale_fill_viridis_c()

  return (pq)
}

## make plots of BestAction according to Pos x Vel
#
# data : s.pos s.vel a_best
plot_policy <- function( data )
{
  # best action as factors, so as to rename
  a.str <- as.factor( data$a_best )
  levels(a.str) <- action_names
  data$a_best.str <- a.str

  p <- ggplot(data, aes(x=s.pos, y=s.vel))
  pp <- p + geom_raster( aes(fill=a_best.str))
  pp <- pp + scale_fill_viridis_d()
  pp <- pp + labs(fill="Action")

  return (pp)
}


###############################################################################
## Load a file with a header '## ' as a data frame
##
load_headerfile <- function( name )
{
  ## Load first line to get the name of the fields
  df <- file( name, "r")
  header <- readLines( df, n=1 )
  # split
  tokens <- strsplit( header, '\t')[[1]]
  # remove "## " from first token
  tokens[1] <- substring( tokens[1], 4, 1000)
  close(df)

  ## Now read data and name it
  data <- read.table( file=name )
  names(data) <- tokens

  return (data)
}
###############################################################################
