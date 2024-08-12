# View learning performance of running LSPI on the Mountain Car
# plots depend of various CSV files
#
# canb used in batch mode (to save figure only)
# shell> Rscript $PATH_TO_SCRIPTS/view_results.R

# or inside a R interpreter
# > source("$PATH_TO_SCRIPTS/view_results.R")
# > fig <- view_results( "offline")
# > grid.draw( fig )
# > save_fig( fig, "fig_name.png" )

# Load needed libraries
library(ggplot2)
library(grid)
library(gridExtra)

nb_test <- 10
action_names <- c("Left", "-", "Right")

## Prepare a figure with subplots, saved in 'fig_$mode.png'
#
# Param: mode <- "offline" | "online"
#
# 1st row : evolution of performance of greedy policy vs LSPI iterations
# 2nd row : details about the last policy computed (orbits, ValueFunction, Greedy Action)
#
view_results <- function( mode="offline")
{

  # Plot with global performance offline sampling
  data_global_mode <- load_headerfile( paste0("global_", mode, ".csv") )
  fig_global_mode <- plot_global_learning( data_global_mode ) #, "Offline " )
  fig_global_mode <- fig_global_mode + labs( title=paste0("Testing policies in ", mode, " mode."))

  basename <- paste0( mode, "_lspi_", nb_test, ".csv")

  fig_transitions <- NULL
  figs <- NULL

  if (mode == "offline") {
    # if "offline", transitions have only been sampled in "_lspi_0.csv" file
    data_trans <- load_headerfile( "transition_offline_lspi_0.csv" )

    fig_transitions <- plot_transitions( data_trans )
    fig_transitions <- fig_transitions + labs( title="Sampled transitions" )

    figs <- plots_lspi_iteration( basename, with_transitions=FALSE)
  }
  else {
    figs <- plots_lspi_iteration( basename, with_transitions=TRUE )
    fig_transitions <- figs$trans
  }

  fig_iteration <- arrangeGrob( fig_transitions, figs$alltraj, figs$qval, figs$policy,
                              layout_matrix=matrix(c(1,2,3,4), nrow=1, byrow=TRUE),
                              top=paste0("Policy and Value, iteration=", nb_test))

  fig_full <- arrangeGrob( fig_global_mode, fig_iteration,
                          layout_matrix=matrix(c(1,2), nrow=2, byrow=TRUE))

  return (fig_full) # to be shown using grid.draw()
}

## build and save plots for all testing policies
#
make_all_test_plots <- function( mode="offline" )
{
  all_indexes <- sapply( seq(0, nb_test), FUN=as.character)

  save_plot <- function( index ) {
    basename <- paste0( mode, "_lspi_", index, ".csv")
    fig_transitions <- NULL
    figs <- NULL

    if (mode == "offline") {
      # if "offline", transitions have only been sampled in "_lspi_0.csv" file
      data_trans <- load_headerfile( "transition_offline_lspi_0.csv" )
      fig_transitions <- plot_transitions( data_trans )
      fig_transitions <- fig_transitions + labs( title="Sampled transitions" )

      figs <- plots_lspi_iteration( basename, with_transitions=FALSE)
    }
    else {
      figs <- plots_lspi_iteration( basename, with_transitions=TRUE )
      fig_transitions <- figs$trans
    }

    fig_iteration <- arrangeGrob( fig_transitions, figs$alltraj, figs$qval, figs$policy,
                                 layout_matrix=matrix(c(1,2,3,4), nrow=1, byrow=TRUE),
                                 top=paste0("Policy and Value, iteration=", index))

    save_fig( fig_iteration, paste0("fig_", mode, "_lspi_", index, ".png"), size=c(5024, 1362) )
  }
  lapply( all_indexes, save_plot )

  return(invisible(NULL))
}

## build orbits, qval, policy [and transitions] plots
#
# Params: - basename (string) common element in needed CSV filenames
#         - with_transitions (bool) do we compute a transitions plot ?
#
# Returns: list(alltraj, qval, policy, [trans]) of ggplots
plots_lspi_iteration <- function( basename, with_transitions=FALSE )
{
  # Plot all trajectories of last policy testing
  test_filename <- paste0("test_", basename)
  data_alltraj <- load_headerfile( test_filename )
  fig_alltraj <- plot_trajectories( data_alltraj )
  fig_alltraj <- fig_alltraj +labs( title="All test orbits" )

  # Plot ValueFunction and Greedy Policy
  qval_filename <- paste0("qval_", basename)
  data_qval <- load_headerfile( qval_filename )

  fig_qval <- plot_qval( data_qval )
  fig_qval <- fig_qval + labs(title="Value")
  fig_policy <- plot_policy( data_qval )
  fig_policy <- fig_policy + labs(title="Greedy Policy")

  results <- list(alltraj=fig_alltraj, qval=fig_qval, policy=fig_policy)

  # Transition buffer ?
  if (with_transitions) {
    trans_filename <- paste0("transition_", basename)
    data_trans <- load_headerfile( trans_filename )
    fig_trans <- plot_transitions( data_trans )
    fig_trans <- fig_trans + labs( title="Sampled transitions" )

    results$trans <- fig_trans
  }

  return (results)
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

## make plot of all transitions
#
# data : s.pos s.vel next_s.pos a next_s.vel
plot_transitions <- function( data )
{
  p <- ggplot(data, aes(x=s.pos, y=s.vel,
                        xend=next_s.pos, yend=next_s.vel,
                        color=as.factor(a)))
  p <- p + geom_segment( show.legend=FALSE )
  p <- p + labs(x="Pos", y="Vel", color="Action index")
  p <- p + xlim(-1.2, 0.6) + ylim(-0.07, 0.07)

  return (p)
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
## Save a plot in file
save_fig <- function( plot_to_save, filename, size=c(2100, 2100) )
{
  # Saving
  ggsave( filename, plot=plot_to_save, units="px", width=size[1], height=size[2])
  cat( "Saving in ", filename, " ...\n")
}
###############################################################################

###############################################################################
###############################################################################
## When used in batch mode, save figures
args <- commandArgs()
cat( args, "\n")
if (any(grepl("view_results.R", args))) {
  cat("** BATCH MODE**\n")

  fig_offline <- view_results( "offline" )
  save_fig( fig_offline, "fig_offline.png", size=c(5024, 2725) )
  fig_online <- view_results( "online" )
  save_fig( fig_online, "fig_online.png", size=c(5024, 2725) )

  make_all_test_plots( "offline" )
  make_all_test_plots( "online" )
}
