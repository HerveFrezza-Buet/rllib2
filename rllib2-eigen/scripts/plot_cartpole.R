
# Load my libraries
source ("~/Projets/Tools/r_utils/read_data.R")
source ("~/Projets/Tools/r_utils/plot_things.R")

## conversion to degree
to_deg <- function(x) { x/pi*180.0}

# ******************************************************************************
# plot episodes x, theta and length
#
# df <- load_headerfile( "test_rnd.csv" )
# plot_episode( df )
#
# WARNING : multiplot (based on grid and print) does not returns a value !!!!
plot_episodes_cartpole <- function(data)
{
  episode_factor <- as.factor(data$ep)

  # plots s.x=f(t), s.theta=f(t)
  pstate_base <- ggplot(data, aes(x=t, color=episode_factor))
  p1 <- pstate_base + geom_line( aes(y=to_deg(s.theta)), show.legend=FALSE) +
    geom_hline( yintercept=c(-12, 12))
  nice_p1 <- plot_adapt(p1, y="s.theta")
  p2 <- pstate_base + geom_line( aes(y=s.x), show.legend=FALSE) +
    geom_hline( yintercept=c(-0.6, 0.6))
  nice_p2 <- plot_adapt(p2, y="s.x")

  # plot vertical bar length=f(episode)
  episode_l <- unique(data$ep)
  episode_length <- sapply( episode_l, function(x) {max(data$t[data$ep==x])})

  ep_data <- data.frame( ep=episode_l, len=episode_length )
  # p3 <- ggplot(ep_data) + geom_col( aes(x=as.factor(ep), y=len, fill=as.factor(ep)) )
  p3 <- ggplot(ep_data) + geom_point( aes(x=as.factor(ep), y=len, color=as.factor(ep)) ) +
    geom_segment( aes(x=as.factor(ep), xend=as.factor(ep),
                      y=0, yend=len,
                      color=as.factor(ep)) )
  nice_p3 <- plot_adapt(p3, x="episode", y="length") + labs(color="ep")

  # multiplot
  pall <- arrangeGrob( nice_p1, nice_p2, nice_p3,
                      layout_matrix=matrix(c(1,1,3,2,2,3), nrow=2, byrow=TRUE))
  ## multiplot( nice_p1, nice_p2, nice_p3, file="",
  ##           layout=matrix(c(1,1,3,2,2,3), nrow=2, byrow=TRUE) )
  # Beware, multiplot DOES NOT return a value
  # TODO correct/modify multiplot ??
  ## return (list(nice_p1, nice_p2, nice_p3))
  return (pall)
}
plot_episodes_mountain <- function(data)
{
  episode_factor <- as.factor(data$ep)

  # plots s.pos=f(t), s.vel=f(t)
  pstate_base <- ggplot(data, aes(x=t, color=episode_factor))
  p1 <- pstate_base + geom_line( aes(y=s.vel), show.legend=FALSE)
  nice_p1 <- plot_adapt(p1, y="s.vel")
  p2 <- pstate_base + geom_line( aes(y=s.pos), show.legend=FALSE)
  nice_p2 <- plot_adapt(p2, y="s.pos")

  # plot vertical bar length=f(episode)
  episode_l <- unique(data$ep)
  episode_length <- sapply( episode_l, function(x) {max(data$t[data$ep==x])})

  ep_data <- data.frame( ep=episode_l, len=episode_length )
  # p3 <- ggplot(ep_data) + geom_col( aes(x=as.factor(ep), y=len, fill=as.factor(ep)) )
  p3 <- ggplot(ep_data) + geom_point( aes(x=as.factor(ep), y=len, color=as.factor(ep)) ) +
    geom_segment( aes(x=as.factor(ep), xend=as.factor(ep),
                      y=0, yend=len,
                      color=as.factor(ep)) )
  nice_p3 <- plot_adapt(p3, x="episode", y="length") + labs(color="ep")

  # multiplot
  pall <- arrangeGrob( nice_p1, nice_p2, nice_p3,
                      layout_matrix=matrix(c(1,1,3,2,2,3), nrow=2, byrow=TRUE))
  ## multiplot( nice_p1, nice_p2, nice_p3, file="",
  ##           layout=matrix(c(1,1,3,2,2,3), nrow=2, byrow=TRUE) )
  # Beware, multiplot DOES NOT return a value
  # TODO correct/modify multiplot ??
  ## return (list(nice_p1, nice_p2, nice_p3))
  return (pall)
}

make_plots <- function( pb_name, max_ep )
{
  plot_fun <- NULL

  if (pb_name == "cartpole") {
    plot_fun <- plot_episodes_cartpole
  }
  else if (pb_name == "mountain") {
    plot_fun <- plot_episodes_mountain
  }
  else {
    cat( "ERROR: make_plots cannot handle pb=", pb_name, "\n")
    stop()
  }

  # generate filenames
  lspi_names_g <- sapply( seq(0, max_ep), function(x) paste0("test_global_lspi_", x))
  # plots for test_rnd and test_init
  file_names_g <- c( "test_global_random", "test_global_init", lspi_names_g)
  lspi_names_n <- sapply( seq(0, max_ep), function(x) paste0("test_next_lspi_", x))
  # plots for test_rnd and test_init
  file_names_n <- c( "test_next_random", "test_next_init", lspi_names_n)

  save_plot <- function( file_in, plot_function ) {
    cat( "save_plot ", file_in, "\n" )
    data <- load_headerfile( paste0(file_in, ".csv") )
    ## pdf( file=paste0( "fig_", file_in, ".pdf"), paper="a4r")
    p <- plot_function( data )
    ## dev.off()
    ## ggsave( paste0("fig_", file_in, ".df"), plot=p, units="mm", width=290 )
    ggsave( paste0("fig_", file_in, ".png"), plot=p, units="px", width=1900, height=1000 )

  }

  lapply( file_names_g, save_plot, plot_fun)
  lapply( file_names_n, save_plot, plot_fun)

  if (pb_name == "mountain") {
    lspi_names_q <- sapply( seq(0, max_ep), function(x) paste0("test_qval_lspi_", x))
    file_names_q <- c( "test_qval_init", lspi_names_q)
    lapply( file_names_q, save_plot, plot_qval)

    # and now, transitions
    trans_names <- sapply( seq(0, max_ep), function(x) paste0("test_transitions_", x))
    file_names_t <- c( "test_transitions_start", trans_names)
    lapply( file_names_t, save_plot, plot_transitions)
  }
}

plot_learning_cartpole <- function(data, fig_name="")
{
  # prepare data for mean+max/min
  d_stat <- do.call(data.frame,
                    aggregate( len ~ it, data,
                              FUN=function(x) quantile(x, probs=c(0, 0.25, 0.5, 0.75, 1.0))
                              ))
  # rate of success
  data$success <- data$len < 300
  rate <- do.call(data.frame,
                  aggregate( success ~ it, data, FUN=sum )) # count nb of TRUE Values
  d_stat$rate <- rate$success

  p <- ggplot(d_stat, aes(x=it))
  p1 <- p + geom_ribbon( aes(ymin=len.25., ymax=len.75.), alpha=0.3) +
    geom_line( aes(y=len.50.)) +
    geom_line(aes(y=len.100.), linetype=2) +
    geom_line(aes(y=len.0.), linetype=2 )
  nice_p1 <- plot_adapt( p1, title=paste0(fig_name, "Episode length (quantiles)"), y="length")

  # cat(names(d_stat), "\n")
  # print(d_stat)
  p2 <- p + geom_line( aes(y=rate) )
  nice_p2 <- plot_adapt( p2, y="sucess rate")

  # multiplot
  pall <- arrangeGrob( nice_p1, nice_p2,
                      layout_matrix=matrix(c(1, 1, 2, 2), nrow=2, byrow=TRUE))

  return (pall)
}
plot_learning_mountain <- function(data, fig_name="")
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
  nice_p1 <- plot_adapt( p1, title=paste0(fig_name, "Episode length (quantiles)"), y="length")

  return (nice_p1)
}

make_learning <- function( pb_name, nb_last_lspi )
{
  plot_fun <- NULL
  plot_layout <- NULL
  p_qval <- NULL
  if (pb_name == "cartpole") {
    plot_fun <- plot_learning_cartpole
    plot_layout <- matrix( c(1,2, 1,2, 3,4), nrow=3, byrow=TRUE)
  }
  else if (pb_name == "mountain") {
    plot_fun <- plot_learning_mountain
    plot_layout <- matrix( c(1,2, 3,4), nrow=2, byrow=TRUE)
  }
  else {
    cat( "ERROR: make_learning cannot handle pb=", pb_name, "\n")
    return
  }

  data_g <- load_headerfile( "learning_global.csv" )
  pall_g <- plot_fun( data_g, "Q - " )
  ggsave( paste0("fig_learning_global.png"), plot=pall_g, units="px", width=1900, height=1000 )
  # grid.draw(pall_g)

  data_n <- load_headerfile( "learning_next.csv" )
  pall_n <- plot_fun( data_n, "next_Q -" )
  ggsave( paste0("fig_learning_next.png"), plot=pall_n, units="px", width=1900, height=1000 )
  # grid.draw(pall_n)

  data_w <- load_headerfile( "weights.csv" )
  p_w <- plot_weights( data_w )

  if (pb_name == "mountain") {
    p_qval <- plot_qval( load_headerfile( paste0( "test_qval_lspi_", nb_last_lspi-1, ".csv")))
  }

  pboth <- arrangeGrob( pall_g, pall_n, p_w, p_qval,
                      layout_matrix=plot_layout )
  grid.draw(pboth)
}

plot_weights <- function(data)
{
  data$w_maxabs <- sapply( seq(1, nrow(data)),
                          FUN=function(x) max(abs(data[x, 2:ncol(data)])) )

  # also compute ||w_{t+1} - w_t||_1
  deltaw = sapply( seq(2, nrow(data)),
                  FUN=function(i) sum(abs(data[i,] - data[i-1,]))/ncol(data) )
  data$deltaw <- c(0, deltaw)

  p <- ggplot(data, aes(x=it))
  p <- p + geom_line( aes(y=w_maxabs, color="max") ) + scale_y_log10()
  p <- p + geom_line( aes(y=deltaw, color="delta") )
  nice_p <- plot_adapt( p, title="Weights", y="max(abs(w)) et delta_w")

  return (nice_p)
}

plot_qval <- function(data)
{
  p <- ggplot(data, aes(x=pos, y=vel))
  pq <- p + geom_tile( aes(fill=qval))
  pp <- p + geom_tile( aes(fill=a_best))

  pboth <- arrangeGrob( pq, pp,
                       layout_matrix=matrix(c(1,2), nrow=1, byrow=TRUE))

  return (pboth)
}

plot_transitions <- function(data)
{
  # data is pos vel a n_pos n_vel r

  p <- ggplot(data, aes(x=pos, y=vel,
                        xend=n_pos, yend=n_vel,
                        color=as.factor(act)))
  p <- p + geom_segment()
  p <- p + xlim(-1.2, 0.6) + ylim(-0.07, 0.07)

  return (p)
}

run_cartpole <- function( nb_trans=500, nb_lspi=200, alpha=0.0, tau_epsilon=20.0)
{
  cmd <- paste( "examples/example-001-001-lspi-cartpole", nb_trans, nb_lspi, alpha, tau_epsilon, sep=" ")
  system( cmd )
  cat(cmd, "\n")
  make_learning_cartpole()
}
run_mountain <- function( nb_trans=500, nb_lspi=200, alpha=0.0, tau_epsilon=20.0,
                          uniform=TRUE)
{
  cmd <- paste( "examples/example-001-002-lspi-mountain_car", nb_trans, nb_lspi, alpha, tau_epsilon, sep=" ")
  if (!uniform) {
    cmd <- paste( cmd, "on_traj" )
  }
  system( cmd )
  cat(cmd, "\n")
  make_learning( "mountain", nb_lspi )
}
