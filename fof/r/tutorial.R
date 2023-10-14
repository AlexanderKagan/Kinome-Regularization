library(tidyverse)
library(refund.shiny)

## set design elements
set.seed(1)
I = 50
p = 2

beta1 = function(t) { 1 }
beta2 = function(t) { cos(2*t*pi) }

## generate subjects and observation times
concurrent.data =
  data.frame(
    subj = rep(1:I, each = 20)
  ) %>%
  mutate(time = runif(dim(.)[1])) %>%
  arrange(subj, time) %>%
  group_by(subj) %>%
    mutate(Cov_1 = runif(1, .5, 1.5) * sin(2 * pi * time),
           Cov_2 = runif(1, 0, 1) + runif(1, -.5, 2) * time,
           Y = Cov_1 * beta1(time) +
               Cov_2 * beta2(time) +
               rnorm(20, 0, .5)) %>%
  ungroup()

library(vbvs.concurrent)

fit_vb = vb_concurrent(Y ~ Cov_1 + Cov_2 | time, id.var = "subj", data = concurrent.data,
                       t.min = 0, t.max = 1, standardized = TRUE)

fit_vbvs = vbvs_concurrent(Y ~ Cov_1 + Cov_2 | time, id.var = "subj", data = concurrent.data,
                           t.min = 0, t.max = 1, standardized = TRUE)

plot_shiny(fit_vb)
plot_shiny(fit_vbvs)