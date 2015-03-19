# Setting up environment
setwd("~/CodeRClass/Learning")
require(caret)
require(lattice)
require(reshape)


# Loading data
data_train <- read.csv('./pml-training.csv',na.strings = '')
data_test <- read.csv('./pml-testing.csv', na.strings = '')


# Cleaning data
class_info <- data_train$classe
vars <- names(data_train)
misc_vars <- vars %in% c('X', 'user_name', 'raw_timestamp_part_1',
                         'raw_timestamp_part_2', 'cvtd_timestamp',
                         'new_window', 'num_window', 'classe')
window_vars <- grepl('kurtosis|skewness|max|min|amplitude|var|avg|stddev', vars)

data_train <- data_train[!(misc_vars | window_vars)]
data_test <- data_test[!(misc_vars | window_vars)]
any(sapply(data_train, function(x) any(is.na(x))))


# Partitioning
set.seed(1)
train_ind <- createDataPartition(class_info, p = 0.6, list = F)

training <- data_train[train_ind, ]
training_class <- class_info[train_ind]
testing <- data_train[-train_ind, ]
testing_class <- class_info[-train_ind]


# EDA
vars <- names(training)

stripout <- function(keyword, ...) {
    vars_sub <- grepl(keyword, vars)
    sub <- abs(training[, vars_sub])
    long <- data.frame(value = unlist(sub),
                       variable = rep(names(sub), each = nrow(sub)),
                       class = rep(training_class, times = ncol(sub)))
    print(stripplot(class ~ value | variable, groups = class,
                      data = long, alpha = 0.01, pch = 19,
                      auto.key = F, ...))
}

stripout('roll')
stripout('pitch')
stripout('yaw')
stripout('total')

vars_summaries <- grepl('roll|pitch|yaw|total', vars)
sub_summaries <- training[, vars_summaries]

s <- svd(t(as.matrix(sub_summaries)))
d <- s$d
dss <- d ** 2 / sum(d ** 2) * 100
cumdss <- cumsum(dss)
xyplot(cumdss ~ 1:length(cumdss), type = c('s', 'h'), ylim = c(0, 100),
       panel = function(...) {
           panel.xyplot(...)
           panel.abline(h = 95)
       })
recon <- d * t(s$v)
xyplot(recon[1, ] ~ recon[3, ], groups = testing_class,
       alpha = 0.1, pch = 19, auto.key = T)

# Training

training_s <- training[vars_summaries]
testing_s <- testing[vars_summaries]

set.seed(1)
fit <- train(training_class ~ ., training_s, method = 'gbm', verbose = F)

pred_train <- predict(fit, training_s)
confusionMatrix(training_class, pred_train)

pred_test <- predict(fit, testing_s)
confusionMatrix(testing_class, pred_test)

test_mat <- as.matrix(confusionMatrix(testing_class, pred_test))
test_mat <- round(apply(test_mat, 2, function(x) x / sum(x)), 3)
levelplot(test_mat, col.regions = gray(seq(1, 0, -0.01)),
          xlab = 'Prediction', ylab = 'Reference',
          panel = function(...) {
              panel.levelplot(...)
              panel.text(x = rep(1:5, each = 5), y = rep(1:5, 5),
                         labels = as.character(test_mat),
                         col = ifelse(test_mat > .6, 'white', 'black'))
          })

# Answering
query <- data_test[vars_summaries]
answer <- predict(fit, query)
