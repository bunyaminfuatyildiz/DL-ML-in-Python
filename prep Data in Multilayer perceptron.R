library(mxnet) 
library(tidyverse) 
library(caret)
> install.packages("mxnet")

train <- read.csv("adult_processed_train.csv") 
train <- train %>% mutate(dataset = "train") 
test <- read.csv("adult_processed_test.csv") 
test <- test %>% mutate(dataset = "test")
# combine dataset and remove rows with NA
all <- rbind(train,test)
all <- all[complete.cases(all),] 

#we take any factor value and trim the whitespace.

unique(all$sex)
all <- all %>%  mutate_if(~is.factor(.),~trimws(.)) 




#filter just our training data. We will extract the target variable to a 
#vector and convert the values to numeric. Afterward, we can remove the target.

train <- all %>% filter(dataset == "train") 
train_target <- as.numeric(factor(train$target)) 
train <- train %>% select(-target, -dataset)


#step is to separate the data column-wise 

train_chars <- train %>%  
  select_if(is.character)

train_ints <- train %>% 
  select_if(is.integer)



#use the dummyVars() function from the caret package.define the columns that we would like converted to dummy variables
#Since we would like all columns to be converted, we just include a dot after the tilde.
ohe <- caret::dummyVars(" ~ .", data = train_chars) 
train_ohe <- data.frame(predict(ohe, newdata = train_chars))



# combine the data that was already numeric and the data that was converted to a numeric format
train <- cbind(train_ints,train_ohe)


# we will rescale the values so that all values are in a range between 0 and 1.
train <- train %>% mutate_all(funs(scales::rescale(.) %>% as.vector))



#repeat the same steps for the test dataset.
test <- all %>% filter(dataset == "test") 
test_target <- as.numeric(factor(test$target)) 
test <- test %>% select(-target, -dataset)
test_chars <- test %>%  
  select_if(is.character)
test_ints <- test %>%  
  select_if(is.integer)
ohe <- caret::dummyVars(" ~ .", data = test_chars) 
test_ohe <- data.frame(predict(ohe, newdata = test_chars))
test <- cbind(test_ints,test_ohe)
test <- test %>% mutate_all(funs(scales::rescale(.) %>% as.vector)) 

#We find the column that doesn't exist in both datasets and remove it using the following two lines of code:
setdiff(names(train), names(test))
train <- train %>% select(-native.countryHoland.Netherlands)


#vectors coded with values of either 1 or 2; however, we want these to be coded as either 0 or 1
train_target <- train_target-1 
test_target <- test_target-1 
