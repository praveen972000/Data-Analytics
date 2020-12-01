library(tm)#vcorpus
library(SnowballC)
library(klaR)
library(ggplot2)
library(pROC)
library(RColorBrewer)

sms_raw <- read.csv("spam1.csv", stringsAsFactors = FALSE)

str(sms_raw)

table(sms_raw$type)

#checking the distribution of type of messages
theme_set(theme_bw())
ggplot(aes(x=type),data=sms_raw) +
  geom_bar(fill="red",width=0.5)

sms_raw$type <- factor(sms_raw$type)

str(sms_raw$type)

table(sms_raw$type)


library("wordcloud")
library("RColorBrewer")

spam <- subset(sms_raw, type == "spam")

ham <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# collection of text documents VCorpus library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

as.character(sms_corpus[[5]])
lapply(sms_corpus[1:4], as.character) #to see 1 to 4 texts

sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

as.character(sms_corpus_clean[[1]])


sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#The stemming process takes words like learned, learning, and learns, and strips the suffix in order to transform them into the base form, learn library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# The DocumentTermMatrix() function will take a corpus and create a data structure called a Document Term Matrix (DTM) in which rows indicate documents (SMS messages) and columns indicate terms (words).

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE, 
  stopwords = TRUE, 
  removePunctuation = TRUE, 
  stemming = TRUE))
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]


sms_train_labels <- sms_raw[1:4169, ]$type #75%
sms_test_labels <- sms_raw[4170:5559, ]$type

prop.table(table(sms_train_labels))

prop.table(table(sms_test_labels))

findFreqTerms(sms_dtm_train, 5)
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]


convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}


sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)

sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)



library(e1071)

sms_classifier <- naiveBayes(sms_train, sms_train_labels)

sms_test_pred <- predict(sms_classifier, sms_test)

library(caret)
# summarize results
confusionMatrix(sms_test_pred, sms_test_labels)

library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))


#This algorithm constructs tables of probabilities that are used to estimate the likelihood that new examples belong to various classes. 
#The probabilities are calculated using a formula known as Bayes' theorem, which specifies how dependent events are related. 
