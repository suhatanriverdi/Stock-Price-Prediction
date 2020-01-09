# Stock Price Prediction Using Machine Learning Techniques

This is a machine learning research about **Stock Price Prediction in Turkey** using Machine Learning Techniques. This project was given in the course of **Introduction to Data Science.**

![1](https://user-images.githubusercontent.com/36234545/72106070-c3411500-333f-11ea-904f-a504f9695ccb.png)

## Abstract
Stock market analysis is an important method for stock data estimation. With this method, stock market data is analyzed and evaluated statistically. With this method, the future value of a company's stock or financial instrument on the stock exchange can be determined. Forecasting future investments using daily stock data results in large profits for large companies. This study will focus on how to predict the future value of a financial company in the stock market by using machine learning algorithms. Our study provides an effective result with supervised learning algorithms. Keywords: Supervised Learning, Regression (Logistic & Linear Regression), Classification (K nearest neighborhood) 

## Introduction
In this assignment, we will work with historical data about the stock prices of the MIGROS A.Ş. collected from Yahoo Finance. Stock price is one of the hardest and time-consuming data points to predict. Although there are too many factors involved in the prediction, for the beginning, we will only use the collected data. At this point, Machine learning techniques have the potential to reveal patterns and predictions that we have not seen before, and they can be used to accurately make accurate predictions. We will implement machine learning algorithms to predict the future stock price of the MIGROS company, starting with three supervised learning algorithms. Supervised Learning has been broadly classified into 2 types, Regression and Classification. We will use Regression (Logistic & Linear Regression) and Classification (K nearest neighborhood).

Whether there is a relationship between two or more variables, and if so, to determine the degree of this relationship is one of the most frequently studied topics in statistics. **Linear regression** is a correlation between the two variables in relation to statistical changes or mutual exchange of values. If the values of the variable X change and the values of the variable Y change in the same or opposite direction, it can be said that there is a relationship between these two variables. For example, the increase in savings with income levels , the supply of a commodity increases while price falls, increase in profits with sales.

**Logistic regression** is a statistical management that has one or more arguments and is used to determine a result. The analysis of an existing data set gives us two possible results. This method is used in linear classification problems. Logistic regression contains data encoded as binary 1 or 0. The aim of logistic regression is to find the most appropriate model to define the appropriate relationship between a set of independent variables related to the dependent variable with two-way characteristic. With logistic regression, all appropriate variables between the independent variables should be included in the model, thus avoiding the inadequacy of the model. If inappropriate variables from within independent variables are excluded from the model, the model gets rid of complexity and is easier to interpret. 

**Classification** means that we use the training dataset to get better boundary conditions which could be used to determine each target class. Once the boundary conditions are determined, the next task is to predict the target class. The whole process is known as **classification**. The **kNN algorithm**, used for classification purposes, clusters objects according to the proximity relationships between each other. The distance in the algorithm is checked by the difference of each test data and the learned data and the calculation of the distance between these results and the control of the smallest values found here. “K” in the algorithm means neighbors and means the neighbors closest to the point where our test data is located in the coordinate plane.

![2](https://user-images.githubusercontent.com/36234545/72106867-4020be80-3341-11ea-86f8-a988235ce1f3.png)

## Literature Survey

Hakan Gündüz, Zehra Çataltepe and Yusuf Yaslan machine learning algorithms for the stock daily return prediction problem analyzed the estimation not only for the stock itself but also for other stocks and currencies using linear regression, random forest and multilayer perceptron. [1]

They also did a technical analysis and fundamental analysis for the examined the effects of financial news on the Istanbul Stock Exchange and tried to predict the direction of the ISE. They used feature selection, mutual information and naive bayes classifier methods for this analysis. [2]

Guresen, Kayakutlu and Daim artificial neural networks for forecasting the daily NASDAQ stock exchange rate were investigated. The methodology used in this study considered the short-term historical stock prices as well as the day of the week as inputs. [3]

Jaim and Kain used machine learning algorithms to predict stock market values. The methodology linear regression, random forests and multilayer perceptron for this analysis.[4]

Vincent Tatan used machine learning algorithms for Analyse, Visualize and Predict stock prices quickly with Python. [5] [6]

## Understanding The Research Question

We will dive into the explanation part of this assignment, but first it is important to determine the problem we would like to solve. In general, stock market analysis is divided into two parts, Basic Analysis and Technical Analysis. 

Basic Analysis involves analyzing the company's future profitability on the basis of the current business environment and financial performance. Technical Analysis involves reading graphs and using statistical figures to determine trends in the stock market. As you can guess, our focus will be on technical analysis. 

We will be using MIGROS dataset from Yahoo Finance. Financial markets, especially emerging markets, are affected by various factors such as global conditions. Political conditions, economic indicators (inflation rate, unemployment rate, etc.), company policies and merchant expectations. The relationship between these factors makes the quality of the stock market complex, noisy, nonlinear, and dynamic [7]. All these factors make prediction of stock prices/directions a challenging task. This important for investors and investors to make an accurate decision about their investment and forecast stocks plays a key role in performance strategies. There are various methods for forecasting the stock market. many use numerical and structured data, such as technical indicators [8]. Technical in literature Analysis and fundamental analysis are often used to predict future stock price. First uses history stock prices (eg daily, weekly, monthly stock prices) to forecast future prices, the latter Utilize data on the structure of the economy (eg inflation rates, exchange rates, interest rates, percentage of unemployment). Extracting relevant information from financial sources in recent financial research. The data is realized with the help of machine learning algorithms. Artificial neural networks (ANN) and Support vector machines (SVMs) are the most widely used algorithms, pure Bayesian, logistic regression and The K-nearest neighbors are still used because of their robustness and simplicity, as well as their ease of explanation their decisions. We will first load the dataset and define the target variable for the problem. (We’ve also found and appended the **Dollar Exchange Data**)

***
#### MIGROS Stock Market Dataset
![3](https://user-images.githubusercontent.com/36234545/72106868-4020be80-3341-11ea-97ad-6ec6e77c0974.png)
***

There are multiple variables in the dataset, We have columns named “Date”, “Open”, “High”, “Low”, ”Close”, and ”Volume”. The columns “Open” and “Close” represent the starting and final price at which the stock is traded on a specific day. “High” and  “Low” represent the maximum and minimum of the share for the related day. “Date” shows the date. And finally, the “Volume” is commonly reported as the number of shares that changed hands during a given day. It is important to point is that the market is closed on weekends, or some public holidays. Note again the table above, some date values are missing, like 23/10/2019 and 24/10/2018. Some of these dates are the national holidays, the 23th and 24th are weekends. The calculation of profit or loss is usually determined by the “**Closing Price**” of a stock, so we will consider the closing price as the target variable. To understand how it is shaped in our data, let's see the company’s historical analysis;

***
#### Migros Yahoo Finance Data
![4](https://user-images.githubusercontent.com/36234545/72106869-40b95500-3341-11ea-8380-2a13f3302f21.png)
***

## Yahoo Finance Migros Dataset

Our dataset contains MIGROS A.Ş’s BIST stock prices open, high, low, close and volume for each day BIST was open from 2000 through November 29, 2019. 

Since we need some technical indicators to correctly predict future stock prices, we chose the following indicators: 



*   Moving Average: Average of past 10 closing prices.
*   Exponential Moving Average: Same with moving average, but gives more weight to more recent observations.
*   Rate of Change: Ratio of the latest stock close price to 10 days ago.
*   Volatility: Standard deviation of last 10 stock close prices. 


## Exploratory Data Analysis and Feature Selection

Exploratory data analysis is an important part of data science since it helps us see the correlation between our independent and dependent variables. In this case, our independent values are technical indicators we’ve chosen and the dependent value is close price.

***
#### Exploratory data analysis
![5](https://user-images.githubusercontent.com/36234545/72106870-40b95500-3341-11ea-8ca7-f30e757f42d1.jpg)
***

***
#### Exponential Moving Average
![6](https://user-images.githubusercontent.com/36234545/72106871-40b95500-3341-11ea-87c7-fc37f676e0a2.png)
***

As we can see from the plot too, moving average and exponential moving average follows the actual close price, but without the seemingly high volatility in close price. 

***
#### Various Plots
![7](https://user-images.githubusercontent.com/36234545/72106872-40b95500-3341-11ea-8124-1127364ebd31.png)
![8](https://user-images.githubusercontent.com/36234545/72106874-40b95500-3341-11ea-88d7-700c6eb412ab.png)
***

***
#### Correlation Matrix
![9](https://user-images.githubusercontent.com/36234545/72106875-4151eb80-3341-11ea-9a56-686ff13ae971.png)
***

As we can see from the scatterplots and the correlation matrix; moving average, exponential MA. have high levels of correlation. Volatility and Rate of Change have some correlation, but doesn’t affect close price much. Volume has basically no correlation with close price.

### Correlation Analysis

In general, stock correlation relates to how stocks move in relation to each other. Although we can talk about whether asset classes are positively or negatively correlated, we can measure correlation specifically. Correlation should be measured for months or years rather than days to get an idea of ​​how two or more stocks are moving. 

An investor can understand how the two stocks are linked by looking at how each one performs better or underperforms the average return over time. A correlation value of 1 means that the two stocks have an excellent positive correlation. If the other one moves downwards, they have an excellent negative correlation, indicated by -1. If each stock appears to be completely independent of the other, it may be considered irrelevant and may have a value of 0. Following plot shows the Pearson correlation for our MIGROS data. [9]

***
#### MIGROS Correlation Plot
![10](https://user-images.githubusercontent.com/36234545/72106876-4151eb80-3341-11ea-8924-a7049431cdc1.png)
***

As we can see, we have a really strong correlation between attributes “Open - Close”, “High - Low”. We can say what “Open” is highly correlated with “Close” while “Low” is also highly correlated with “High”. We can remove either of them in the Feature Selection.


## Exploring Rolling Mean and Return Rate of Stocks

In this analysis, we analyze stocks using two important metrics: Rolling Average and Return Rate. Rolling Average (Moving Average) - to determine the trend Rolling Average / Moving Average (MA) corrects the price data by creating a continuously updated average price. This is useful for reducing the noise on our price chart. 

In addition, this Moving Average may mean Resistance caused by arising from the downward trend and the uptrend, where you can expect the trend to follow and expect a lower probability of deviation beyond the point of resistance. Moreover, the moving average (MA) is a simple technical analysis tool that corrects price data by creating a constantly updated average price. 

The average is taken over a period of time, such as 10 days, 20 minutes, 30 weeks, or any time the processor chooses. There are advantages to using a moving average in your trade, and options for what kind of moving average will be used. Moving average strategies are also popular and can be adapted to any time frame suitable for both long-term and short-term investors.

***
#### Migros Moving Average
![11](https://user-images.githubusercontent.com/36234545/72106877-4151eb80-3341-11ea-8c4a-46c95bff0dcd.png)
***

Above moving average smoothes the line and shows the rising or decreasing trend of a stock price. The Moving Average in this table shows an upward trend in the rise or fall of stock prices. Logically, you should buy when stocks fall, and you should sell as stocks rise.

## Return Deviation

To determine risk and return, firstly we need to understand the certain definitions. Expected return measures the average or expected value of the probability distribution of return on investment. The expected return on a portfolio is calculated by multiplying the weight of each asset by the expected return and adding values for each investment. We can use the following formula [8];

***
#### Formula for Returns
![12](https://user-images.githubusercontent.com/36234545/72106878-41ea8200-3341-11ea-82bc-5c6829eba81a.png)
***

***
#### Plotting the Return Rate
![13](https://user-images.githubusercontent.com/36234545/72106879-41ea8200-3341-11ea-907a-e2e8cd9e6ffa.png)
***

Logically, our ideal stocks should return as high and stable as possible. If you are hedging, you may want to avoid these stocks when you see a 10% drop in 2013. This decision is subject to your general opinion on stocks and competitor analysis in general.


## MIGROS Competitive Stock Analysis

This section is one of the most exciting research sections because we will see and compare the stocks of other competitors. We will analyze how a company performs compared to its competitors. Suppose we are interested in retail companies(supermarket) and want to compare: BIM Birlesik Magazalar, A101, CarrefourSA, Kipa, Sok Marketler. We can only explore the companies that we have access to their data on yahoo finance. [10]

***
#### Stocks Price For Bim, Bizim, Carrefoursa, Sok Supermarkets
![14](https://user-images.githubusercontent.com/36234545/72106882-42831880-3341-11ea-9ba5-dc7c4fa2b41d.png)
***

## Correlation Analysis with Competitors

Correlation Analysis answers whether if one competitor affects others. We can analyze the competition by running the percentage change and correlation function in pandas. The percentage change will see how much the price has changed from the previous day, which defines the return. Knowing the correlation will help us see if returns are affected by other stocks.

***
#### Correlations Among Competing Stocks
![15](https://user-images.githubusercontent.com/36234545/72106884-431baf00-3341-11ea-9f34-443d9ada8b43.png)
***

We can see that BİM A.S and MIGROS are more connected than other competitors. Let's draw MIGROS and BİM A.S plots with scatter plot to view returns distributions. 

***
#### Scatter Plot Of Bi̇m A.S And Migros
![16](https://user-images.githubusercontent.com/36234545/72106854-3dbe6480-3341-11ea-86f0-a7b34b1a68a2.png)
***

Distribution Chart of MIGROS and BİM Here we see a slight positive correlation between MIGROS returns and BİM returns. The higher BİM, the higher MIGROS in most cases. Let's further develop our analysis by drawing scattermatrix to visualize possible correlations between competing stocks. At the cross point, we will make the Core Density Estimate (KDE). KDE is a basic data correction problem that makes inferences about population based on limited data sample. It helps generate estimates of general distributions. 

***
#### Kde Plots And Scatter Matrix
![17](https://user-images.githubusercontent.com/36234545/72106855-3dbe6480-3341-11ea-93d9-9c208c52f00a.png)
***

We can find great relationships between rival stocks from Scatter Matrix and Heatmap. However, this may not be causal and may show the trend in the retail (supermarket) industry, rather than showing how competitor stocks affect each other.


## Stock Return Rate and Risk

 In addition to correlation, we analyze the risks and returns of each stock. In this case, we deduct the average return (Return Rate) and the standard return deviation (Risk).

***
#### Quick Scatter Plot among Stocks Risk and Returns
![18](https://user-images.githubusercontent.com/36234545/72106856-3e56fb00-3341-11ea-96c3-bc1f3f467512.png)
***

You can now view this clean risk table and see return comparisons for competitor stocks. Logically, you want to minimize risk and maximize returns. Therefore, you want to draw the line for your risk-return tolerance (Red line). You then set up rules to take these stocks below the red line (BIZIM, and BIM) and sell those stocks above the red line (SOK and CarrefourSA). This red line shows your expected value threshold and your basic level for a buy / sell decision.


## Predicting Stocks Price

We will use these three machine learning models to predict our stocks. These are Simple Linear Analysis, Quadratic Discriminant Analysis (QDA), and K Nearest Neighbor (kNN). But first, we need to do some feature engineering for features lise High Low Percentage and Percentage Change.

***
#### Data Frame Produced
![19](https://user-images.githubusercontent.com/36234545/72106857-3e56fb00-3341-11ea-97e1-c437cbd5f0da.png)
***

## Preprocessing and Splitting Data (Train/Test)

We cleaned and processed the data using the following steps before putting it in the prediction models by doing following steps, we also;

1. Drop missing values
2. Separate the label here, we want to predict the Close
3. Scale the X so that everyone can have the same distribution for linear regression
4. Finally We want to find Data Series of late X (test) and early X (train) for model generation and evaluation
5. Separate label and identify it as y
6. Separation of training and testing of model by train test split

## K Nearest Neighbor (kNN)

This kNN uses feature similarity to estimate the values of the data points. This makes the assigned new point similar to the points in the data set. To find the similarity, we will subtract the points to release the minimum distance (for example: Euclidean Distance).

***
#### KNN Model Visualization
![20](https://user-images.githubusercontent.com/36234545/72106858-3e56fb00-3341-11ea-8d68-57a8601ff82b.png)
***

kNN can be used for both classification and regression estimation problems. However, it is more widely used for classification problems in the industry. When we apply kNN to our data, we get the following plot:

***
#### Predictions Displayed in Plot
***![21](https://user-images.githubusercontent.com/36234545/72106860-3eef9180-3341-11ea-9dda-12d31f5b42ee.png)

## kNN Evaluation & Test Results

A simple quick method to evaluate is to use the score method in each trained model. The score method finds the mean accuracy of self.predict(X) with y of the test data set. We get following scores;

***
####  kNN Result Metrics
![22](https://user-images.githubusercontent.com/36234545/72106861-3eef9180-3341-11ea-9076-c85af47f33bb.png)
***

This shows a tremendous accuracy score (> 0.95) for most models. However, this does not mean that we can blindly place our stocks. 

There are many issues to consider, especially with different companies that have different price trajectories over time. As we have seen, the blue color discussed the stock price forecast based on the regression. The forecast predicted there would be a crisis that would not last long, and then he would recover. Therefore, we can buy stocks during the crisis and sell in the uptrend period.

To further analyse the stocks, here are some ideas on how you can contribute to better analyze stocks. These ideas will be useful for obtaining a more comprehensive analysis of stocks. Analyze economic qualitative factors such as news (news source and emotional analysis) Analyze economic inequality between economic quantitative factors, such as the IPB of a particular country, the company's origin.


## Multiple Linear Regression

One of the most basic machine learning algorithms that can be applied to this data is linear regression. In statistics, linear regression is a linear approach for modelling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. For more than one independent variable, the process is called multiple linear regression. The linear regression model returns an equation that determines the relationship between the independent variables and the dependent variable. The equation for linear regression is as follows:

![23](https://user-images.githubusercontent.com/36234545/72106862-3f882800-3341-11ea-8e6f-6dad2c7bc962.png)
***


In above figure, x1, x2, … xn represent the independent variables while the coefficients represent the weights. For our research question, as we have several attributes, we will use Multiple or multivariate linear regression that is a case of linear regression with two or more independent variables. After we apply Linear Regression we get following results.

***
#### Linear Regression Prediction
![24](https://user-images.githubusercontent.com/36234545/72106863-3f882800-3341-11ea-95a6-4a91b27d9a4c.png)
***

***
## Regression Evaluation & Test Results
![25](https://user-images.githubusercontent.com/36234545/72106864-3f882800-3341-11ea-9580-b5dc65c0df5e.png)
***

**Confusion Matrix:** An N*N matrix; where N is the number of predicted classes. Here are a few definitions to remember for a confusion matrix: 

**Accuracy:** The ratio of the total number of estimates. 

**Precision:** Proportion of positive cases correctly identified. 

**Negative Estimated Value:** The proportion of adverse cases identified correctly. 

**Recall:** The proportion of true positive cases identified correctly. 

**Specificity:** Proportion of true adverse events that are correctly defined.


## Conclusion

Estimating the values ​​in the stock exchange has an important place for economists, investment banks, hedge funds and other types of companies. Available information is processed by machine learning algorithms and other techniques at those companies and results are carefully analyzed to make important strategic decisions at those companies.

We picked Migros stock close price as our predicting target. We couldn’t predict stock price of Migros in a perfectly accurate way using the past stock and dollar exchange information, but keep in mind that there are many critics of the effective market hypothesis. Our idea is that it is not possible to predict stock values based only on past stock prices and there must be other information sources to exploit.

We've split the dataset as before November 4 and after November 4. Before part was used as the training data, after part was the test data. By using machine learning algorithms, we tried to estimate the descent and rise values of the stock market according to the closing value. We have completed this study using 3 methods and more than 2 metrics. kNN algorithm reached the most accurate result among all the methods we use.

During this project, we’ve gained important experience in the Data Science field because of all the research we did. Although we could not obtain exact results, we’ve done a lot of research and gained useful knowledge. Especially, the part of preprocessing and first analysis of the data took most of our time. Only after analyzing the data properly, you can actually use only the machine learning algorithms properly. We have learned a lot of interesting information on Turkey’s retail industry thanks to this project, it proved to be a good experiment in the end.

***
#### “Garbage in, Garbage out”
![26](https://user-images.githubusercontent.com/36234545/72106865-3f882800-3341-11ea-97c7-0fadcc07cee9.png)
***

## References

[1] Gündüz H, Çataltepe Z, Yaslan Y, Department of Computer Engineering, Faculty of Computer and Informatics Engineering, Istanbul Technical University, [https://journals.tubitak.gov.tr/elektrik/abstract.htm?id=21666](https://journals.tubitak.gov.tr/elektrik/abstract.htm?id=21666)

[2] Gunduz H, Çataltepe Z. Borsa İstanbul(BIST) daily prediction using financial news and balanced feature selection. Expert Syst Appl 2015; 42: 9001-9011, [https://www.researchgate.net/publication/259035088](https://www.researchgate.net/publication/259035088)

[3] Guresen E, Kayakutlu G, Daim TU. Using artificial neural network models in stock market index prediction, [http://www.scielo.org.pe/pdf/jefas/v21n41/a07v21n41.pdf](http://www.scielo.org.pe/pdf/jefas/v21n41/a07v21n41.pdf)

[4] Jaim S , Kain M , Prediction for Stock Marketing Using Machine Learning, [https://pdfs.semanticscholar.org/dbaf/9c14d2673f8ce5408f0d36445883930306d0.pdf](https://pdfs.semanticscholar.org/dbaf/9c14d2673f8ce5408f0d36445883930306d0.pdf)

[5] Towards Data Science - Python Examples, [https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7](https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7)

[6] Stock Prices Prediction Using Machine Learning and Deep Learning Techniques (with Python codes), [https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python)

[7] Stock daily return prediction using expanded features and feature selection, Turkish Journal of Electrical Engineering & Computer Sciences, [http://journals.tubitak.gov.tr/elektrik](http://journals.tubitak.gov.tr/elektrik)

[8] Atsalakis GS, Valavanis KP. Surveying stock market forecasting techniques

[9] Stock Correlations, [https://finance.yahoo.com/news/stock-correlation-212133633.html](https://finance.yahoo.com/news/stock-correlation-212133633.html)

[10] Investopedia - Return Deviation, [https://www.investopedia.com/terms/s/stock.asp](https://www.investopedia.com/terms/s/stock.asp)

## License
Eye Color Detector is licensed under the MIT license. See LICENSE for more information.

## Project Status
You can download the latest release from this repository.

## Disclaimer
This project was prepared and shared for educational purposes only. You can use or edit any file as you wish.

## About
Süha TANRIVERDİ Çankaya University, Computer Engineering
