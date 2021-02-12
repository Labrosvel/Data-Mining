#
# 1. generate some random data for classification
#
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.model_selection as model_select
from sklearn import metrics

x, y = datasets.make_classification(n_features=1, n_redundant=0, n_informative=1,
        n_classes=2, n_clusters_per_class=1, n_samples=100)

M=len(x)
xmin = 0.95 * min( x )
xmax = 1.05 * max( x )

# 2. plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
for j in range( M ):
    if ( y[j] == 0 ):
        h0, = plt.plot( x[j], x[j], 'b.', markersize=10 )
    else:
        h1, = plt.plot( x[j], x[j], 'r+', markersize=10 )
plt.legend(( h0, h1 ), ( 'class0', 'class1' ), loc='upper left' )
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( xmin, xmax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'x' , fontsize=14 )
plt.title( 'raw data', fontsize=14 )
plt.show()

#-partition the data
x_train, x_test, y_train, y_test = model_select.train_test_split( x, y, test_size=0.10 )
M_train = len( x_train )
M_test = len( x_test )

#-plot partitioned data
plt.figure()
# plot data points
for j in range( len( x_train )):
    if ( y_train[j] == 0 ):
        h0train, = plt.plot( x_train[j], x_train[j], 'b.', markersize=10 )
    else:
        h1train, = plt.plot( x_train[j], x_train[j], 'r+', markersize=10 )
for j in range( len( x_test )):
    if ( y_test[j] == 0 ):
        h0test, = plt.plot( x_test[j], x_test[j], 'b>', markersize=10 )
    else:
        h1test, = plt.plot( x_test[j], x_test[j], 'rs', markersize=10 )
plt.legend(( h0train, h1train, h0test, h1test ), ( 'train, class0', 'train, class1', 'test, class0', 'test, class1' ), loc='upper left' )
# set plot axis limits so it all displays nicely
plt.xlim(( xmin, xmax ))
plt.ylim(( xmin, xmax ))
# add plot labels and ticks
plt.xticks( fontsize=14 )
plt.yticks( fontsize=14 )
plt.xlabel( 'x' , fontsize=14 )
plt.ylabel( 'x' , fontsize=14 )
plt.title( 'partitioned data', fontsize=14 )
plt.show()


# 3. Initialise a perceptron model
from sklearn import linear_model
per = linear_model.Perceptron() #Initialise the perceptron model
per.fit(x_train,y_train) # build the model by adjusting the weights to fit the data
print('perceptron weights:')
print('w0 = {}, w1 = {}'.format(per.intercept_, per.coef_))

#-run prediction with model on training data
y_hat = per.predict( x_train ) 
print('training accuracy = ', ( metrics.accuracy_score( y_train, y_hat, normalize=True )))
# plot results 
plt.figure()
# plot raw data points
for j in range( M_train ):
    if ( y_train[j] == 0 ):
        h0train, = plt.plot( x_train[j], x_train[j], 'b.', markersize=10 )
    else:
        h1train, = plt.plot( x_train[j], x_train[j], 'r+', markersize=10 )
plt.legend(( h0train, h1train ), ( 'train, class0', 'train, class1' ), loc='upper left' )
# plot boundary
[xmin,xmax] = plt.xlim()
xx = []
yy = []
for j in range( M_train ):
    xx.append( x_train[j] )
    y_hat = per.intercept_ + x_train[j] * per.coef_[0,0] # plots the decision boundary
    yy.append( y_hat )
plt.plot( xx, yy, 'k-' )
plt.show()

#-run prediction with model on test data
y_hat = per.predict( x_test ) 
print('test accuracy =', ( metrics.accuracy_score( y_test, y_hat, normalize=True )))
# plot results
plt.figure()
# plot raw data points
for j in range( M_test ):
    if ( y_test[j] == 0 ):
        h0test, = plt.plot( x_test[j], x_test[j], 'b>', markersize=10 )
    else:
        h1test, = plt.plot( x_test[j], x_test[j], 'rs', markersize=10 )
plt.legend(( h0test, h1test ), ( 'test, class0', 'test, class1' ), loc='upper left' )
plt.title( 'decision boundary' )
# plot boundary
[xmin,xmax] = plt.xlim()
xx = []
yy = []
for j in range( M_test ):
    xx.append( x_test[j] )
    y_hat = per.intercept_ + x_test[j] * per.coef_[0,0]
    yy.append( y_hat )
plt.plot( xx, yy, 'k-' )
plt.show()



















