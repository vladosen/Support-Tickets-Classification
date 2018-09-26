import os
import sys
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
sys.path.append(".")
sys.path.append("..")


#file_path='.datasets//'
file_path='c:\DS\Support-Tickets-Classification\\' # for local testing
# double backslash above in order to avoid	an exceptions because of x07ll_tickets.csv or EOL_in_litiral complains
filename = file_path + 'all_tickets.csv'  
# For the increment testing purpose there are two subsets (All_tickets_short.csv - 1st 1000 rows, All_tickets_short_50.csv - 1st 50 rows)
#filename = file_path + 'All_tickets_short.csv'         
#filename = file_path + 'All_tickets_short_50.csv'

#Load CSV File With Pandas
df_alltickets = pd.read_csv(filename)
df_alltickets=df_alltickets.fillna("VD_NaN") # NaNs are fobidden here (column "type" has planty of them)

column_to_predict = "ticket_type" 
# column "impact" is not bad choice for feature column, but added here mainly to satisfy Pandas DataFrame datatype later
text_columns = ["body", "impact"]

labelData = df_alltickets[column_to_predict]
data = df_alltickets[text_columns]

# DataSet split   (train_data=X_train, train_labels=Y_train, test_data=X_test, test_labels=Y_test)  (shuffle: default=True)
train_data, test_data, train_labels, test_labels = train_test_split(data, labelData, test_size=0.2)

# Input functions
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_data, y=train_labels, batch_size=40, shuffle=True, num_epochs=1)
eval_input_fn  = tf.estimator.inputs.pandas_input_fn(x=test_data,  y=test_labels,  batch_size=40, shuffle=True, num_epochs=1)
pred_input_fn  = tf.estimator.inputs.pandas_input_fn(x=test_data,  batch_size=40, shuffle=True, num_epochs=1)
	
	
	
# Preparation of feature columns ("body", "impact") for usage within estimator
# parameters for tuning:   vocabulary_size, dimension
vocabulary_feature_column_body = tf.feature_column.categorical_column_with_vocabulary_file( 
        key="body", 
        vocabulary_file = filename, 
        vocabulary_size=100)

vec_voc_feature_column_body = tf.feature_column.embedding_column(
        categorical_column=vocabulary_feature_column_body,
        dimension=100)

numeric_feature_column_impact = tf.feature_column.numeric_column(key="impact")

my_feature_columns=[vec_voc_feature_column_body, numeric_feature_column_impact]



# Build the model by premade estimator DNNClassifier ===============================================
  # hidden_units: could/should be tweaked
  # n_classes: We are choosing between two ticket_type classes (0 and 1)
# parameters for tuning:   hidden_units (number of nodes and their sizes)  
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,   
    hidden_units=[10, 10],   
    n_classes=2)


	
# Train the Model ==================================================================================
# parameters for tuning: steps so far
classifier.train(input_fn = train_input_fn, steps=200)


	
# Evaluate the Model ===============================================================================
# parameters for tuning: steps so far
#classifier.evaluate(input_fn = eval_input_fn, steps=200)
eval_result=classifier.evaluate(input_fn = eval_input_fn, steps=200)
print("eval_result:")
print(eval_result)
# CURRENT WARINING: eval_result.accuracy is almost 1.0 even for
#  -entire datasets 
#  -relative low number of epoch (~100)
#  -deliberately used  shuffle in  eval_input_fn (in train_test_split shuffle is True by default)



# Prediction  ======================================================================================
prediction=classifier.predict(input_fn=pred_input_fn, predict_keys=['class_ids'], yield_single_examples=False)
predictions_dict = next(prediction)
# Print only small subsets
# print(predictions_dict)
# for keys,values in predictions_dict.items():
    # print(keys)
    # print("  " + str(values))

	
# for pred_dict, expec in zip(prediction, test_labels):
  # class_id=pred_dict['class_ids'][0]
  # probability=pred_dict['probabilities'][class_id]
  # print ('class_ids=', class_id, ' probabilities=', probability)
	
print("==== Used variables and its data types ====")
print("type(test_labels): "  + str(type(test_labels)) )
print("type(train_labels): "  + str(type(train_labels)) )
print("type(test_data): "  + str(type(test_data)) )
print("type(train_data): "  + str(type(train_data)) )
print("type(vocabulary_feature_column_body): "  + str(type(vocabulary_feature_column_body)) )
print("type(vec_voc_feature_column_body): "  + str(type(vec_voc_feature_column_body)) )
print("type(numeric_feature_column_impact): "  + str(type(numeric_feature_column_impact)) )
print("type(my_feature_columns): "  + str(type(my_feature_columns)) )
print("type(classifier): "  + str(type(classifier)) )
print("type(eval_result): "  + str(type(eval_result)) )
print("type(prediction): "  + str(type(prediction)) )
print("type(predictions_dict): "  + str(type(predictions_dict)) 



#  sklearn.metrics  ========================================================
# CURRENT OBSTACLE: predictions_nparray should be properly converted from prediction|prediction_dict|prediction_list and reshape if it is needed

# prediction_CM=metrics.confusion_matrix(test_labels, predictions_nparray)
# print("Confusion Matrix: " + str(prediction_CM))



# tf.metrics ========================================================
# CURRENT OBSTACLE: predictions_nparray should be properly converted from prediction|prediction_dict|prediction_list and reshape if it is needed

# acc, acc_op = tf.metrics.accuracy(test_labels, predictions_nparray)
# value_tensor_TP, update_op_TP = tf.metrics.true_positives(test_labels, predictions_nparray)
# value_tensor_TN, update_op_TN = tf.metrics.true_negatives(test_labels, predictions_nparray)
# value_tensor_FP, update_op_FP = tf.metrics.false_positives(test_labels, predictions_nparray)
# value_tensor_FN, update_op_FN = tf.metrics.false_negatives(test_labels, predictions_nparray)
# sess = tf.Session()    
# sess.run(tf.local_variables_initializer())    
# print("(TP/TN/FP/FN), update_op_(TP/TN/FP/FN)]):")
# print(sess.run([value_tensor_TP,update_op_TP]))
# print(sess.run([value_tensor_TN,update_op_TN]))
# print(sess.run([value_tensor_FP,update_op_FP]))
# print(sess.run([value_tensor_FN,update_op_FN]))
# print("Accuracy, acc_op]):")
# print(sess.run([acc, acc_op]))
