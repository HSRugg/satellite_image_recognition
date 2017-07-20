import tensorflow as tf
import numpy as np
from skimage import io
import pandas as pd
from tensorport import get_data_path, get_logs_path
import os
from timeit import default_timer as timer




flags = tf.app.flags
FLAGS = flags.FLAGS

# print ('flags init done')

#start of tport snippet 1
#Path to your data locally. This will enable to run the model both locally and on
PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Desktop/compBios/satellite_image_rec/projects/sat_image_proj/logs/')
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/data/Harrison/sat_img_dataset/')
#end of tport snippet 1


#Define the path from the root data directory to your data.
flags.DEFINE_string(
    "train_data_dir",
    get_data_path(
        dataset_name = "Harrison/satimages-1",
        local_root = ROOT_PATH_TO_LOCAL_DATA,
        local_repo = "",
        path = ''
        ),
        "Path to dataset. It is recommended to use get_data_path()"
        "to define your data directory.so that you can switch "
        "from local to tensorport without changing your code."
        "If you set the data directory manually makue sure to use"
        "/data/ as root path when running on TensorPort cloud."
        )
flags.DEFINE_string("logs_dir",
    get_logs_path(root=PATH_TO_LOCAL_LOGS),
    "Path to store logs and checkpoints. It is recommended"
    "to use get_logs_path() to define your logs directory."
    "so that you can switch from local to tensorport without"
    "changing your code."
    "If you set your logs directory manually make sure"
    "to use /logs/ when running on TensorPort cloud.")


print ('flags for path set to:')
print (FLAGS.train_data_dir)


df = pd.read_csv(FLAGS.train_data_dir + 'train_v2.csv')


def weather(tag):
    if 'haze' in(tag):
        return 1
    elif 'partly_cloudy' in(tag):
        return 2
    elif 'cloudy' in(tag):
        return 3
    elif 'clear' in(tag):
        return 4
    elif 'water' in(tag):
        return 5
    else: return ''
    

df['weather'] = df.tags.apply(weather)
df.head()
df.weather.value_counts()
df_weather = pd.get_dummies(df.weather) #gets dummies

df[['1','2','3','4','5']] = df_weather #adds dummies to main df
df = df.reset_index(drop=True) # drops index
df.head()


def get_data(batch_size, df):
    path_to_files = FLAGS.train_data_dir + 'train_jpg/'
    for i in range(batch_size):
        if i < 1:
            batch_data = io.imread(path_to_files+df.image_name[i]+".jpg")
            batch_data = batch_data.flatten()
            
            batch_lables = df[['1','2','3','4','5']][i:i+1]
            
        else:
            data = io.imread(path_to_files+df.image_name[i]+".jpg")
            data = data.flatten()
            
            lables = df[['1','2','3','4','5']][i:i+1]
#             print (lables)
            batch_data   = np.vstack((batch_data,data))
            batch_lables = np.vstack((batch_lables,lables))


    df = df[batch_size:]
    df = df.reset_index(drop=True)
    return batch_data.astype(np.float32), batch_lables.astype(np.float32), df
   



def main(df):
    #start the timer
    start = timer()


    #define placeholder for the flat image and true lables; must be fed 
    x = tf.placeholder(tf.float32, [None, 196608])
    y_ = tf.placeholder(tf.float32, [None, 5])
    
    #define weight and bais Variables; modifyid when training. 
    W = tf.Variable(tf.zeros([196608, 5]))
    b = tf.Variable(tf.zeros([5]))
    
    # Create the model
    y = tf.matmul(x, W) + b

        

    
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    #Initilize Variable and session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    
    # Train
    for i in range(400):
        # get data and lables with the get_data function
        batch_xs, batch_ys, df = get_data(100,df)
        
        #run a training step
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        
        #print compute and print accuracy every ten steps,
        #also prints the rduced mean of the weight tensonsor to varifiy its changing
        #Note: it is better to test accruacy with new data; this does not.
        if i % 10 == 0:
            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run([accuracy,tf.reduce_sum(W),tf.argmax(y, 1)], feed_dict={x: batch_xs,
                                          y_: batch_ys}))
            
            
    print ("done")
    
    #print the total time
    end = timer()
    print("Total load time",end - start)




main(df)