import tensorflow as tf
import numpy as np
from skimage import io
import pandas as pd
from tensorport import get_data_path, get_logs_path
import os



print ('imports susessful')
#Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

print ('flags init done')



try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None

#Path to your data locally. This will enable to run the model both locally and on
# tensorport without changes
PATH_TO_LOCAL_LOGS = '/Users/Harrison/Desktop/compBios/satellite_image_rec/projects/sat_image_proj/logs/'
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/data/Harrison/sat_img_dataset/')
#end of tport snippet 1


#Define the path from the root data directory to your data.
#We use glob to match any .h5 datasets in Documents/comma locally, or in data/ on tensorport
flags.DEFINE_string(
    "train_data_dir",
    get_data_path(
        dataset_name = "Harrison/satimages-1", #all mounted repo
        local_root = ROOT_PATH_TO_LOCAL_DATA,
        local_repo = "", #all repos (we use glob downstream, see read_data.py)
        path = ''#all .h5 files
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

print ('reading csv with pandas susessful')

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

#remove 1st image coz its no good.
df = df[1:]
df_weather = pd.get_dummies(df.weather)

df[['1','2','3','4','5']] = df_weather
df = df.reset_index(drop=True)



def get_data(batch_size, df):
    path_to_files = FLAGS.train_data_dir + 'train_jpg/'
    for i in range(batch_size):
        if i < 1:
            batch_data = io.imread(path_to_files+df.image_name[i]+".jpg")
            batch_lables = df[['1','2','3','4','5']][i:i+1]
        else:
            data = io.imread(path_to_files+df.image_name[i]+".jpg")
            lables = df[['1','2','3','4','5']][i:i+1]
            batch_data   = np.vstack((batch_data,data))
            batch_lables = np.vstack((batch_lables,lables))
    df = df[batch_size:]
    df = df.reset_index(drop=True)
    return batch_data.astype(np.float32), batch_lables.astype(np.float32), df
   

img_h, img_w = 256,256






# Define worker specific environment variables. Handled automatically.
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                    "Worker task index, should be >= 0. task_index=0 is "
                    "the chief worker task the performs the variable "
                    "initialization")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")
#  ---- end of tport snippet 1 ----


def deepnn(x, keep_prob):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
    x: an input tensor with the dimensions (N_examples, 256,256,3)
    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 5)
    keep_prob is a scalar placeholder for the probability of
    dropout.
    """


    # Reshape to use within a convolutional neural net.
    x_image = tf.reshape(x, [-1, img_h, img_w, 3], name='input_iamges')
    tf.summary.image('image', x_image, max_outputs=5)
    W_conv1 = weight_variable([5, 5, 3, 256])
    
        
    b_conv1 = bias_variable([256])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # print(h_conv1.get_shape(),"b_conv1")
    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)
    # print(h_pool1.get_shape(),"pool1")

    tf.summary.scalar('max_W_conv1', tf.reduce_max(W_conv1))
    tf.summary.scalar('min_W_conv1', tf.reduce_min(W_conv1))

    tf.summary.scalar('max_b_conv1', tf.reduce_max(b_conv1))
    tf.summary.scalar('min_b_conv1', tf.reduce_min(b_conv1))

    tf.summary.scalar('max_h_conv1', tf.reduce_max(h_conv1))
    tf.summary.scalar('min_h_conv1', tf.reduce_min(h_conv1))

    tf.summary.scalar('max_h_pool1', tf.reduce_max(h_pool1))
    tf.summary.scalar('min_h_pool1', tf.reduce_min(h_pool1))

    # Second convolutional layer -- maps 256 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 256, 64])

    
    # tf.summary.scalar('max_W_conv2', tf.reduce_max(W_conv2))
    # tf.summary.scalar('min_W_conv2', tf.reduce_min(W_conv2))
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # print(h_conv2.get_shape(),"h_conv2")

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)
    # print(h_pool2.get_shape(),"h_pool2")


    tf.summary.scalar('max_W_conv2', tf.reduce_max(W_conv2))
    tf.summary.scalar('min_W_conv2', tf.reduce_min(W_conv2))
    
    tf.summary.scalar('max_b_conv2', tf.reduce_max(b_conv2))
    tf.summary.scalar('min_b_conv2', tf.reduce_min(b_conv2))
    
    tf.summary.scalar('max_h_conv2', tf.reduce_max(h_conv2))
    tf.summary.scalar('min_h_conv2', tf.reduce_min(h_conv2))
    
    tf.summary.scalar('max_h_pool2', tf.reduce_max(h_pool2))
    tf.summary.scalar('min_h_pool2', tf.reduce_min(h_pool2))

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([2621440/10, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 2621440/10])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    tf.summary.scalar('max_W_fc1', tf.reduce_max(W_fc1))
    tf.summary.scalar('min_W_fc1', tf.reduce_min(W_fc1))
    
    tf.summary.scalar('max_b_fc1', tf.reduce_max(b_fc1))
    tf.summary.scalar('min_b_fc1', tf.reduce_min(b_fc1))
    
    tf.summary.scalar('max_h_conv2', tf.reduce_max(h_conv2))
    tf.summary.scalar('min_h_conv2', tf.reduce_min(h_conv2))
    
    tf.summary.scalar('max_h_pool2', tf.reduce_max(h_pool2))
    tf.summary.scalar('min_h_pool2', tf.reduce_min(h_pool2))
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)




def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow.
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
            "ps": FLAGS.ps_hosts.split(","),
            "worker": FLAGS.worker_hosts.split(","),
    })

    server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
            tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
            server.target,
    )

def main(_):
    global df

    device, target = device_and_target()
    # ---- end of tport snippet 2 ----
    # Create the model
    print ('Starting to train model')

    with tf.device(device):

        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        x = tf.placeholder(np.float32, [None,img_w,3], name="x")
    
        y_labels = tf.placeholder(np.float32, [None, 5], name="y_labels")
    



        # Build the graph for the deep net
        # y_conv, keep_prob = deepnn(x)

        y_conv, keep_prob = deepnn(x, keep_prob)

        global_step = tf.contrib.framework.get_or_create_global_step()
        # cross_entropy = test_var
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_conv))
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_labels, 1))

        # with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        	# tf.summary.scalar('training accuracy', accuracy)
        	# merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('./logs/')    

        init = tf.global_variables_initializer() 

        hooks=[tf.train.StopAtStepHook(num_steps=10)] # Increment number of required training steps
        i = 1

        print(">>>>>>>>FLAG")

        with tf.train.MonitoredTrainingSession(master=target,
            is_chief=(FLAGS.task_index == 0), save_summaries_steps=1,
            checkpoint_dir=FLAGS.logs_dir,hooks = hooks) as sess:

            # print(">>>>>>>>FLAG2")
            while not sess.should_stop():   
                print(">>>>>>>>FLAG3")
                batch_train = get_data(20, df)
                df = batch_train[2]

                print(">>>>>>>>FLAG4")

                # print (type(batch_train[0]),type(batch_train[1]))
                # print ((batch_train[0].shape),(batch_train[1].shape))
                # print(batch_train[1].astype(np.float32))
                feed_dict = {   
                                y_labels : batch_train[1],
                                x : batch_train[0],
                                keep_prob : 0.5
                                
                            }

                print(">>>>>>>>FLAG5")
                variables = [accuracy,train_step]
                #,y_labels,W_conv1,b_conv1,h_conv1
                # print sess.run(variables, feed_dict)
                # print(sess.run(y_labels,feed_dict=feed_dict).shape)
                # print(sess.run(b_conv1,feed_dict=feed_dict).shape)
                # print(sess.run(h_conv1,feed_dict=feed_dict).shape)


                acc, _ = sess.run(variables,feed_dict=feed_dict)
                print (acc)
                

    print ('Main Done')     

            
	        
if __name__ == '__main__':
    tf.app.run(main=main)
