import tensorflow as tf
import numpy as np

def make_grey(image_data):
    # just taking the average doesn't work well - the resultant greyscale images look bad
    # matlab documentation here: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    # indicates that this is the way to calculate luminance per Rec.ITU-R BT.601-7
    return ((0.2990 * image_data[:,:,:,0] + 0.5870 * image_data[:,:,:,1] + 0.1140 * image_data[:,:,:,2]) ).astype(np.float32)

def preprocess(image_data):
    image_data_min = np.min(image_data)
    image_data_max = np.max(image_data)
    image_data_mean = np.mean(image_data)
    image_data_stddev = np.std(image_data)
    zero_mean = image_data - image_data_mean  #has zero mean
    new_std_dev = 1.0
    stddev_adj = image_data * (new_std_dev / image_data_stddev)
    print("old std dev: ", image_data_stddev)
    print("new std dev: ", np.std(stddev_adj))
    return stddev_adj

tf.set_random_seed(1)

### Define your architecture here.
### Feel free to use as many code cells as needed.

def get_network(x, keep_prob):
    
    mu = 0
    sigma = 0.1

    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0   = tf.contrib.layers.flatten(conv2)
    
    fc0b   = tf.contrib.layers.flatten(conv1)
    
#     fc0c = tf.concat([fc0, fc0b], 1)

    fc0c = fc0

    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*32 + 28*28*6*0, 200), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(200))
    fc1   = tf.matmul(fc0c, fc1_W) + fc1_b

    fc1    = tf.nn.relu(fc1)

    fc2_W  = tf.Variable(tf.truncated_normal(shape=(200, 400), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(400))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    fc2    = tf.nn.relu(fc2)
    
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    fc2B_W = tf.Variable(tf.truncated_normal(shape=(400, 600), mean = mu, stddev = sigma))
    fc2B_b  = tf.Variable(tf.zeros(600))
    fc2B    = tf.matmul(fc2, fc2B_W) + fc2B_b
    
    
    
    fc2C_W = tf.Variable(tf.truncated_normal(shape=(600, 400), mean = mu, stddev = sigma))
    fc2C_b  = tf.Variable(tf.zeros(400))
    fc2C    = tf.matmul(fc2B, fc2C_W) + fc2C_b

    fc3_W  = tf.Variable(tf.truncated_normal(shape=(400, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    
    fc3 = tf.matmul(fc2C, fc3_W) + fc3_b
    
    fc3 = tf.nn.dropout(fc3, keep_prob)
    
    logits = fc3

    return conv1, conv2, logits



from sklearn.utils import shuffle

def calc_accuracy(batch_size, X, y, keep_prob, features, labels, accuracy_operation):
    
    sess = tf.get_default_session()
    
    total_accuracy = 0
    
    for offset in range(0, len(features), batch_size):
        end = offset + batch_size
        batch_x, batch_y = features[offset:end], labels[offset:end]
        accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
        
    nn_accuracy = total_accuracy / len(features)
    
    return nn_accuracy

def train_and_validate(num_epochs, batch_size, train_data, train_labels, valid_data, valid_labels ):

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
    valid_data = np.reshape(valid_data, (valid_data.shape[0], valid_data.shape[1], valid_data.shape[2], 1))

    X = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    
    keep_prob = tf.placeholder(tf.float32)
    one_hot_y = tf.one_hot(y, 43)

    conv1, conv2, logits = get_network(X, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    print('num training examples: ', len(train_data), " ", len(train_labels))
    print('num validation examples: ', len(valid_data), " ", len(valid_labels))
    
    print('training data min/max: ', np.min(train_data), "/", np.max(train_data))
    print('training data min/max: ', np.min(valid_data), "/", np.max(valid_data))
    
    valid_accuracies = []
    train_accuracies = []

    n_train = len(train_labels)
    
    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(num_epochs):
            print("starting epoch {} ...".format(i+1))
            features, labels = shuffle(train_data, train_labels)
            for offset in range(0, n_train, batch_size):
                end = offset + batch_size
                batch_x, batch_y = train_data[offset:end], train_labels[offset:end]
                sess.run(training_operation, feed_dict={X: batch_x, y: batch_y, keep_prob: 0.5})


            #calculate accuracy on the validation set
            print("finished training, testing on validation set...")

            validation_accuracy = calc_accuracy(batch_size, X, y, keep_prob, valid_data, valid_labels, accuracy_operation)
            
            valid_accuracies.append(validation_accuracy)

            print("finished validation set testing, testing on training set...")

            training_accuracy = calc_accuracy(batch_size, X, y, keep_prob, train_data, train_labels, accuracy_operation)
            
            train_accuracies.append(training_accuracy)


            print("finished epoch {} ...".format(i+1))
            print("Validation Accuracy = {:.6f}".format(validation_accuracy))
            print("Training Accuracy = {:.6f}".format(training_accuracy))
            print()
            
        

        saver.save(sess, './traff_sign_classifier')
        print("Model saved")
        
        return valid_accuracies, train_accuracies