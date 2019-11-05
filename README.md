# Live-feed-object-device-identification-using-Tensorflow-and-OpenCV
How to train a TensorFlow Object Detection Classifier for multiple object detection on Windows
This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows. It was originally written using TensorFlow version 1.4.0, and certain .py files might produce errors with TensorFlow version 2.0.0 since many modules have been replaced and/or removed. 

In order to build an object classification model, you'll need to follow these steps:

 **1.Installing Anaconda and setting up the virtual environment**
  
 **2.Gathering the images which will serve as the dataset for your model and getting all the required files**
  
 **3.Creating labelmaps and configuring training**
  
 **4.Training**
  
 **5.Exporting the inference graph** 
  
 **6.Running the code Object_detection_tutorial.pynb**
 
 A: Please follow the steps in the link attached in order to set up the virtual environment and install tensorflow gpu for better performance. https://www.youtube.com/watch?v=tPq6NIboLSc&t=183s
      *** MAKE SURE YOU USE THE CODE CONDA INSTALL TENSORFLOW-GPU=1.X.X(X can be any value. for eg: tensorflow-fpu=1.14.0 in my case) SINCE SOME MODULES IN THE CODE HAVE BEEN REPLACED IN TENSORFLOW V2.0.0 LIBRARY AND I WILL TRY TO UPDATE IT FOR THE SAME
  
  B: This is where you'll be clicking pictures of the objects you want to classify or getting it from the internet, in my case, I used 4 devices: Potentiometer,Voltmeter,Ammeter,Rheostat.
  Since the first three devices were almost similar, and the only difference was in the dial of the devices, I trained my model in a slightly different manner which I'll get into in the next step.
  Once you're done with this, download the repository from this github link where you'll find all the files required: https://github.com/tensorflow/models/tree/master/research.
Download the model from here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
The model I am using for my purpose is Faster-RCNN-Inception-V2
Compile Protobufs and run setup.py from the command prompt. Head to the research directory and type the following command to convert all the .proto files into a _pb2.py  file:

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

(THIS IS NEEDED TO BE DONE FOR ALL FILES INDIVIDUALLY BECAUSE OF THE UPDATE AND HENCE THE *.PROTO METHOD WON'T WORK)
  
  C: With all the pictures gathered, it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.
https://github.com/tzutalin/labelImg
https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1
Just download the labelIMG tool and run the exe.
Skip to 0:18 of the video in case you have any difficulties in using the LabelIMG tool: https://www.youtube.com/watch?v=_FC6mr7k694 

Now comes the part where I labeled my images.
I classified the images of the three similar devices into 3 buckets:
  1: where we can see the device, from a distant, but the dial is either not visible or is not clear
  2: where the front view of the device is visible, and so is the dial
  3: Only the dial of the device is visible to the camera ( I have uploaded sample pictures for better understanding of this)
 In the first case, I have used the labelIMG tool to labe the device as "Focus on the dial of the device"
 In the THIRD CASE, I have simply classified the dial of the devices
 In the second case, I have labeled the device as "Please focus on the dial for confirmation", and since there will be another recognition from our third case, this totally makes sense because we want to make sure we are classifying the object correctly, and for that, we need to bring the dial of the device as close as possible
  
  D. Now that the images are labeled, we need to convert these to xml files to csv first, and then generate tfrecord.
  Find the files xml_to_csv.py and generate_tfrecord.py. Open the generate_tfrecord.py file and make suitable changes.
  
  For example, say you are training a classifier to detect A, B, and C(A,B and C representing 3 different class labels). You will replace the following code in generate_tfrecord.py:

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == ' ':
        return 1
    elif row_label == '':
        return 2
    elif row_label == '':
        return 3
    elif row_label == '':
        return 4
    elif row_label == '':
        return 5
    elif row_label == '':
        return 6
    else:
        None
With this:

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'A':
        return 1
    elif row_label == 'B':
        return 2
    elif row_label == 'C':
        return 3
    else:
        None
  
  Activate your environment using conda activate gputest and change your directory to your working directory with cd ...\models\research\object_detection in the anaconda command prompt. 
  From the \object_detection folder, execute the following command in the Anaconda command prompt:

(gputest) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py

Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

  E: The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below :

The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the A,B,C detector example mentioned in Step 4, the labelmap.pbtxt file will look like:

item {
  id: 1
  name: 'A'
}

item {
  id: 2
  name: 'B'
}

item {
  id: 3
  name: 'C'
}

Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to C:\tensorflow1\models\research\object_detection\training\configs and copy the faster_rcnn_inception_v2_pets.config, then, open the file with a text editor. Make changes to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

F: From the \object_detection directory, issue the following command to begin training:

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins.
Each step of training reports the loss. It will start high and get lower and lower as training progresses.
 
Let it run for 2-3 hours. You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:

(gputest) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training

This will create a webpage on your local machine at YourPCName:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.

Once the training is stopped, export the inference graph using the command:
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph where XXXX is the model number of the highest valued .ckpt file in your training directory.


type Jupyter Notebook and run the Object_detection_tutorial.py file.
Run the code except for the downloading part. Your live object detector is now ready. 





