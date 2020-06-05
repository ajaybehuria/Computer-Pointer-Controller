'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore
import math

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights) 
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding CPU Extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("Extension was Ineffective")
                    exit(1)
                print("CPU Extension Successful")
            else:
                print("CPU Extension Required")
                exit(1)
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

        
    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_proccesed_img, right_proccesed_img = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':left_proccesed_img, 'right_eye_image':right_proccesed_img})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector

    def check_model(self):
         supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
         unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
         if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding CPU Extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("Extension Ineffective")
                    exit(1)
                print("Extension Succesful")
            else:
                print("Extension Required")
                exit(1)
         

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_resized_img = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        right_resized_img = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        left_proccesed_img = np.transpose(np.expand_dims(left_resized_img,axis=0), (0,3,1,2))
        right_proccesed_img = np.transpose(np.expand_dims(right_resized_img,axis=0), (0,3,1,2))
        return left_proccesed_img, right_proccesed_img
            

    def preprocess_output(self, outputs,hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector
        
        
