'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class FacialLandmarksDetectionModel:
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
                    print("Extension Ineffective")
                    exit(1)
                print("Extension Successful")
            else:
                print("Extension Required")
                exit(1)
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_img = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed_img})
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32) 
        left_minx=coords[0]-10
        left_miny=coords[1]-10
        left_maxx=coords[0]+10
        left_maxy=coords[1]+10
        
        right_minx=coords[2]-10
        right_miny=coords[3]-10
        right_maxx=coords[2]+10
        right_maxy=coords[3]+10
        left_eye =  image[left_miny:left_maxy, left_minx:left_maxx]
        right_eye = image[right_miny:right_maxy, right_minx:right_maxx]
        eye_coords = [[left_minx,left_miny,left_maxx,left_maxy], [right_minx,right_miny,right_maxx,right_maxy]]
        return left_eye, right_eye, eye_coords
        
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
                print("Extension Successful")
            else:
                print("Extension Required")
                exit(1)
         

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        processed_img = np.transpose(np.expand_dims(resized_img,axis=0), (0,3,1,2))
        return processed_img
            

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)
