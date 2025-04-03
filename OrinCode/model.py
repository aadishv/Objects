import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import sys, os
from PIL import ImageDraw
# Import from the new fixed data processing module
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import common

# Set print options for NumPy, allowing the full array to be printed
np.set_printoptions(threshold=sys.maxsize)

# Define a constant for explicit batch processing
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# Create a logger instance for TensorRT
TRT_LOGGER = trt.Logger()

class Model:
    @staticmethod
    def get_engine(onnx_file_path, engine_file_path=""):
        # Attempts to load a pre-existing TensorRT engine, otherwise builds and returns a new one.

        def build_engine():
            print("Building engine file from onnx, this could take a while")
            # Builds and returns a TensorRT engine from an ONNX file using TensorRT 8+ APIs.
            with trt.Builder(TRT_LOGGER) as builder, \
                    builder.create_network(common.EXPLICIT_BATCH) as network, \
                    builder.create_builder_config() as config, \
                    trt.OnnxParser(network, TRT_LOGGER) as parser, \
                    trt.Runtime(TRT_LOGGER) as runtime:

                # Configure workspace size - 256MB
                config.max_workspace_size = 1 << 28

                # Check if ONNX file exists
                if not os.path.exists(onnx_file_path):
                    print("ONNX file {} not found.".format(onnx_file_path))
                    exit(0)

                # Load and parse the ONNX file
                with open(onnx_file_path, "rb") as model:
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None

                # Set input shape for the network
                network.get_input(0).shape = [1, 3, 320, 320]

                # Build and serialize the network using TensorRT 8+ API
                serialized_engine = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(serialized_engine)
                
                # Save the engine to file
                with open(engine_file_path, "wb") as f:
                    f.write(serialized_engine)
                    
                return engine

        # Check if a serialized engine file exists and load it if so, otherwise build a new one
        if os.path.exists(engine_file_path):
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()


    def __init__(self):
        # Initialize the TensorRT engine and execution context.

        # Define file paths for ONNX and engine files
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        onnx_file_path = os.path.join(current_folder_path, "highstakes_lite.onnx")  # If you change the onnx file to your own model, adjust the file name here
        engine_file_path = os.path.join(current_folder_path, "highstakes_lite_new.trt")  # This should match the .onnx file name

        # Get the TensorRT engine
        self.engine = Model.get_engine(onnx_file_path, engine_file_path)

        # Create an execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers for input and output using the updated allocate_buffers function
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def inference(self, inputImage):
        # Perform inference on the given image and return the bounding boxes, scores, and classes of detected objects.

        # Define input resolution and create preprocessor
        input_resolution_yolov3_HW = (320, 320)
        preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)

        # Process the image and get original shape
        image_raw, image = preprocessor.process(inputImage)
        shape_orig_WH = image_raw.size

        # Get output shapes dynamically from the engine using tensor-centric API
        output_shapes = []
        
        # Find all output tensors by checking tensor mode
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                shape = self.engine.get_tensor_shape(tensor_name)
                output_shapes.append(shape)
            
        # Print debug info about outputs
        print(f"Number of outputs: {len(output_shapes)}")
        for i, shape in enumerate(output_shapes):
            print(f"Output {i} shape: {shape}")

        # Set the input and perform inference using TensorRT 8+ APIs
        self.inputs[0].host = image
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs,
                                            outputs=self.outputs, stream=self.stream)

        # Print actual output sizes for debugging
        print(f"Actual output sizes:")
        for i, output in enumerate(trt_outputs):
            print(f"Output {i} size: {output.size}")

        # Calculate correct output shapes based on actual sizes
        reshaped_outputs = []
        
        # For YOLO outputs, we know the format is typically [batch, anchors*(5+classes), grid_h, grid_w]
        # Where 5 is for [x, y, w, h, obj_conf] and classes is the number of classes
        # For this model, we have 24 channels (3 anchors * (5 + 3 classes))
        channels = 24
        
        for i, output in enumerate(trt_outputs):
            # Calculate grid dimensions based on actual size
            total_elements = output.size
            
            # Extract batch size from engine shape but validate against actual size
            engine_batch = output_shapes[i][0] if i < len(output_shapes) else 32
            
            # Calculate what the grid size should be based on actual elements
            # For batch=1 with channels=24
            elements_per_batch = total_elements // engine_batch
            
            if elements_per_batch % channels == 0:
                # This is likely the correct structure
                grid_points = elements_per_batch // channels
                # Find reasonable h,w grid dimensions
                for grid_h in range(1, int(np.sqrt(grid_points)) + 1):
                    if grid_points % grid_h == 0:
                        grid_w = grid_points // grid_h
                        # Sanity check: Does batch=1 match the expected size?
                        expected_size = 1 * channels * grid_h * grid_w
                        if expected_size == total_elements / engine_batch * engine_batch:
                            print(f"Output {i}: Calculated grid dimensions: {grid_h}x{grid_w}")
                            # Use batch=1 for our processing
                            shape = (1, channels, grid_h, grid_w)
                            try:
                                # Attempt to reshape directly for batch=1
                                if engine_batch == 1:
                                    reshaped = output.reshape(shape)
                                else:
                                    # Need to handle batch size discrepancy
                                    # First reshape with engine's batch size
                                    temp_shape = (engine_batch, channels, grid_h, grid_w)
                                    # Then extract just the first batch
                                    temp = output.reshape(temp_shape)
                                    reshaped = temp[0:1]
                                
                                reshaped_outputs.append(reshaped)
                                print(f"Successfully reshaped output {i} to {shape}")
                                break
                            except Exception as e:
                                print(f"Error during reshaping: {e}")
                                continue
            
            # Fallback to known shapes if calculation fails
            if i >= len(reshaped_outputs):
                print(f"Failed to calculate shape for output {i}, using fallback")
                known_shapes = {
                    307200: (1, 24, 80, 160),    # For 307200 size output
                    1228800: (1, 24, 200, 256),  # For 1228800 size output
                }
                
                if output.size in known_shapes:
                    shape = known_shapes[output.size]
                    try:
                        reshaped = output.reshape(shape)
                        reshaped_outputs.append(reshaped)
                        print(f"Used fallback shape {shape} for output {i}")
                    except ValueError as e:
                        print(f"Even fallback reshape failed: {e}")
                        # Last resort - just reshape to something compatible with post-processing
                        h = int(np.sqrt(output.size / (24 * engine_batch)))
                        w = h
                        while h * w * 24 * engine_batch < output.size:
                            w += 1
                        temp = np.zeros((engine_batch, 24, h, w), dtype=output.dtype)
                        flat_temp = temp.flatten()
                        np.copyto(flat_temp[:min(flat_temp.size, output.size)], 
                                output[:min(flat_temp.size, output.size)])
                        reshaped_outputs.append(temp[0:1])
                        print(f"Created compatible array with shape (1,24,{h},{w})")
        
        # Use reshaped_outputs for post-processing
        trt_outputs = reshaped_outputs

        # Define arguments for post-processing based on the model configuration
        # These should match your model's anchor configuration
        postprocessor_args = {
            "yolo_masks": [(3, 4, 5), (0, 1, 2)],
            "yolo_anchors": [
            (10, 14),
            (23, 27),
            (37, 58),
            (81, 82),
            (135, 169),
            (344, 319),
            ],
            # Lower thresholds to detect more objects (original was 0.5)
            "obj_threshold": [0.25, 0.25, 0.25],  # Lower thresholds for each class
            "nms_threshold": 0.5,
            "yolo_input_resolution": input_resolution_yolov3_HW,
        }

        # Perform post-processing
        postprocessor = PostprocessYOLO(**postprocessor_args)
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

        Detections = []

        # Handle case with no detections
        if boxes is None or classes is None or scores is None:
            print("No objects were detected. Try lowering detection thresholds.")
            return inputImage, Detections

        # Draw bounding boxes and return detected objects
        obj_detected_img = Model.draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES, Detections)
        return np.array(obj_detected_img), Detections

    @staticmethod
    def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, Detections, bbox_color="white"):
        # Draw bounding boxes on the original image and return it.

        # Create drawing context
        draw = ImageDraw.Draw(image_raw)

        print(f"Number of detections: {len(bboxes)}")
        
        # Draw each bounding box
        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

            # Choose color based on class
            if category == 0:
                bbox_color = "green"  # Green class
            elif category == 1:
                bbox_color = "red"    # Red class
            elif category == 2:
                bbox_color = "blue"   # Blue class
            else:
                bbox_color = "white"

            # Always draw bounding boxes for debugging
            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color, width=3)
            draw.text((left, top - 12), "{0} {1:.2f}".format(all_categories[category], score), fill=bbox_color)
            
            print(f"Detection: Class={all_categories[category]}, Score={score:.4f}, Box=[{left},{top},{right},{bottom}]")

            # Create and store the raw detection object
            raw_detection = rawDetection(int(left), int(top), [x_coord, y_coord], int(width), int(height), score,
                                         category)
            Detections.append(raw_detection)

        return image_raw


class rawDetection:
    def __init__(self, x: int, y: int, center: [], width: int, height: int, prob: float, classID: int):
        # Class to store information about a detected object.

        self.x = x
        self.y = y
        self.Center = center
        self.Width = width
        self.Height = height
        self.Prob = prob
        self.ClassID = classID
