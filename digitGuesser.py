#!/usr/bin/env python3
"""
AI Digit Drawing App
Draw digits with your mouse and get AI predictions in real-time!
"""

# ===============================================================================
# IMPORTS
# ===============================================================================
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# ===============================================================================
# NEURAL NETWORK CLASS (EXACT COPY FROM TRAIN.PY)
# ===============================================================================

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientsWeights = []
        self.gradientsBiases = []
        self.iterations = 0

        # Input to Hidden Layers Network
        # np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            # np.random.seed(0)
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # Hidden Layers Network to Output
        # np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        self.outputs = [inputs]
        self.outputsTesting = ["inputs"]

        for i in range(len(self.weights)):
            # Dot Product to 
            self.outputs.append(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i])
            self.outputsTesting.append("dense")

            # Activation Functions (ReLU + SoftMax)
            if i == len(self.weights)-1:
                finalOutput = np.exp(self.outputs[-1] - np.max(self.outputs[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims=True)
                self.outputs.append(finalOutput)
                self.outputsTesting.append("softmax")
            else:
                self.outputs.append(np.maximum(0, self.outputs[-1]))
                self.outputsTesting.append("relu")
        
        return self.outputs[-1]


class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Digit Predictor - Draw Your Number!")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Drawing settings
        self.logical_size = 28      # 28x28 logical pixels (MNIST size)
        self.display_size = 280     # 280x280 display pixels (10x scale for visibility)
        self.pixel_size = self.display_size // self.logical_size  # 10x10 pixels per logical pixel
        self.brush_size = 1         # Brush size in logical pixels
        
        # Initialize components
        self.setup_ui()
        self.load_model()
        self.reset_drawing()

    # ===============================================================================
    # USER INTERFACE SETUP
    # ===============================================================================
    
    def setup_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title and instructions
        self._setup_title_and_instructions(main_frame)
        
        # Left side - Drawing area
        self._setup_drawing_area(main_frame)
        
        # Right side - Results display
        self._setup_results_area(main_frame)
    
    def _setup_title_and_instructions(self, parent):
        """Setup title and instruction labels"""
        # Title
        title_label = ttk.Label(parent, text="AI Digit Predictor", 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(parent, 
                                text="Draw a digit (0-9) in the pixelated canvas - 28x28 logical pixels displayed large!",
                                font=("Arial", 11))
        instructions.grid(row=1, column=0, columnspan=2, pady=(0, 15))
    
    def _setup_drawing_area(self, parent):
        """Setup the drawing canvas and controls"""
        # Left side frame
        left_frame = ttk.LabelFrame(parent, text="Drawing Area", padding="15")
        left_frame.grid(row=2, column=0, padx=(0, 20), sticky=(tk.N, tk.W))
        
        # Drawing canvas
        self.canvas = tk.Canvas(left_frame, width=self.display_size, height=self.display_size, 
                               bg='white', cursor='pencil', relief=tk.SUNKEN, borderwidth=2)
        self.canvas.grid(row=0, column=0, pady=(0, 15))
        
        # Bind drawing events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.end_draw)
        
        # Control buttons
        self._setup_control_buttons(left_frame)
        
        # Brush size control
        self._setup_brush_control(left_frame)
    
    def _setup_control_buttons(self, parent):
        """Setup control buttons (Clear, Predict)"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, pady=(0, 10))
        
        clear_btn = ttk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_btn.grid(row=0, column=0, padx=(0, 10))
        
        predict_btn = ttk.Button(button_frame, text="Predict Digit!", command=self.predict_digit)
        predict_btn.grid(row=0, column=1)
    
    def _setup_brush_control(self, parent):
        """Setup brush size control slider"""
        brush_frame = ttk.Frame(parent)
        brush_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(brush_frame, text="Brush Size (logical pixels):").grid(row=0, column=0, sticky=tk.W)
        self.brush_slider = tk.Scale(brush_frame, from_=1, to=3, orient=tk.HORIZONTAL, 
                                    command=self.update_brush_size, length=200)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def _setup_results_area(self, parent):
        """Setup the AI prediction results area"""
        # Right side frame
        right_frame = ttk.LabelFrame(parent, text="AI Prediction Results", padding="15")
        right_frame.grid(row=2, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Prediction display
        self._setup_prediction_display(right_frame)
        
        # Probability chart
        self._setup_probability_chart(right_frame)
    
    def _setup_prediction_display(self, parent):
        """Setup the main prediction display area"""
        pred_frame = ttk.Frame(parent)
        pred_frame.grid(row=0, column=0, pady=(0, 20))
        
        ttk.Label(pred_frame, text="Predicted Digit:", font=("Arial", 14)).grid(row=0, column=0)
        
        self.prediction_label = ttk.Label(pred_frame, text="?", font=("Arial", 60, "bold"), 
                                         foreground="blue")
        self.prediction_label.grid(row=1, column=0, pady=10)
        
        self.confidence_label = ttk.Label(pred_frame, text="Draw a digit to get prediction", 
                                         font=("Arial", 12), foreground="gray")
        self.confidence_label.grid(row=2, column=0)
    
    def _setup_probability_chart(self, parent):
        """Setup the probability distribution chart"""
        chart_frame = ttk.Frame(parent)
        chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(chart_frame, text="Probability Distribution", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=(0, 10))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas_chart = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas_chart.get_tk_widget().grid(row=1, column=0)
        
        # Initialize empty chart
        self.update_probability_chart([0.1]*10, -1)

    # ===============================================================================
    # DRAWING FUNCTIONALITY
    # ===============================================================================
    
    def reset_drawing(self):
        """Initialize/reset the drawing canvas"""
        # Create 28x28 logical pixel array
        self.pixel_grid = np.ones((self.logical_size, self.logical_size), dtype=np.uint8) * 255  # White background
        self.last_x = None
        self.last_y = None
    
    def start_draw(self, event):
        """Start drawing when mouse is pressed"""
        self.draw_pixel(event.x, event.y)
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        """Draw line as mouse moves"""
        if self.last_x and self.last_y:
            # Draw line between last position and current position
            self.draw_line_pixels(self.last_x, self.last_y, event.x, event.y)
        else:
            self.draw_pixel(event.x, event.y)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def end_draw(self, event):
        """End drawing when mouse is released"""
        self.last_x = None
        self.last_y = None
    
    def draw_pixel(self, display_x, display_y):
        """Draw a pixel at the given display coordinates"""
        # Convert display coordinates to logical pixel coordinates
        logical_x = min(display_x // self.pixel_size, self.logical_size - 1)
        logical_y = min(display_y // self.pixel_size, self.logical_size - 1)
        
        # Draw brush area in logical pixel grid
        for dy in range(-self.brush_size//2, self.brush_size//2 + 1):
            for dx in range(-self.brush_size//2, self.brush_size//2 + 1):
                px = logical_x + dx
                py = logical_y + dy
                
                if 0 <= px < self.logical_size and 0 <= py < self.logical_size:
                    # Set pixel to black in logical grid
                    self.pixel_grid[py, px] = 0
                    
                    # Draw visual rectangle on canvas
                    x1 = px * self.pixel_size
                    y1 = py * self.pixel_size
                    x2 = x1 + self.pixel_size
                    y2 = y1 + self.pixel_size
                    
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')
    
    def draw_line_pixels(self, x1, y1, x2, y2):
        """Draw a line between two points using logical pixels"""
        # Simple line drawing algorithm
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            self.draw_pixel(x1, y1)
            return
            
        dx = (x2 - x1) / steps
        dy = (y2 - y1) / steps
        
        for i in range(steps + 1):
            x = int(x1 + i * dx)
            y = int(y1 + i * dy)
            self.draw_pixel(x, y)
    
    def update_brush_size(self, value):
        """Update brush size from slider"""
        self.brush_size = int(value)
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.reset_drawing()
        
        # Reset prediction display
        self.prediction_label.config(text="?", foreground="blue")
        self.confidence_label.config(text="Draw a digit to get prediction", foreground="gray")
        self.update_probability_chart([0.1]*10, -1)

    # ===============================================================================
    # NEURAL NETWORK - MODEL LOADING
    # ===============================================================================
    
    def load_model(self):
        """Load the pre-trained neural network weights"""
        try:
            print("Loading AI model...")
            
            # Create neural network with same architecture as training
            self.neural_net = NeuralNetwork(hidden_layers=[128, 128])
            
            # Load trained weights and biases
            for i in range(len(self.neural_net.weights)):
                self.neural_net.weights[i] = np.load(f"trained_weights_{i}.npy")
            for i in range(len(self.neural_net.biases)):
                self.neural_net.biases[i] = np.load(f"trained_biases_{i}.npy")
            
            print(f"Model loaded! Architecture: 784 -> [128, 128] -> 10")
            
        except FileNotFoundError as e:
            error_msg = f"Could not load AI model weights!\n\nMissing file: {e}\n\nPlease run train.py first to generate model weights."
            print(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)
            self.neural_net = None
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)
            self.neural_net = None

    # ===============================================================================
    # NEURAL NETWORK - FORWARD PASS (MATHEMATICAL CORE)
    # ===============================================================================
    
    def forward_pass(self, image_data):
        """
        Run neural network prediction using exact same implementation as training
        
        Architecture: 784 -> [128, 128] -> 10
        Activations: ReLU for hidden layers, Softmax for output
        """
        if self.neural_net is None:
            return np.array([0.1] * 10)  # Return uniform if no model
        
        try:
            # Use the exact same forward method from training
            output = self.neural_net.forward(image_data)
            return output[0]  # Return probabilities for single image
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([0.1] * 10)

    # ===============================================================================
    # NEURAL NETWORK - IMAGE PREPROCESSING
    # ===============================================================================
    
    def _preprocess_image_for_ai(self, pixel_grid):
        """
        Convert drawn image to AI-ready format
        
        Steps:
        1. Convert to float32
        2. Invert colors (MNIST has white digits on black background)
        3. Normalize to [-1, 1] range (same as training data)
        4. Flatten to 1D array (784 elements)
        """
        # Use the logical pixel grid directly - already 28x28!
        image_array = pixel_grid.astype(np.float32)
        
        # Invert colors (MNIST has white digits on black background)
        image_array = 255 - image_array
        
        # Normalize to [-1, 1] range (same as training data)
        image_array = (image_array - 127.5) / 127.5
        
        # Flatten to 1D array for neural network
        image_data = image_array.reshape(1, 784)
        
        return image_data

    # ===============================================================================
    # NEURAL NETWORK - PREDICTION PIPELINE
    # ===============================================================================
    
    def predict_digit(self):
        """
        Complete prediction pipeline:
        1. Preprocess drawn image
        2. Run through neural network
        3. Extract results and update UI
        """
        try:
            print("Making prediction...")
            
            # ===== STEP 1: PREPROCESS IMAGE =====
            image_data = self._preprocess_image_for_ai(self.pixel_grid)
            
            # ===== STEP 2: NEURAL NETWORK FORWARD PASS =====
            probabilities = self.forward_pass(image_data)
            
            # ===== STEP 3: ANALYZE RESULTS =====
            # Debug: Print all probabilities
            print(f"All probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  Digit {i}: {prob:.6f} ({prob:.2%})")
            
            # Check for zero probabilities
            zero_digits = [i for i, p in enumerate(probabilities) if p < 1e-10]
            if zero_digits:
                print(f"  Digits with zero probability: {zero_digits}")
            
            # Find predicted digit and confidence
            predicted_digit = np.argmax(probabilities)
            confidence = probabilities[predicted_digit]
            
            # ===== STEP 4: UPDATE UI =====
            self._update_prediction_ui(predicted_digit, confidence, probabilities)
            
            print(f"Prediction: {predicted_digit} (Confidence: {confidence:.1%})")
            
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            print(error_msg)
            messagebox.showerror("Prediction Error", error_msg)
    
    def _update_prediction_ui(self, predicted_digit, confidence, probabilities):
        """Update the UI with prediction results"""
        # Color code confidence level
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        
        # Update prediction display
        self.prediction_label.config(text=str(predicted_digit), foreground=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}", foreground=color)
        
        # Update probability chart
        self.update_probability_chart(probabilities, predicted_digit)

    # ===============================================================================
    # UI VISUALIZATION - PROBABILITY CHART
    # ===============================================================================
    
    def update_probability_chart(self, probabilities, predicted_digit):
        """Update the probability bar chart with current predictions"""
        self.ax.clear()
        
        digits = list(range(10))
        colors = ['red' if i == predicted_digit else 'lightblue' for i in digits]
        
        bars = self.ax.bar(digits, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        self.ax.set_xlabel('Digit', fontsize=10)
        self.ax.set_ylabel('Probability', fontsize=10)
        self.ax.set_title('AI Confidence for Each Digit', fontsize=11, fontweight='bold')
        self.ax.set_xticks(digits)
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.05:  # Only show labels for probabilities > 5%
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{prob:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self.fig.tight_layout()
        self.canvas_chart.draw()


# ===============================================================================
# MAIN APPLICATION ENTRY POINT
# ===============================================================================

def main():
    """Launch the AI Digit Drawing App"""
    print("Starting AI Digit Drawing App...")
    print("=" * 50)
    
    # Check if model weights exist
    if not os.path.exists("trained_weights_0.npy"):
        print("Warning: Trained model weights not found!")
        print("Please run train.py first to generate model weights.")
    else:
        print("Trained model weights found!")
    
    # Create and run the app
    root = tk.Tk()
    app = DigitDrawingApp(root)
    
    print("App launched! Draw a digit and click 'Predict!' to see AI predictions.")
    print("Tip: Draw thick, clear digits for best results!")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApp closed by user")
    except Exception as e:
        print(f"App error: {e}")


if __name__ == "__main__":
    main()