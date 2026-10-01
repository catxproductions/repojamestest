import numpy as np
import re
import random
import json
import os
from datetime import datetime

class UltimateEngineJamesBot:
    def __init__(self, weights_file="weights.json"):
        self.version = "5.0 (Adaptive Neural AI Engine)"
        self.weights_file = weights_file
        
        # 1. Define vocabulary for text-to-vector conversion
        self.vocabulary = [
            "hey", "hi", "hello", "yo", "sup", "howdy",
            "code", "coding", "python", "ai", "computer", "software", "program", "bot", "script",
            "game", "games", "play", "playing", "xbox", "playstation", "minecraft", "roblox", "pc",
            "school", "math", "science", "class", "subject", "study", "teacher", "homework"
        ]
        self.vocab_size = len(self.vocabulary)
        self.num_classes = 4  # Greetings, Tech, Gaming, School
        self.class_names = ["greetings", "tech", "gaming", "school"]
        self.current_mood_topic = "greetings"
        self.turns_learned = 0

        # ===== UPGRADE 1: EXPANDED HIDDEN LAYER =====
        self.hidden_size = 128
        
        # ===== UPGRADE 2: INITIALIZE WEIGHTS AND BIASES =====
        self.W1 = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * 0.01
        self.b2 = np.zeros((1, self.num_classes))

        # ===== UPGRADE 3: MOMENTUM TRACKING FOR OPTIMIZATION =====
        self.W1_momentum = np.zeros_like(self.W1)
        self.b1_momentum = np.zeros_like(self.b1)
        self.W2_momentum = np.zeros_like(self.W2)
        self.b2_momentum = np.zeros_like(self.b2)
        self.momentum_beta = 0.9  

        # ===== UPGRADE 4: ADAPTIVE LEARNING RATE (Adam-like) =====
        self.learning_rate = 0.01
        self.base_learning_rate = 0.01
        
        self.m_W1 = np.zeros_like(self.W1)  
        self.v_W1 = np.zeros_like(self.W1)  
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_t = 0  

        # ===== UPGRADE 5: CONTEXTUAL WEIGHTED MEMORY =====
        self.context_scores = {"greetings": 0.0, "tech": 0.0, "gaming": 0.0, "school": 0.0}
        self.context_decay_factor = 0.95  
        self.recent_topics = []  
        self.max_recent_length = 10

        self.responses = {
            "greetings": [
                "Hey what is up! Just running some local neural activation loops over here.",
                "Hello again! My weights are perfectly optimized to speak with you today.",
                "Yo! No internet connection required here!",
                "Greetings! My neural pathways are firing at optimal efficiency."
            ],
            "tech": [
                "Coding is brilliant once you realize it is all just matrix multiplication under the hood.",
                "Python scripts running deep learning nets from scratch hit completely different.",
                "The matrix is the foundation of all computation.",
                "Neural networks: where mathematics becomes intelligence.",
                "Every algorithm is just applied linear algebra in disguise."
            ],
            "gaming": [
                "Video games are pure masterclasses in logic routing and physics simulations.",
                "Minecraft is incredible because you can construct fully active redstone networks.",
                "Multiplayer gaming teaches you about distributed systems!",
                "Game design requires understanding complex state management.",
                "Strategic gameplay is essentially real-time decision optimization."
            ],
            "school": [
                "School is a solid proving ground, but engineering real AI parameters is way more exciting.",
                "Mathematics class gets infinitely better when you start applying algebra to cutting-edge neural networks.",
                "Knowledge compounds over time, just like learning iterations.",
                "Education builds the foundation for understanding intelligence.",
                "Every subject connects back to pattern recognition and optimization."
            ]
        }
        
        self.last_seed_source = "Adaptive Neural Network Inference"
        self.last_tier_trace = {}
        
        # ===== UPGRADE 6: TELEMETRY TRACKING =====
        self.error_history = []
        self.gradient_history = []
        self.learning_rate_history = []
        self.max_history_length = 50
        self.cumulative_error = 0.0
        self.last_gradient_magnitude = 0.0

        # ===== UPGRADE 7: LOAD PERSISTENT WEIGHTS IF AVAILABLE =====
        self.load_weights()

    def clean_text(self, text):
        text = text.lower().replace("'", "").replace('"', "")
        return re.sub(r'[^\w\s]', ' ', text).strip()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def text_to_vector(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        vector = np.zeros((1, self.vocab_size))
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary.index(word)
                vector[0, idx] = 1.0
        return vector

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def train_step(self, X, target_class_idx):
        y = np.zeros((1, self.num_classes))
        y[0, target_class_idx] = 1.0

        output = self.forward(X)
        error_output = y - output
        error_magnitude = np.linalg.norm(error_output)
        
        self.error_history.append(float(error_magnitude))
        if len(self.error_history) > self.max_history_length:
            self.error_history.pop(0)
        self.cumulative_error += float(error_magnitude)

        d_output = error_output
        gradient_magnitude = np.linalg.norm(d_output)
        self.gradient_history.append(float(gradient_magnitude))
        if len(self.gradient_history) > self.max_history_length:
            self.gradient_history.pop(0)
        self.last_gradient_magnitude = float(gradient_magnitude)

        error_hidden = np.dot(d_output, self.W2.T)
        relu_derivative = (self.z1 > 0).astype(float)
        d_hidden = error_hidden * relu_derivative

        self.adam_t += 1
        grad_W2 = -np.dot(self.a1.T, d_output)
        grad_b2 = -np.sum(d_output, axis=0, keepdims=True)
        grad_W1 = -np.dot(X.T, d_hidden)
        grad_b1 = -np.sum(d_hidden, axis=0, keepdims=True)

        # Adam W2
        self.m_W2 = self.adam_beta1 * self.m_W2 + (1 - self.adam_beta1) * grad_W2
        self.v_W2 = self.adam_beta2 * self.v_W2 + (1 - self.adam_beta2) * (grad_W2 ** 2)
        m_hat_W2 = self.m_W2 / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_W2 = self.v_W2 / (1 - self.adam_beta2 ** self.adam_t)
        self.W2 += self.learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.adam_epsilon)

        # Adam b2
        self.m_b2 = self.adam_beta1 * self.m_b2 + (1 - self.adam_beta1) * grad_b2
        self.v_b2 = self.adam_beta2 * self.v_b2 + (1 - self.adam_beta2) * (grad_b2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b2 = self.v_b2 / (1 - self.adam_beta2 ** self.adam_t)
        self.b2 += self.learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + self.adam_epsilon)

        # Adam W1
        self.m_W1 = self.adam_beta1 * self.m_W1 + (1 - self.adam_beta1) * grad_W1
        self.v_W1 = self.adam_beta2 * self.v_W1 + (1 - self.adam_beta2) * (grad_W1 ** 2)
        m_hat_W1 = self.m_W1 / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_W1 = self.v_W1 / (1 - self.adam_beta2 ** self.adam_t)
        self.W1 += self.learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.adam_epsilon)

        # Adam b1
        self.m_b1 = self.adam_beta1 * self.m_b1 + (1 - self.adam_beta1) * grad_b1
        self.v_b1 = self.adam_beta2 * self.v_b1 + (1 - self.adam_beta2) * (grad_b1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b1 = self.v_b1 / (1 - self.adam_beta2 ** self.adam_t)
        self.b1 += self.learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + self.adam_epsilon)

    def adapt_learning_rate(self):
        if len(self.error_history) < 5:
            return
        recent_errors = self.error_history[-5:]
        error_trend = recent_errors[-1] - recent_errors[0]
        if error_trend > 0.01:
            self.learning_rate = max(0.0001, self.learning_rate * 0.95)
        elif error_trend < -0.01 and self.learning_rate < 0.1:
            self.learning_rate = min(0.1, self.learning_rate * 1.02)

        self.learning_rate_history.append(self.learning_rate)
        if len(self.learning_rate_history) > self.max_history_length:
            self.learning_rate_history.pop(0)

    def generate_reply(self, user_message):
        X = self.text_to_vector(user_message)
        if np.sum(X) == 0:
            self.current_mood_topic = "greetings"
            self.context_scores = {k: 0.25 for k in self.class_names}
            return "Hey! Type keywords about gaming, tech, or school so my neural weights can activate."

        probabilities = self.forward(X)[0]
        predicted_idx = np.argmax(probabilities)
        self.current_mood_topic = self.class_names[predicted_idx]
for key in self.context_scores:
self.context_scores[key] *= self.context_decay_factor
for i, class_name in enumerate(self.class_names):
self.context_scores[class_name] = (
self.context_scores[class_name] * 0.3 +
float(probabilities[i]) * 0.7
)
self.recent_topics.append(self.current_mood_topic)
if len(self.recent_topics) > self.max_recent_length:
self.recent_topics.pop(0)
self.last_tier_trace = {
"Layer_1_Neurons": self.hidden_size,
"Active_Synapses": self.vocab_size * self.hidden_size,
"Output_Confidence": round(float(probabilities[predicted_idx]), 4)
}
self.train_step(X, predicted_idx)
self.turns_learned += 1
if self.turns_learned % 5 == 0:
self.adapt_learning_rate()
if self.turns_learned % 20 == 0:
self.save_weights()
return random.choice(self.responses[self.current_mood_topic])
def save_weights(self):
try:
weights_data = {
"timestamp": datetime.now().isoformat(),
"turns_learned": self.turns_learned,
"W1": self.W1.tolist(),
"b1": self.b1.tolist(),
"W2": self.W2.tolist(),
"b2": self.b2.tolist(),
"learning_rate": float(self.learning_rate),
"version": self.version
}
with open(self.weights_file, 'w') as f:
json.dump(weights_data, f, indent=2)
except Exception as e:
print(f"Warning: Could not save weights: {e}")
def load_weights(self):
if not os.path.exists(self.weights_file):
return
try:
with open(self.weights_file, 'r') as f:
weights_data = json.load(f)
self.W1 = np.array(weights_data.get("W1", self.W1.tolist()))
self.b1 = np.array(weights_data.get("b1", self.b1.tolist()))
self.W2 = np.array(weights_data.get("W2", self.W2.tolist()))
self.b2 = np.array(weights_data.get("b2", self.b2.tolist()))
self.learning_rate = weights_data.get("learning_rate", self.base_learning_rate)
self.turns_learned = weights_data.get("turns_learned", 0)
print(f"✓ Loaded persistent weights from {self.weights_file}")
except Exception as e:
print(f"Warning: Could not load weights: {e}")
def get_telemetry(self):
avg_error_rate = np.mean(self.error_history) if self.error_history else 0.0
error_trend = self.error_history[-1] - self.error_history[0] if len(self.error_history) >= 2 else 0.0
current_confidence = max(self.context_scores.values()) if self.context_scores else 0.0
active_synapses = int(np.sum(np.abs(self.W1) > 0.001)) + int(np.sum(np.abs(self.W2) > 0.001))
return {
"turns_learned": self.turns_learned,
"active_topic": self.current_mood_topic,
"learning_rate": round(self.learning_rate, 6),
"error_rate": round(float(avg_error_rate), 4),
"error_trend": round(float(error_trend), 4),
"confidence": round(float(current_confidence), 4),
"gradient_magnitude": round(self.last_gradient_magnitude, 4),
"active_synapses": active_synapses,
"hidden_neurons": self.hidden_size,
"cumulative_error": round(float(self.cumulative_error), 4)
}
=====================================================================
INTERACTIVE CLI TERMINAL APP BLOCK
=====================================================================
if name == "main":
# Initialize the engine
bot = UltimateEngineJamesBot()
print("\n" + "="*60)
print(f"  {bot.version} Initialized Successfully!")
print("  Type your message and press Enter.")
print("  Type 'telemetry' to view model stats or 'exit' to quit.")
print("="*60 + "\n")
while True:
try:
user_input = input("You: ").strip()
if not user_input:
continue
if user_input.lower() == 'exit':
print("Saving weights and shutting down. Goodbye!")
bot.save_weights()
break
if user_input.lower() == 'telemetry':
stats = bot.get_telemetry()
print("\n--- NEURAL NETWORK TELEMETRY ---")
for key, val in stats.items():
print(f" {key.replace('_', ' ').title()}: {val}")
print("--------------------------------\n")
continue
# Generate and print the AI reply
reply = bot.generate_reply(user_input)
print(f"AI: {reply}\n")
except KeyboardInterrupt:
print("\nForce quitting... Saving engine weights.")
bot.save_weights()
break
