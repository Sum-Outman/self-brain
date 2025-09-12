# Copyright 2025 AGI System Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Sub-model specifications data
MODEL_SPECS = [
    {
        "model_id": "A",
        "name": "Language Model",
        "capabilities": "Multilingual processing, sentiment analysis, intent recognition",
        "input_format": "Text/Speech stream",
        "output_format": "Structured JSON",
        "latency": 200,
        "accuracy": 95,
        "error_rate": 1,
        "dataset": "Multilingual corpus (100GB+)",
        "training_cycles": 50,
        "hardware": "4×A100 GPU",
        "metrics": "BLEU, ROUGE, Sentiment accuracy"
    },
    {
        "model_id": "B",
        "name": "Audio Processor",
        "capabilities": "Speech synthesis, noise reduction, voiceprint recognition",
        "input_format": "Audio stream",
        "output_format": "Text/Control commands",
        "latency": 150,
        "accuracy": 92,
        "error_rate": 2,
        "dataset": "Multilingual speech dataset",
        "training_cycles": 40,
        "hardware": "2×A100 GPU",
        "metrics": "WER, CER, MOS"
    },
    {
        "model_id": "C",
        "name": "Image Processor",
        "capabilities": "Object recognition, scene understanding, image generation",
        "input_format": "Image stream/Video frames",
        "output_format": "Feature vectors/Descriptive text",
        "latency": 300,
        "accuracy": 90,
        "error_rate": 3,
        "dataset": "COCO, ImageNet extended",
        "training_cycles": 60,
        "hardware": "4×A100 GPU",
        "metrics": "mAP, IoU, FID"
    },
    {
        "model_id": "D",
        "name": "Video Processor",
        "capabilities": "Action recognition, behavior analysis, video summarization",
        "input_format": "Video stream",
        "output_format": "Structured event descriptions",
        "latency": 500,
        "accuracy": 88,
        "error_rate": 4,
        "dataset": "Kinetics, AVA",
        "training_cycles": 70,
        "hardware": "4×A100 GPU",
        "metrics": "mAP, AR@AN"
    },
    {
        "model_id": "E",
        "name": "Spatial Locator",
        "capabilities": "3D scene reconstruction, motion prediction, spatial relationship reasoning",
        "input_format": "Sensor data/Point clouds",
        "output_format": "3D coordinates/Path planning",
        "latency": 250,
        "accuracy": 93,
        "error_rate": 2,
        "dataset": "Indoor/outdoor 3D scene dataset",
        "training_cycles": 55,
        "hardware": "2×A100 GPU",
        "metrics": "RMSE, Precision@k"
    },
    {
        "model_id": "F",
        "name": "Sensor Processor",
        "capabilities": "Multi-sensor fusion, anomaly detection, environmental modeling",
        "input_format": "Multi-source sensor data streams",
        "output_format": "Environmental status reports",
        "latency": 100,
        "accuracy": 97,
        "error_rate": 0.5,
        "dataset": "Multimodal sensor dataset",
        "training_cycles": 45,
        "hardware": "1×A100 GPU",
        "metrics": "F1-score, MAE"
    },
    {
        "model_id": "G",
        "name": "Computer Controller",
        "capabilities": "Cross-platform control, automated script execution, system monitoring",
        "input_format": "Natural language commands",
        "output_format": "System commands/API calls",
        "latency": 50,
        "accuracy": 99,
        "error_rate": 0.1,
        "dataset": "CLI command dataset",
        "training_cycles": 35,
        "hardware": "CPU intensive",
        "metrics": "Task completion rate, execution time"
    },
    {
        "model_id": "H",
        "name": "Motion Controller",
        "capabilities": "Device control, motion planning, real-time feedback",
        "input_format": "Control commands/Sensor feedback",
        "output_format": "Motor control signals",
        "latency": 20,
        "accuracy": 99.5,
        "error_rate": 0.05,
        "dataset": "Robot motion dataset",
        "training_cycles": 50,
        "hardware": "Real-time CPU+FPGA",
        "metrics": "Trajectory error, response time"
    },
    {
        "model_id": "I",
        "name": "Knowledge Base",
        "capabilities": "Knowledge retrieval, reasoning engine, question-answering system",
        "input_format": "Natural language queries",
        "output_format": "Structured knowledge",
        "latency": 300,
        "accuracy": 85,
        "error_rate": 5,
        "dataset": "Encyclopedic knowledge graph",
        "training_cycles": 100,
        "hardware": "Large memory server",
        "metrics": "MRR, NDCG"
    },
    {
        "model_id": "J",
        "name": "Programming Model",
        "capabilities": "Code generation, algorithm design, program optimization",
        "input_format": "Natural language description",
        "output_format": "Executable code/Algorithm",
        "latency": 400,
        "accuracy": 80,
        "error_rate": 8,
        "dataset": "Open source code repositories",
        "training_cycles": 80,
        "hardware": "2×A100 GPU",
        "metrics": "BLEU, Compilation success rate"
    },
    {
        "model_id": "K",
        "name": "Management Model",
        "capabilities": "Task scheduling, resource allocation, decision optimization",
        "input_format": "System status/Task requirements",
        "output_format": "Management decisions/Execution plans",
        "latency": 1000,
        "accuracy": 95,
        "error_rate": 2,
        "dataset": "Management decision case library",
        "training_cycles": 120,
        "hardware": "Large memory server",
        "metrics": "Decision accuracy, Resource utilization"
    }
]

import json

def generate_specs():
    """Generate model specification documents"""
    output_dir = os.path.join(os.path.dirname(__file__), "specs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main specification file
    main_specs = {
        "version": "1.0",
        "last_updated": "2024-12-19",
        "models": MODEL_SPECS
    }
    
    with open(os.path.join(output_dir, "model_specs.json"), 'w', encoding='utf-8') as f:
        json.dump(main_specs, f, ensure_ascii=False, indent=2)
    
    # Generate individual documents for each model
    for spec in MODEL_SPECS:
        model_id = spec["model_id"]
        with open(os.path.join(output_dir, f"model_{model_id}_spec.json"), 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(MODEL_SPECS)} model specification documents")
    return output_dir

if __name__ == "__main__":
    output_dir = generate_specs()
    print(f"Specification documents saved to: {output_dir}")
