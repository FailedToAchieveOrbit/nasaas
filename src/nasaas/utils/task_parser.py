"""
Task Parser - Natural Language to ML Task Conversion
Analyzes natural language descriptions and extracts ML task specifications
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


class TaskType(Enum):
    """Enumeration of supported ML task types"""
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection" 
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    LANGUAGE_MODELING = "language_modeling"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TABULAR_REGRESSION = "tabular_regression"
    TABULAR_CLASSIFICATION = "tabular_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Enumeration of data types"""
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ParsedTask:
    """Represents a parsed ML task specification"""
    task_type: TaskType
    data_type: DataType
    description: str
    dataset_hints: List[str]
    num_classes: Optional[int]
    input_shape: Optional[Tuple[int, ...]]
    performance_requirements: Dict[str, Any]
    hardware_constraints: Dict[str, Any]
    search_space: Dict[str, Any]
    confidence: float


class TaskParser:
    """
    Parses natural language task descriptions into structured ML task specifications
    
    Uses NLP techniques and pattern matching to extract:
    - Task type (classification, detection, etc.)
    - Data type and characteristics
    - Performance requirements
    - Hardware constraints
    - Dataset information
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Task type patterns
        self.task_patterns = {
            TaskType.IMAGE_CLASSIFICATION: [
                r'classify.*images?',
                r'image.*classification',
                r'recognize.*objects?',
                r'identify.*(?:cats?|dogs?|animals?|objects?)',
                r'predict.*class.*image',
                r'categorize.*(?:photos?|pictures?|images?)'
            ],
            TaskType.OBJECT_DETECTION: [
                r'detect.*objects?',
                r'object.*detection',
                r'find.*objects?.*image',
                r'locate.*objects?',
                r'bounding.*box',
                r'yolo|rcnn|ssd'
            ],
            TaskType.SEMANTIC_SEGMENTATION: [
                r'segment.*image',
                r'pixel.*classification',
                r'semantic.*segmentation',
                r'mask.*image',
                r'per.*pixel.*prediction'
            ],
            TaskType.TEXT_CLASSIFICATION: [
                r'classify.*text',
                r'text.*classification',
                r'sentiment.*analysis',
                r'spam.*detection',
                r'categorize.*documents?',
                r'nlp.*classification'
            ],
            TaskType.SEQUENCE_TO_SEQUENCE: [
                r'translation',
                r'seq2seq',
                r'sequence.*to.*sequence',
                r'text.*generation',
                r'summarization',
                r'question.*answering'
            ],
            TaskType.TIME_SERIES_FORECASTING: [
                r'forecast.*time.*series',
                r'predict.*future.*values?',
                r'time.*series.*prediction',
                r'stock.*price.*prediction',
                r'sales.*forecasting'
            ],
            TaskType.TABULAR_REGRESSION: [
                r'predict.*(?:price|value|amount)',
                r'regression.*tabular',
                r'continuous.*prediction',
                r'estimate.*numerical'
            ],
            TaskType.TABULAR_CLASSIFICATION: [
                r'classify.*tabular.*data',
                r'predict.*category.*tabular',
                r'classification.*structured.*data'
            ]
        }
        
        # Data type patterns
        self.data_patterns = {
            DataType.IMAGE: [
                r'images?', r'photos?', r'pictures?', r'visual', r'computer.*vision',
                r'cifar|imagenet|coco', r'jpg|jpeg|png', r'pixels?'
            ],
            DataType.TEXT: [
                r'text', r'documents?', r'sentences?', r'words?', r'nlp',
                r'natural.*language', r'corpus', r'strings?'
            ],
            DataType.TABULAR: [
                r'tabular', r'structured.*data', r'csv', r'database',
                r'features?', r'columns?', r'rows?', r'spreadsheet'
            ],
            DataType.TIME_SERIES: [
                r'time.*series', r'temporal.*data', r'sequential.*data',
                r'timestamp', r'over.*time'
            ]
        }
        
        # Performance requirement patterns
        self.performance_patterns = {
            'accuracy': r'(?:accuracy|acc|correct).*?(\d+(?:\.\d+)?)\s*%?',
            'latency': r'(?:latency|inference.*time|response.*time).*?(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?|seconds?|s)',
            'throughput': r'(?:throughput|qps|requests.*per.*second).*?(\d+(?:\.\d+)?)',
            'model_size': r'(?:model.*size|memory|parameters?).*?(\d+(?:\.\d+)?)\s*(?:mb|gb|million|m|billion|b)?'
        }
        
        # Hardware constraint patterns  
        self.hardware_patterns = {
            'mobile': r'mobile|phone|android|ios|edge.*device',
            'gpu': r'gpu|cuda|nvidia|tensor.*core',
            'cpu_only': r'cpu.*only|no.*gpu',
            'low_power': r'low.*power|battery|energy.*efficient',
            'real_time': r'real.*time|live|streaming'
        }
        
        # Dataset hint patterns
        self.dataset_patterns = {
            'cifar': r'cifar[-_]?(?:10|100)',
            'imagenet': r'imagenet',
            'coco': r'coco|common.*objects',
            'imdb': r'imdb|movie.*reviews',
            'custom': r'custom.*dataset|my.*data|own.*data'
        }
    
    def parse(self, description: str) -> ParsedTask:
        """
        Parse a natural language task description
        
        Args:
            description: Natural language description of the ML task
            
        Returns:
            ParsedTask object with extracted specifications
        """
        self.logger.info(f"Parsing task description: {description[:100]}...")
        
        # Clean and normalize text
        clean_desc = self._clean_text(description)
        
        # Extract task type
        task_type, task_confidence = self._extract_task_type(clean_desc)
        
        # Extract data type
        data_type = self._extract_data_type(clean_desc)
        
        # Extract performance requirements
        performance_reqs = self._extract_performance_requirements(clean_desc)
        
        # Extract hardware constraints
        hardware_constraints = self._extract_hardware_constraints(clean_desc)
        
        # Extract dataset hints
        dataset_hints = self._extract_dataset_hints(clean_desc)
        
        # Extract numerical specifications
        num_classes = self._extract_num_classes(clean_desc)
        input_shape = self._extract_input_shape(clean_desc, data_type)
        
        # Generate search space based on task type
        search_space = self._generate_search_space(task_type, data_type, performance_reqs)
        
        parsed_task = ParsedTask(
            task_type=task_type,
            data_type=data_type,
            description=description,
            dataset_hints=dataset_hints,
            num_classes=num_classes,
            input_shape=input_shape,
            performance_requirements=performance_reqs,
            hardware_constraints=hardware_constraints,
            search_space=search_space,
            confidence=task_confidence
        )
        
        self.logger.info(f"Parsed task: {task_type.value}, confidence: {task_confidence:.3f}")
        return parsed_task
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\.\,\%]', ' ', text)
        
        return text.strip()
    
    def _extract_task_type(self, text: str) -> Tuple[TaskType, float]:
        """Extract the primary task type from text"""
        scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            scores[task_type] = score
        
        # Find best match
        if not any(scores.values()):
            return TaskType.UNKNOWN, 0.0
        
        best_task = max(scores, key=scores.get)
        max_score = scores[best_task]
        
        # Calculate confidence
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        return best_task, confidence
    
    def _extract_data_type(self, text: str) -> DataType:
        """Extract the primary data type from text"""
        scores = {}
        
        for data_type, patterns in self.data_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            scores[data_type] = score
        
        # Find best match
        if not any(scores.values()):
            return DataType.IMAGE  # Default assumption
        
        return max(scores, key=scores.get)
    
    def _extract_performance_requirements(self, text: str) -> Dict[str, Any]:
        """Extract performance requirements from text"""
        requirements = {}
        
        for req_type, pattern in self.performance_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the first numeric match
                    value = float(matches[0])
                    requirements[req_type] = value
                except (ValueError, IndexError):
                    continue
        
        return requirements
    
    def _extract_hardware_constraints(self, text: str) -> Dict[str, Any]:
        """Extract hardware constraints from text"""
        constraints = {}
        
        for constraint_type, pattern in self.hardware_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                constraints[constraint_type] = True
        
        return constraints
    
    def _extract_dataset_hints(self, text: str) -> List[str]:
        """Extract dataset hints from text"""
        hints = []
        
        for dataset_name, pattern in self.dataset_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                hints.append(dataset_name)
        
        return hints
    
    def _extract_num_classes(self, text: str) -> Optional[int]:
        """Extract number of classes from text"""
        # Look for explicit class count
        patterns = [
            r'(\d+)\s+classes?',
            r'classify.*into.*(\d+)',
            r'(\d+)\s+categories?',
            r'(\d+)\s+types?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return int(matches[0])
                except (ValueError, IndexError):
                    continue
        
        # Check for common dataset class counts
        if 'cifar-10' in text or 'cifar10' in text:
            return 10
        elif 'cifar-100' in text or 'cifar100' in text:
            return 100
        elif 'imagenet' in text:
            return 1000
        
        return None
    
    def _extract_input_shape(self, text: str, data_type: DataType) -> Optional[Tuple[int, ...]]:
        """Extract input shape information"""
        # Look for explicit shape mentions
        shape_patterns = [
            r'(\d+)x(\d+)(?:x(\d+))?',
            r'shape.*?(\d+),\s*(\d+)(?:,\s*(\d+))?',
            r'(\d+)\s*by\s*(\d+)(?:\s*by\s*(\d+))?'
        ]
        
        for pattern in shape_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    dims = [int(d) for d in matches[0] if d]
                    return tuple(dims)
                except (ValueError, IndexError):
                    continue
        
        # Default shapes based on data type and dataset hints
        if data_type == DataType.IMAGE:
            if 'cifar' in text:
                return (32, 32, 3)
            elif 'imagenet' in text:
                return (224, 224, 3)
            else:
                return (224, 224, 3)  # Default
        
        return None
    
    def _generate_search_space(self, task_type: TaskType, data_type: DataType, 
                             performance_reqs: Dict) -> Dict[str, Any]:
        """Generate appropriate search space based on task characteristics"""
        search_space = {
            'task_type': task_type.value,
            'data_type': data_type.value
        }
        
        # Define operations based on task and data type
        if data_type == DataType.IMAGE:
            search_space['operations'] = [
                'conv_3x3', 'conv_5x5', 'sep_conv_3x3', 'sep_conv_5x5',
                'dilated_conv_3x3', 'max_pool_3x3', 'avg_pool_3x3',
                'skip_connect', 'none'
            ]
            search_space['num_cells'] = [2, 8]
            search_space['init_channels'] = [16, 64]
        
        elif data_type == DataType.TEXT:
            search_space['operations'] = [
                'lstm', 'gru', 'attention', 'transformer_block',
                'conv_1d', 'linear', 'dropout', 'skip_connect'
            ]
            search_space['hidden_dim'] = [128, 1024]
            search_space['num_layers'] = [1, 6]
        
        elif data_type == DataType.TABULAR:
            search_space['operations'] = [
                'linear', 'relu', 'leaky_relu', 'elu', 'gelu',
                'dropout', 'batch_norm', 'skip_connect'
            ]
            search_space['hidden_dims'] = [[32, 64], [512, 1024]]
            search_space['num_layers'] = [2, 8]
        
        # Add constraints based on performance requirements
        if 'model_size' in performance_reqs:
            max_size_mb = performance_reqs['model_size']
            search_space['max_params'] = int(max_size_mb * 1024 * 1024 / 4)  # Assume float32
        
        if 'latency' in performance_reqs:
            max_latency_ms = performance_reqs['latency']
            search_space['max_latency_ms'] = max_latency_ms
        
        # Set number of classes if available
        if task_type in [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_CLASSIFICATION, 
                        TaskType.TABULAR_CLASSIFICATION]:
            search_space['num_classes'] = 10  # Default, will be updated with actual data
        
        return search_space
    
    def to_json(self, parsed_task: ParsedTask) -> str:
        """Convert parsed task to JSON string"""
        task_dict = {
            'task_type': parsed_task.task_type.value,
            'data_type': parsed_task.data_type.value,
            'description': parsed_task.description,
            'dataset_hints': parsed_task.dataset_hints,
            'num_classes': parsed_task.num_classes,
            'input_shape': parsed_task.input_shape,
            'performance_requirements': parsed_task.performance_requirements,
            'hardware_constraints': parsed_task.hardware_constraints,
            'search_space': parsed_task.search_space,
            'confidence': parsed_task.confidence
        }
        
        return json.dumps(task_dict, indent=2)
    
    def from_json(self, json_str: str) -> ParsedTask:
        """Load parsed task from JSON string"""
        task_dict = json.loads(json_str)
        
        return ParsedTask(
            task_type=TaskType(task_dict['task_type']),
            data_type=DataType(task_dict['data_type']),
            description=task_dict['description'],
            dataset_hints=task_dict['dataset_hints'],
            num_classes=task_dict['num_classes'],
            input_shape=tuple(task_dict['input_shape']) if task_dict['input_shape'] else None,
            performance_requirements=task_dict['performance_requirements'],
            hardware_constraints=task_dict['hardware_constraints'],
            search_space=task_dict['search_space'],
            confidence=task_dict['confidence']
        )