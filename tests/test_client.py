"""
Tests for NASaaS client functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from nasaas.client import NASClient, SearchResult
from nasaas.core.nas_engine import SearchStatus
from nasaas.utils.task_parser import TaskType, DataType


class TestNASClient:
    """Test suite for NAS client"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = NASClient()
        
    def test_client_initialization(self):
        """Test client initializes correctly"""
        assert self.client is not None
        assert self.client.mode == "local"
        assert self.client.engine is not None
        assert self.client.task_parser is not None
    
    def test_parse_task_description(self):
        """Test task description parsing"""
        description = "Classify images of cats and dogs with 95% accuracy"
        parsed = self.client.parse_task_description(description)
        
        assert 'task_type' in parsed
        assert 'confidence' in parsed
        assert parsed['task_type'] == TaskType.IMAGE_CLASSIFICATION.value
        assert parsed['confidence'] > 0.5
    
    def test_parse_text_classification_task(self):
        """Test parsing of text classification task"""
        description = "Build a sentiment analysis model for customer reviews"
        parsed = self.client.parse_task_description(description)
        
        assert parsed['task_type'] == TaskType.TEXT_CLASSIFICATION.value
        assert parsed['data_type'] == DataType.TEXT.value
    
    @pytest.mark.asyncio
    async def test_architecture_search(self):
        """Test architecture search functionality"""
        description = "Classify small images quickly"
        
        # Mock the engine to avoid long search times
        with patch.object(self.client.engine, 'start_search') as mock_start, \
             patch.object(self.client.engine, 'get_job_status') as mock_status:
            
            # Setup mocks
            mock_start.return_value = "test-job-id"
            mock_job = Mock()
            mock_job.job_id = "test-job-id"
            mock_job.status = SearchStatus.COMPLETED
            mock_job.progress = 100.0
            mock_job.created_at = "2025-09-18T10:00:00"
            mock_job.updated_at = "2025-09-18T10:05:00"
            mock_job.results = {
                'architecture': {'type': 'CNN', 'layers': 5},
                'evaluation': {'accuracy': 0.95, 'parameters': 1000000}
            }
            mock_job.error = None
            mock_status.return_value = mock_job
            
            # Run search
            result = self.client.search_architecture(
                description=description,
                constraints={"max_latency_ms": 100},
                timeout=10
            )
            
            # Verify results
            assert isinstance(result, SearchResult)
            assert result.status == "completed"
            assert result.progress == 100.0
            assert result.architecture is not None
            assert result.performance is not None
    
    def test_get_engine_statistics(self):
        """Test engine statistics retrieval"""
        stats = self.client.get_engine_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_jobs' in stats
        assert 'success_rate' in stats
        assert stats['total_jobs'] >= 0
        assert 0 <= stats['success_rate'] <= 1
    
    def test_progress_callbacks(self):
        """Test progress callback functionality"""
        callback_calls = []
        
        def test_callback(job_id, status, progress):
            callback_calls.append((job_id, status, progress))
        
        self.client.add_progress_callback(test_callback)
        
        # Trigger a callback through the engine
        self.client.engine._notify_progress(
            "test-id", SearchStatus.SEARCHING, 50.0, "test message"
        )
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "test-id"
        assert callback_calls[0][1] == "searching"
        assert callback_calls[0][2] == 50.0
        
        # Remove callback
        self.client.remove_progress_callback(test_callback)
        
        # Should not trigger anymore
        self.client.engine._notify_progress(
            "test-id-2", SearchStatus.COMPLETED, 100.0
        )
        
        assert len(callback_calls) == 1  # No new calls
    
    def test_list_jobs(self):
        """Test job listing functionality"""
        jobs = self.client.list_jobs()
        assert isinstance(jobs, list)
        
        # Test with status filter
        completed_jobs = self.client.list_jobs(status="completed")
        assert isinstance(completed_jobs, list)
    
    @pytest.mark.parametrize("description,expected_type", [
        ("Classify medical images", TaskType.IMAGE_CLASSIFICATION.value),
        ("Detect objects in photos", TaskType.OBJECT_DETECTION.value),
        ("Analyze sentiment in text", TaskType.TEXT_CLASSIFICATION.value),
        ("Forecast stock prices", TaskType.TIME_SERIES_FORECASTING.value),
        ("Predict house prices from features", TaskType.TABULAR_REGRESSION.value),
    ])
    def test_task_type_detection(self, description, expected_type):
        """Test various task type detections"""
        parsed = self.client.parse_task_description(description)
        assert parsed['task_type'] == expected_type
    
    def test_constraints_parsing(self):
        """Test constraint parsing from descriptions"""
        description = "Build a model with 95% accuracy and under 100ms latency"
        parsed = self.client.parse_task_description(description)
        
        reqs = parsed['performance_requirements']
        assert 'accuracy' in reqs
        assert 'latency' in reqs
        assert reqs['accuracy'] == 95.0
        assert reqs['latency'] == 100.0
    
    def test_hardware_constraints(self):
        """Test hardware constraint detection"""
        description = "Deploy on mobile devices with GPU acceleration"
        parsed = self.client.parse_task_description(description)
        
        constraints = parsed['hardware_constraints']
        assert constraints.get('mobile') is True
        assert constraints.get('gpu') is True


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test SearchResult object creation"""
        result = SearchResult(
            job_id="test-123",
            status="completed",
            progress=100.0,
            architecture={'type': 'CNN'},
            performance={'accuracy': 0.95},
            deployment_info=None,
            created_at="2025-09-18T10:00:00",
            completed_at="2025-09-18T10:05:00"
        )
        
        assert result.job_id == "test-123"
        assert result.status == "completed"
        assert result.progress == 100.0
        assert result.architecture['type'] == 'CNN'
        assert result.performance['accuracy'] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])