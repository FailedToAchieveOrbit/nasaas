"""
NASaaS Client API - Simple interface for architecture search
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json

from .core.nas_engine import NASEngine, SearchJob, SearchStatus
from .utils.task_parser import TaskParser
from .deployment.mcp_manager import MCPDeploymentManager


@dataclass
class SearchResult:
    """Represents the result of an architecture search"""
    job_id: str
    status: str
    progress: float
    architecture: Optional[Dict]
    performance: Optional[Dict] 
    deployment_info: Optional[Dict]
    created_at: str
    completed_at: Optional[str]
    error: Optional[str] = None


class NASClient:
    """
    High-level client for NASaaS
    
    Provides a simple, user-friendly interface to the NAS engine
    with both synchronous and asynchronous methods.
    """
    
    def __init__(self, config: Optional[Dict] = None, server_url: Optional[str] = None):
        """
        Initialize NAS client
        
        Args:
            config: Configuration dictionary for local engine
            server_url: URL of remote NASaaS server (for API use)
        """
        self.logger = logging.getLogger(__name__)
        self.server_url = server_url
        
        if server_url:
            self.mode = "remote"
            self.logger.info(f"Initialized remote client for {server_url}")
        else:
            self.mode = "local"
            # Initialize local components
            self.config = config or self._get_default_config()
            self.engine = NASEngine(self.config)
            self.task_parser = TaskParser()
            self.logger.info("Initialized local client")
        
        # Progress callbacks
        self._progress_callbacks: List[Callable] = []
        
        # Setup logging
        self._setup_logging()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for local engine"""
        return {
            'log_level': 'INFO',
            'search': {
                'epochs': 25,  # Reduced for faster demo
                'learning_rate': 0.025,
                'batch_size': 32
            },
            'mcp': {
                'servers': {
                    'aws': {'region': 'us-east-1'},
                    'gcp': {'region': 'us-central1', 'project_id': None},
                    'azure': {'region': 'eastus'},
                    'local': {}
                },
                'default_platform': 'local'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO') if self.mode == 'local' else 'INFO'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def search_architecture(self, description: str, constraints: Dict = None,
                          algorithm: str = "darts", wait: bool = True,
                          timeout: int = 300) -> SearchResult:
        """
        Search for optimal neural architecture
        
        Args:
            description: Natural language description of the task
            constraints: Performance and resource constraints
            algorithm: Search algorithm ('darts', 'enas', 'progressive')
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            SearchResult with architecture and performance information
        """
        self.logger.info(f"Starting architecture search: {description[:100]}...")
        
        if self.mode == "local":
            return self._search_local(description, constraints, algorithm, wait, timeout)
        else:
            return self._search_remote(description, constraints, algorithm, wait, timeout)
    
    def _search_local(self, description: str, constraints: Dict, algorithm: str,
                     wait: bool, timeout: int) -> SearchResult:
        """Execute search using local engine"""
        
        # Parse the task first
        parsed_task = self.task_parser.parse(description)
        self.logger.info(f"Parsed task type: {parsed_task.task_type.value} (confidence: {parsed_task.confidence:.3f})")
        
        # Merge parsed requirements with explicit constraints
        final_constraints = {**parsed_task.performance_requirements, **(constraints or {})}
        
        # Start search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            job_id = loop.run_until_complete(
                self.engine.start_search(description, final_constraints, algorithm)
            )
            
            if not wait:
                return SearchResult(
                    job_id=job_id,
                    status="running",
                    progress=0.0,
                    architecture=None,
                    performance=None,
                    deployment_info=None,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    completed_at=None
                )
            
            # Wait for completion with progress updates
            start_time = time.time()
            last_progress = -1
            
            while time.time() - start_time < timeout:
                job = self.engine.get_job_status(job_id)
                if not job:
                    raise RuntimeError(f"Job {job_id} not found")
                
                # Call progress callbacks if progress changed
                if job.progress != last_progress:
                    for callback in self._progress_callbacks:
                        try:
                            callback(job_id, job.status.value, job.progress)
                        except Exception as e:
                            self.logger.warning(f"Progress callback failed: {e}")
                    last_progress = job.progress
                
                if job.status == SearchStatus.COMPLETED:
                    return SearchResult(
                        job_id=job_id,
                        status="completed",
                        progress=100.0,
                        architecture=job.results.get('architecture') if job.results else None,
                        performance=job.results.get('evaluation') if job.results else None,
                        deployment_info=job.results.get('deployment') if job.results else None,
                        created_at=job.created_at,
                        completed_at=job.updated_at
                    )
                elif job.status == SearchStatus.FAILED:
                    return SearchResult(
                        job_id=job_id,
                        status="failed",
                        progress=job.progress,
                        architecture=None,
                        performance=None,
                        deployment_info=None,
                        created_at=job.created_at,
                        completed_at=job.updated_at,
                        error=job.error
                    )
                
                time.sleep(2)  # Poll every 2 seconds
            
            raise TimeoutError(f"Search timed out after {timeout} seconds")
            
        finally:
            loop.close()
    
    def _search_remote(self, description: str, constraints: Dict, algorithm: str,
                      wait: bool, timeout: int) -> SearchResult:
        """Execute search using remote API (placeholder)"""
        # In a full implementation, this would make HTTP requests to a remote NASaaS server
        raise NotImplementedError("Remote API not implemented in this version")
    
    def get_job_status(self, job_id: str) -> SearchResult:
        """Get current status of a search job"""
        
        if self.mode == "local":
            job = self.engine.get_job_status(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            return SearchResult(
                job_id=job_id,
                status=job.status.value,
                progress=job.progress,
                architecture=job.results.get('architecture') if job.results else None,
                performance=job.results.get('evaluation') if job.results else None,
                deployment_info=job.results.get('deployment') if job.results else None,
                created_at=job.created_at,
                completed_at=job.updated_at if job.status == SearchStatus.COMPLETED else None,
                error=job.error
            )
        else:
            raise NotImplementedError("Remote API not implemented")
    
    def deploy(self, job_id: str, platform: str = "local", 
               config: Dict = None) -> Dict:
        """
        Deploy a completed model
        
        Args:
            job_id: ID of completed search job
            platform: Target platform ('aws', 'gcp', 'azure', 'local')
            config: Platform-specific configuration
            
        Returns:
            Deployment information
        """
        self.logger.info(f"Deploying job {job_id} to {platform}")
        
        if self.mode == "local":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(
                    self.engine.deploy_model(job_id, platform, config or {})
                )
            finally:
                loop.close()
        else:
            raise NotImplementedError("Remote deployment not implemented")
    
    def list_jobs(self, status: Optional[str] = None) -> List[SearchResult]:
        """List all search jobs"""
        
        if self.mode == "local":
            status_filter = SearchStatus(status) if status else None
            jobs = self.engine.list_jobs(status_filter)
            
            return [
                SearchResult(
                    job_id=job.job_id,
                    status=job.status.value,
                    progress=job.progress,
                    architecture=job.results.get('architecture') if job.results else None,
                    performance=job.results.get('evaluation') if job.results else None,
                    deployment_info=job.results.get('deployment') if job.results else None,
                    created_at=job.created_at,
                    completed_at=job.updated_at if job.status == SearchStatus.COMPLETED else None,
                    error=job.error
                )
                for job in jobs
            ]
        else:
            raise NotImplementedError("Remote job listing not implemented")
    
    def add_progress_callback(self, callback: Callable[[str, str, float], None]):
        """Add a callback function for progress updates"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """Remove a progress callback"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get NAS engine statistics"""
        if self.mode == "local":
            return self.engine.get_statistics()
        else:
            raise NotImplementedError("Remote statistics not implemented")
    
    def save_job_results(self, job_id: str, filepath: str):
        """Save job results to file"""
        if self.mode == "local":
            self.engine.save_job_results(job_id, filepath)
        else:
            raise NotImplementedError("Remote save not implemented")
    
    def parse_task_description(self, description: str) -> Dict:
        """Parse a task description and return the parsed specification"""
        if self.mode == "local":
            parsed_task = self.task_parser.parse(description)
            return {
                'task_type': parsed_task.task_type.value,
                'data_type': parsed_task.data_type.value,
                'confidence': parsed_task.confidence,
                'performance_requirements': parsed_task.performance_requirements,
                'hardware_constraints': parsed_task.hardware_constraints,
                'dataset_hints': parsed_task.dataset_hints,
                'num_classes': parsed_task.num_classes,
                'input_shape': parsed_task.input_shape,
                'search_space': parsed_task.search_space
            }
        else:
            raise NotImplementedError("Remote parsing not implemented")


# Convenience functions
def quick_search(description: str, **kwargs) -> SearchResult:
    """
    Convenience function for quick architecture search
    
    Args:
        description: Natural language task description
        **kwargs: Additional arguments passed to search_architecture
        
    Returns:
        SearchResult with found architecture
    """
    client = NASClient()
    return client.search_architecture(description, **kwargs)


def parse_task(description: str) -> Dict:
    """
    Convenience function to parse a task description
    
    Args:
        description: Natural language task description
        
    Returns:
        Parsed task specification
    """
    client = NASClient()
    return client.parse_task_description(description)


# Example usage demonstration
if __name__ == "__main__":
    # Example 1: Simple synchronous search
    print("=== NASaaS Client Demo ===")
    
    client = NASClient()
    
    def progress_callback(job_id, status, progress):
        print(f"Job {job_id}: {status} ({progress:.1f}%)")
    
    client.add_progress_callback(progress_callback)
    
    # Parse a task description
    task_desc = "Classify images of cats and dogs with 95% accuracy for mobile deployment"
    parsed = client.parse_task_description(task_desc)
    print(f"\nParsed task: {parsed['task_type']} (confidence: {parsed['confidence']:.3f})")
    print(f"Performance requirements: {parsed['performance_requirements']}")
    
    # Run architecture search
    print(f"\nStarting architecture search...")
    result = client.search_architecture(
        task_desc,
        constraints={
            "max_latency_ms": 100,
            "max_model_size_mb": 50
        },
        algorithm="darts",
        timeout=60  # Short timeout for demo
    )
    
    if result.status == "completed":
        print(f"\nSearch completed successfully!")
        print(f"Architecture: {result.architecture}")
        print(f"Performance: {result.performance}")
        
        # Deploy the model locally
        print(f"\nDeploying model locally...")
        deployment = client.deploy(result.job_id, platform="local")
        print(f"Model deployed at: {deployment['endpoint_url']}")
    else:
        print(f"\nSearch failed: {result.error}")
    
    # Show engine statistics
    stats = client.get_engine_statistics()
    print(f"\nEngine Statistics:")
    print(f"Total jobs: {stats['total_jobs']}")
    print(f"Success rate: {stats['success_rate']:.2%}")