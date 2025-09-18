#!/usr/bin/env python3
"""
NASaaS Quickstart Example

Demonstrates basic usage of the Neural Architecture Search as a Service system.
"""

import time
from nasaas import NASClient


def main():
    print("=== NASaaS Quickstart Example ===")
    print()
    
    # Initialize the client
    print("Initializing NASaaS client...")
    client = NASClient()
    
    # Add progress callback for real-time updates
    def progress_callback(job_id, status, progress):
        print(f"  Progress: {status} ({progress:.1f}%)")
    
    client.add_progress_callback(progress_callback)
    
    # Example 1: Image Classification
    print("\n1. Image Classification Example")
    print("-" * 40)
    
    task_description = """
    I need to classify images of different dog breeds. 
    The model should achieve 90% accuracy and run efficiently on mobile devices.
    """
    
    # Parse the task to see how it's interpreted
    parsed = client.parse_task_description(task_description)
    print(f"Task type: {parsed['task_type']} (confidence: {parsed['confidence']:.3f})")
    print(f"Performance requirements: {parsed['performance_requirements']}")
    print()
    
    # Run architecture search
    print("Starting DARTS architecture search...")
    result = client.search_architecture(
        description=task_description,
        constraints={
            "max_latency_ms": 150,
            "max_model_size_mb": 25,
            "target_accuracy": 0.90
        },
        algorithm="darts",
        timeout=60  # Short timeout for demo
    )
    
    if result.status == "completed":
        print("\n✓ Search completed successfully!")
        print(f"Architecture: {result.architecture}")
        print(f"Performance: {result.performance}")
        
        # Deploy the model locally
        print("\nDeploying model locally...")
        deployment = client.deploy(result.job_id, platform="local")
        print(f"Model deployed at: {deployment['endpoint_url']}")
        
    else:
        print(f"\n✗ Search failed: {result.error}")
    
    
    # Example 2: Text Classification
    print("\n\n2. Text Classification Example")
    print("-" * 40)
    
    text_task = "Build a sentiment analysis model for customer reviews with high accuracy"
    
    # Parse task
    parsed = client.parse_task_description(text_task)
    print(f"Task type: {parsed['task_type']} (confidence: {parsed['confidence']:.3f})")
    print()
    
    # Run architecture search with different algorithm
    print("Starting ENAS architecture search...")
    result = client.search_architecture(
        description=text_task,
        constraints={
            "target_accuracy": 0.85,
            "max_latency_ms": 100
        },
        algorithm="enas",
        timeout=45
    )
    
    if result.status == "completed":
        print("\n✓ Search completed successfully!")
        print(f"Performance: {result.performance}")
    else:
        print(f"\n✗ Search failed: {result.error}")
    
    
    # Example 3: Show engine statistics
    print("\n\n3. Engine Statistics")
    print("-" * 40)
    
    stats = client.get_engine_statistics()
    print(f"Total jobs run: {stats['total_jobs']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average accuracy: {stats.get('average_accuracy', 0):.3f}")
    
    
    # Example 4: List all jobs
    print("\n\n4. Job History")
    print("-" * 40)
    
    jobs = client.list_jobs()
    for job in jobs:
        print(f"Job {job.job_id[:8]}... | "
              f"Status: {job.status} | "
              f"Progress: {job.progress:.1f}%")
    
    print("\n=== Quickstart Complete ===")


if __name__ == "__main__":
    main()